# coding: utf-8
# Jonathan Vacher, Juin 2016

import numpy as np
import scipy.io
import os 
#import cv2

from reikna.cbrng import CBRNG
from reikna.cbrng.bijections import threefry
from reikna.cbrng.samplers import normal_bm
from reikna import cluda
from reikna.cluda import functions, any_api, ocl_api
from reikna.core import Type, Annotation, Parameter
from reikna.algorithms import PureParallel
from reikna.fft import FFT

# define cuda/opencl kernel for the recursion
def mul_array(carr_t1):
    return PureParallel(
            [Parameter('output', Annotation(carr_t1, 'o')),
            Parameter('input1', Annotation(carr_t1,'i')),
            Parameter('input1p', Annotation(carr_t1,'i')),
            Parameter('input2', Annotation(carr_t1,'i')),
            Parameter('input2p', Annotation(carr_t1,'i')),
            Parameter('input3', Annotation(carr_t1,'i')),
            Parameter('input3p', Annotation(carr_t1,'i'))],
            "${output.store_same}(${mul}(${input1.load_same}, ${input1p.load_same}) \
            +${mul}(${input2.load_same}, ${input2p.load_same})+${mul}(${input3.load_same}, ${input3p.load_same}));",
            render_kwds=dict(mul=functions.mul(carr_t1.dtype, carr_t1.dtype, out_dtype=carr_t1.dtype)))

# copy array on the device
def copy_array(carr_t1):
    return PureParallel(
            [Parameter('output', Annotation(carr_t1, 'o')),
            Parameter('input', Annotation(carr_t1,'i'))],
            "${output.store_same}(${input.load_same});")

global ppcm
ppcm = 65.0

class motionCloud:
    
    def __init__(self, overSamp = 1, timeOffset = 2, N = 512, framePerSecond=60, varConst=35.0, chooseDev=0, show=0):
        self.overSamp = overSamp
        self.timeOffset = timeOffset
        self.N = N
        self.framePerSecond = framePerSecond
        self.varConst = varConst
        self.dt = 1.0/(self.overSamp*self.framePerSecond)
        self.chooseDev=chooseDev
        self.show=show
    def mcKernel(self, fM, fS, th, thS, fT, v, octa):
        # cycle per image (cpi), NxN px, 65 px/cm, ex: 256x256 px  256/65 = 3.94 cm 
        # fMode (en c/°) * px / ppcm = cpi ('freq en px') 
        self.theta = th*np.pi/180
        self.thetaSpread= thS*np.pi/180
        self.fMode = fM * self.N/ppcm
        self.fSpread = fS 
        self.LifeTime= 1.0/fT
        self.octave = octa
        if self.octave == 1:
            u = np.sqrt(np.exp((self.fSpread/np.sqrt(8)*np.sqrt(np.log(2)))**2)-1)
        elif self.octave == 0:
            u=np.roots([1,0,3,0, 3,0,1,0,-self.fSpread/self.fMode**2])
            u=u[np.where(np.isreal(u))]
            u=np.real(u[np.where(u>0)])
            u=u[0]

        self.rho=self.fMode*(1+u**2)
        self.srho= u 
        self.sv=1/(self.rho*self.LifeTime)
        if self.show:
            if self.sv >(-2*np.sqrt(2)+4)/(self.N*self.dt):
                print('LifeTime=%f must be greater than %f \n' %(self.LifeTime,((self.N*self.dt)/((-2*np.sqrt(2)+4)*self.rho))) )
            else:
                print('Correct parameters LifeTime = %f > %f \n' % (self.LifeTime,((self.N*self.dt)/((-2*np.sqrt(2)+4)*self.rho))) )

        Lx=np.concatenate((np.linspace(0,self.N/2-1,self.N/2),np.linspace(-self.N/2,-1,self.N/2)))
        x,y=np.meshgrid(Lx,Lx)
        R=np.sqrt(x**2+y**2)
        R[0,0]=10**(-6)
        Theta=np.arctan2(y,x)

        # CAR coefficients
        oneovertau=self.sv*R
        a=2*oneovertau
        b=oneovertau**2

        # AR coefficients
        self.al=np.complex64((2-self.dt*a-self.dt**2*b)*np.exp(2*np.pi*1j*(v[0]*x/self.N+v[1]*y/self.N)))
        self.be=np.complex64((-1+self.dt*a)*np.exp(2*np.pi*1j*2*(v[0]*x/self.N+v[1]*y/self.N)))
        
        # Spacial kernel
        angular=np.exp(np.cos(2*(Theta-self.theta))/self.thetaSpread)
        radial=np.exp(-(np.log(R/self.rho)**2/np.log(1+self.srho**2))/2 )*(self.rho/R)
        self.spatialKernel=angular*radial*(self.rho/R)**2*oneovertau

        # Compute normalization constant
        C=self.dt**3*self.N**2/np.sum(self.spatialKernel/(4*oneovertau**3))
        self.spatialKernel=self.varConst*np.sqrt(C*self.spatialKernel)

    def initGPU(self):
        #arr_t = Type(np.float32, shape=(N,N))
        carr_t = Type(np.complex64, shape=(self.N,self.N))

        # Initialize random numbers generator
        bij=threefry(32,2)
        samp=normal_bm(bij, np.complex64, mean=0, std=1)
        
        # Initialize functions on GPU (random numbers, fft, recursion, copy)
        rng=CBRNG(carr_t, 1, samp)
        fft = FFT(carr_t) 
        recur = mul_array(rng.parameter.randoms)
        copy = copy_array(rng.parameter.randoms)
        
        # Choose gpu/cpu and platform cuda/opencl
        #os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        #PYOPENCL_COMPILER_OUTPUT=1
        
        api = ocl_api()
        dev = api.cl.get_platforms()[0].get_devices()[0]
        #print dev
        if self.chooseDev == 1:
            thr = api.Thread.create(api)
        else:
            thr = api.Thread(dev).create()
                       
        # Compile functions on gpu
        self.fftc = fft.compile(thr)

        counters = rng.create_counters()
        self.counters_dev = thr.to_device(counters)
        self.rngc = rng.compile(thr)

        self.recurc = recur.compile(thr)
        self.copyc = copy.compile(thr)               
                       
        # initialize value on device
        Z = np.zeros((self.N,self.N), dtype=np.complex64)
        self.spatialKernel=self.spatialKernel.astype(np.complex64)

        self.TX = thr.to_device(Z)
        self.ITX = thr.to_device(Z)
        self.w_dev = thr.to_device(Z)

        self.A = thr.to_device(self.al)
        self.B = thr.to_device(self.be)
        self.C = thr.to_device(self.spatialKernel)

        self.F1 = thr.to_device(Z)
        self.F2 = thr.to_device(Z)
                       
        
    def getFrame(self):
        # Noise 
        self.rngc(self.counters_dev, self.w_dev)
        # AR(2) recursion
        self.recurc(self.ITX, self.A, self.F1, self.B, self.F2, self.C, self.w_dev)
        # update values
        self.copyc(self.F2, self.F1)
        self.copyc(self.F1, self.ITX)
        # ifft
        self.fftc(self.TX,self.ITX/self.N,0)
        # get frame from gpu
        TT=0.12*np.real(self.TX.get())/self.varConst+0.5#/(256*512.0) + 0.5
        return TT





















