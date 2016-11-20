# coding: utf-8
# Jonathan Vacher, November 2016

import numpy as np
import scipy.io
import imageio
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

class dynTex:
    
    def __init__(self,directory,filename):
        self.directory = directory
        self.filename = filename
        
    def read(self):
        vid = imageio.get_reader(self.directory+self.filename,  'ffmpeg')
        shape = vid.get_meta_data()['size'][::-1]+(3,vid.get_meta_data()['nframes'],)
        self.Ny, self.Nx, _, self.Nf = shape 
        self.mov = np.empty(shape)
        i = 0
        for frame in vid.iter_data():
            self.mov[:,:,:,i] = frame
            i+=1
        
    def colorPCA(self,n):
        X = np.matrix(np.reshape(self.mov[:,:,:,n], (self.Ny*self.Nx,3)))
        Xm = X.mean(axis=0)
        X = X - Xm
        C = X.T*X/(X.shape[0]-1)
        _, _, self.V = np.linalg.svd(C)
    

    def color2pca(self):
        self.Xm = np.zeros((3,self.Nf))
        for i in range(self.Nf):
            X = np.matrix(np.reshape(self.mov[:,:,:,i], (self.Ny*self.Nx,3)))
            self.Xm[:,i] = X.mean(axis=0)
            self.mov[:,:,:,i] = np.reshape( np.array((X - self.Xm[:,i])*self.V.T), (self.Ny, self.Nx, 3))
            
    def pca2color(self, syn=0):
        if syn==0:
            for i in range(self.Nf):
                X = np.matrix(np.reshape(self.mov[:,:,:,i], (self.Ny*self.Nx,3)))
                self.mov[:,:,:,i] = np.reshape( np.array(X*self.V + self.Xm[:,i]) , (self.Ny, self.Nx, 3))
        elif syn==1:
            for i in range(self.movSyn.shape[3]):
                X = np.matrix(np.reshape(self.movSyn[:,:,:,i], (self.Ny*self.Nx,3))) #.mean(axis=1)
                self.movSyn[:,:,:,i] = np.reshape( np.array(X*self.V + self.Xm[:,0]) , (self.Ny, self.Nx, 3))
            self.movSyn[self.movSyn>255]=255  
            self.movSyn[self.movSyn<0]=0
            
    def perComp(self):
        self.movPer = np.empty_like(self.mov, dtype = np.complex64)
        for i in range(self.Nf):
            for k in range(3):
                self.movPer[:,:,k,i] = periodicComp(self.mov[:,:,k,i])



class motionCloud(dynTex):
    
    def __init__(self, directory=None, fileName=None, overSamp = 1, timeOffset = 2,\
                     N = 512, framePerSecond=60, stdConst=35.0, chooseDev=0, show=0):
        dynTex.__init__(self, directory, fileName)
        self.overSamp = overSamp
        self.timeOffset = timeOffset
        self.N = N
        self.framePerSecond = framePerSecond
        self.stdConst = stdConst
        self.dt = 1.0/(self.overSamp*self.framePerSecond)
        self.chooseDev=chooseDev
        self.show=show
    def mcKernel(self, fM, fS, th, thS, fT, v, octa):
        # cycle per image (cpi), NxN px, 65 px/cm, ex: 256x256 px  256/65 = 3.94 cm 
        #   pxnumber / (ppcm * fMode (en c/°) ) = cpi ('freq en px') 
        self.theta = th*np.pi/180
        self.thetaSpread= thS*np.pi/180
        self.fMode = self.N/ppcm*fM 
        self.LifeTime= 1.0/fT
        self.octave = octa
        if self.octave == 1:
            self.fSpread = fS 
            u = np.sqrt(np.exp((self.fSpread/np.sqrt(8)*np.sqrt(np.log(2)))**2)-1)
        elif self.octave == 0:
            self.fSpread = self.N/ppcm*fS #/self.N 
            u=np.roots([1,0,3,0, 3,0,1,0,-self.fSpread**2/self.fMode**2]) # 
            u=u[np.where(np.isreal(u))]
            u=np.real(u[np.where(u>0)])
            u=u[0]

        self.rho=self.fMode*(1.0+u**2)
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
        angular=np.exp(np.cos(2*(Theta-self.theta))/(4*self.thetaSpread**2))
        radial=np.exp(-(np.log(R/self.rho)**2/np.log(1+self.srho**2))/2 )*(1.0/R)
        self.spatialKernel=angular*radial*(1.0/R)**2*4*(oneovertau*self.dt)**3

        # Compute normalization constant
        C = 1.0/np.sum(self.spatialKernel/(4*(oneovertau*self.dt)**3)) 
        self.spatialKernel= self.stdConst*np.sqrt(C*self.spatialKernel)
        
        
    def learnKernel(self, Fmov, dt):
        
        Ny, Nx, Nf = Fmov.shape
    
        # a b estim
        DDFmovDDt=np.diff(Fmov,2)/dt**2
        DFmovDt=np.diff(Fmov,1)/dt

        M1=np.sum(np.absolute(DFmovDt)**2, axis=-1)
        M2=np.sum(np.absolute(Fmov)**2, axis=-1)
        Md=np.sum(np.conj(DFmovDt)*Fmov[:,:,0:Nf-1], axis=-1)

        N1=-np.sum(DDFmovDDt*np.conj(DFmovDt[:,:,0:Nf-2]), axis=-1)
        N2=-np.sum(DDFmovDDt*np.conj(Fmov[:,:,0:Nf-2]), axis=-1)

        a=np.zeros((Ny,Nx), dtype=np.complex64)
        b=np.zeros((Ny,Nx), dtype=np.complex64)

        for i in range(Ny):
            for j in range(Nx):
                if j!=0 or i!=0:
                    A = np.array([ [M1[i,j], Md[i,j]] , [np.conj(Md[i,j]) , M2[i,j]] ])
                    B = np.array([ N1[i,j] , N2[i,j] ])
                    if np.linalg.det(A)==0:
                        x=np.array([0,0])
                    else:
                        x=np.linalg.solve(A, B)  

                    a[i,j]=x[0]
                    b[i,j]=x[1]

        a=np.conj(a)
        b=np.conj(b)
        self.a = a
        self.b = b
        # AR coefficients
        self.al=np.complex64((2-self.dt*a-self.dt**2*b)) #*np.exp(2*np.pi*1j*(v[0]*x/self.N+v[1]*y/self.N))
        self.be=np.complex64((-1+self.dt*a)) #*np.exp(2*np.pi*1j*2*(v[0]*x/self.N+v[1]*y/self.N))
        
        # Spacial kernel
        self.spatialKernel=(np.sum(np.absolute(DDFmovDDt)**2,axis=-1)+np.absolute(a)**2*M1+np.absolute(b)**2*M2+\
                2*np.real(np.conj(a)*(-N1)+np.conj(b)*(-N2)+np.conj(a)*b*Md))/Nf
        
        #delta = np.sqrt(a**2 - 4.0*b)*self.dt
        #r1 = 0.5*(-a*self.dt-delta)
        #r2 = 0.5*(-a*self.dt+delta)
        C = 1.0*a*(a**2-4.0*b) #1.0/(2*np.real(1.0/(np.conj(r1)+(r2))) -1.0/(2.0*np.real(r1))-1.0/(2.0*np.real(r2)))
        self.spatialKernel = self.stdConst*self.dt**(1.5)*np.sqrt( C*self.spatialKernel/ np.sum(self.spatialKernel) ) #
        #self.dt**3*0.5*self.al*(self.al**2-2.0*self.be)* *
        
        
    def initGPU(self):
        #arr_t = Type(np.float32, shape=(N,N))
        carr_t = Type(np.complex64, shape=(self.N,self.N))

        # Initialize random numbers generator
        bij=threefry(32,2)
        samp=normal_bm(bij, np.complex64, mean=0, std=np.sqrt(2)) # to get a real part of std 1
        
        # Initialize functions on GPU (random numbers, fft, recursion, copy)
        rng=CBRNG(carr_t, 1, samp)
        fft = FFT(carr_t) 
        recur = mul_array(rng.parameter.randoms)
        copy = copy_array(rng.parameter.randoms)
        
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
        
        for i in range(200):
            self.getFrame()
                       
        
    def getFrame(self):
        # Noise 
        self.rngc(self.counters_dev, self.w_dev)
        # AR(2) recursion
        self.recurc(self.ITX, self.A, self.F1, self.B, self.F2, self.C, self.w_dev)
        # update values
        self.copyc(self.F2, self.F1)
        self.copyc(self.F1, self.ITX)
        # ifft
        self.fftc(self.TX,self.ITX,0) 
        # get frame from gpu
        TT = np.real(self.TX.get()) 
        
        return TT
    
    def synTex(self, Nf):
        self.movSyn = np.empty((self.N,self.N,3,Nf))
        for i in range(Nf):
            frame = self.getFrame()
            self.movSyn[:,:,0,i]= frame #255.0*(frame-frame.min())/(frame.max()-frame.min())-127.5
    

# compute periodic component

def periodicComp(I):
    
    M, N = I.shape
    I = np.float64(I)
    
    # energy border
    v1 = np.zeros((M,N))
    v1[0,:] = I[M-1,:]-I[0,:]
    v1[M-1,:] = I[0,:]-I[M-1,:]

    v2 = np.zeros((M,N))
    v2[:,0] = I[:,N-1]-I[:,0]
    v2[:,N-1] = I[:,0]-I[:,N-1]

    v=v1+v2

    # compute the discrete laplacian of u
    lapI = -4.0*I
    lapI[:,0:N-1] = lapI[:,0:N-1] + I[:,1:N]
    lapI[:,1:N] = lapI[:,1:N] + I[:,0:N-1]
    lapI[0:M-1,:] = lapI[0:M-1,:] + I[1:M,:]
    lapI[1:M,:] = lapI[1:M,:] + I[0:M-1,:]
    lapI[0,:] = lapI[0,:] +  I[M-1,:]
    lapI[M-1,:] = lapI[M-1,:] + I[0,:]
    lapI[:,0] = lapI[:,0] +  I[:,N-1]
    lapI[:,N-1] = lapI[:,N-1] + I[:,0]

    # Fourier transform of ((lapI) - v) 
    DI_v = np.fft.fft2(lapI-v, norm='ortho')

    #compute the fourier transform of the periodic component
    Lx = np.linspace(0,N-1,N)
    Ly = np.linspace(0,M-1,M)
    X,Y = np.meshgrid(Lx,Ly)

    div = (2.0*np.cos(2*np.pi*X/N)+2.0*np.cos(2*np.pi*Y/M)-4.0)
    div[0,0] = 1.0
    perufft = DI_v/div
    perufft[0,0] = np.sum(I)

    return perufft

    
    
    
















