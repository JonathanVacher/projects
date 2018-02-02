#   Copyright (c) 2016 Jonathan Vacher

# stock fitted likelihood widths [2:7], prior slopes [0:2] and marginal errors here [7:9] 
xx = np.zeros(9)

# set the number of random init for the first step (R0) and second step (R1)
R0 = 2*48
R1 = 2*48

# set bounds to the initialization of variables (2 first are necessary, 2 last are large enough)
dview.execute('bnds = np.array([0.05, 0.05, 3.0, 1.0])')

# measure time
t = time.time()
    
# send data on kernels
dview.push(dict(psych0=psych0[:,:,:nBlock], nBlock=nBlock,\
                options=options, speedSamp=speedSamp, freq=freq))
    
# define obj fun on kernel    
dview.execute('objfun = build_objfun1_Alin(psych0, speedSamp, freq, nBlock)')
    
# initialize results of step 1
res0 = ()
xx0 = np.zeros((4,nF))
    
# step 1: compute initial conditions
for u in range(nF):
    # initialize objective functions of step 1 (indiv spat freq fitting)
    dview.push(dict(u=u))
    dview.execute('objfun0 = build_objfun0(psych0[:,u,:], speedSamp, nBlock)')
    # random init fitting using nelder-mead 
    # improvement is possible by computing the gradient but the problem is still non-convex
    
    def iter_init0(n):
        x0 = np.array([bnds[0]*np.random.rand(), bnds[1]*np.random.rand(),\
                       bnds[2]*(np.random.rand()-1), bnds[3]*np.random.rand()])
        res0 = opt.minimize(objfun0, x0, method='Nelder-Mead', options=options)
        return res0
        
    
    # compute results in parallel (lview.map functions)                
    res0 = res0 + (lview.map(iter_init0, range(R0)),)
    
cond0 = np.array([res0[i].ready() for i in range(nF)]) 

# wait for all parallel results     
while not cond0.prod():
    time.sleep(1)
    clear_output(wait=True)
    p0 = np.array([np.float32(res0[i].progress) for i in range(nF)]).sum()
    print 'Blocks number:', nBlock, '\n Local opt.', '\n Progress:', p0/(nF*R0)*100, '%'
    sys.stdout.flush()
    cond0 = np.array([res0[i].ready() for i in range(nF)]) 
    
# stock the result for the two condtions in xx0 and xx1    
for u in range(nF): 
    fun0 = np.array([res0[u][i]['fun'] for i in range(R0) if res0[u][i]['success']==1 ]) #if res0[i]['success']==1
    xx0[:,u] = res0[u][fun0.argmin()]['x']
    
# solve the linear system that link likelihood widths and prior slopes to threshold and differential PSE    
az00, sz00 = initCond1Alin(xx0, freq)
    
# initialize step 2    
x0 = np.zeros(9)
x0[0:2] = az00
x0[2:7] = np.maximum(sz00, 0.005*np.ones(5)) #np.abs(sz00)
x0[7] = xx0[0,:].mean()
x0[8] = xx0[1,:].mean()

   
# sent initial condition to parallel kernels    
dview.push(dict(x0=x0))

# define random iteration fitting functions (random around the initial conditions above, empirical parameters)
def iter_init00(n):
    x00 = np.zeros(9)
    x00[0:2] = x0[0:2] + 0.05*(np.random.rand(2)-0.5)
    x00[2:7] = np.maximum(x0[2:7]+ 0.05*(np.random.rand(5)-0.5), 0.01*np.ones(5))
    x00[7:9] = x0[7:9]
    res = opt.minimize(objfun, x00, method='Nelder-Mead',options=options) # L-BFGS-B
    return res
  
# run the computations in parallel    
res00 = lview.map(iter_init00, range(R1))

# wait for parallel results
while not res00.ready():
    time.sleep(1)
    clear_output(wait=True)
    print 'Blocks number:', nBlock,'\n Global opt.', '\n Progress:',\
    np.float32(res00.progress)/R1*100, '%'
    sys.stdout.flush()   

# keep the succesful fit    
fun00 = np.array([res00[i]['fun'] for i in range(R1) if res00[i]['success']==1 ]) #if res0[i]['success']==1
res00X = np.array([res00[i]['x'] for i in range(R1) if res00[i]['success']==1 ])

# stock the results in xxx
xx = res00X[fun00.argmin(),:]

# keep a trace of fit results after step 1
az0 = az00
sz0 = sz00

llhValue = fun00.min()


muzz0 = (xx[0]*freq[4]+xx[1])*xx[4]-(xx[0]*freq[2:]+xx[1])*xx[2:7] 
sigzz0 = np.sqrt(xx[2:7] + xx[4])
  
    
elapsed = time.time() - t
print 'Time (s):', elapsed
    