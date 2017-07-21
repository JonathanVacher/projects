#   Copyright (c) 2016 Jonathan Vacher

## a affine
xx = np.zeros(13)

R0 = 2*48
R1 = 2*48

dview.execute('bnds = np.array([0.05, 0.05, 3.0, 1.0])') 

t = time.time()

#speedSamp = speed10
#%px speedSamp = speed10
    
dview.push(dict(psych0=psych0[:,:,:nBlock], psych1=psych1[:,:,:nBlock], nBlock=nBlock,\
                options=options, speedSamp=speedSamp, freq=freq))
    
dview.execute('objfun = build_objfun_Alin(psych0, psych1, speedSamp, freq, nBlock)')
    
res1 = ()
res0 = ()
xx0 = np.zeros((4,nF))
xx1 = np.zeros((4,nF))
    
# compute initial condition
for u in range(nF):
    dview.push(dict(u=u))
    #objfun0 = build_objfun0(psych0[:,u,:], speedSamp, nBlock)
    dview.execute('objfun0 = build_objfun0(psych0[:,u,:], speedSamp, nBlock)')
    dview.execute('objfun1 = build_objfun0(psych1[:,u,:], speedSamp, nBlock)')    
    def iter_init0(n):
        x0 = np.array([bnds[0]*np.random.rand(), bnds[1]*np.random.rand(),\
                       bnds[2]*(np.random.rand()-1), bnds[3]*np.random.rand()])
        res0 = opt.minimize(objfun0, x0, method='Nelder-Mead', options=options)
        return res0
        
    def iter_init1(n):
        x0 = np.array([bnds[0]*np.random.rand(), bnds[1]*np.random.rand(),\
                       bnds[2]*(np.random.rand()-1), bnds[3]*np.random.rand()])
        res1 = opt.minimize(objfun1, x0, method='Nelder-Mead', options=options)
        return res1
    
                    
    res0 = res0 + (lview.map(iter_init0, range(R0)),)
    res1 = res1 + (lview.map(iter_init1, range(R0)),)
    
cond0 = np.array([res0[i].ready() for i in range(nF)]) 
cond1 = np.array([res1[i].ready() for i in range(nF)])
condP = cond0.prod()*cond1.prod()
    
while not condP :
    time.sleep(1)
    clear_output(wait=True)
    p0 = np.array([np.float32(res0[i].progress) for i in range(nF)]).sum()
    p1 = np.array([np.float32(res1[i].progress) for i in range(nF)]).sum()
    print 'Blocks number:', nBlock, '\n Local opt.', '\n Progress:', (p0 + p1)/(2*nF*R0)*100, '%'
    sys.stdout.flush()
    cond0 = np.array([res0[i].ready() for i in range(nF)]) 
    cond1 = np.array([res1[i].ready() for i in range(nF)])
    condP = cond0.prod()*cond1.prod()
    
for u in range(nF): 
    fun0 = np.array([res0[u][i]['fun'] for i in range(R0) if res0[u][i]['success']==1 ]) #if res0[i]['success']==1
    fun1 = np.array([res1[u][i]['fun'] for i in range(R0) if res1[u][i]['success']==1 ]) # if res1[i]['success']==1
        
    xx0[:,u] = res0[u][fun0.argmin()]['x']
    xx1[:,u] = res1[u][fun1.argmin()]['x']
    
az00, sz00 = initCondAlin(xx0, xx1, freq)
    
    
x0 = np.zeros(13)
x0[0:2] = az00
x0[2:9] = np.abs(sz00)
x0[9] = xx0[0,:].mean()
x0[10] = xx0[1,:].mean()
x0[11] = xx1[0,:].mean()
x0[12] = xx1[1,:].mean()
    
dview.push(dict(x0=x0))
    
def iter_init(n):
    x00 = np.zeros(13)
    x00[0:2] = x0[0:2] + 0.5*(np.random.rand(2)-0.5)
    x00[2:9] = np.maximum(x0[2:9]+ 0.01*(np.random.rand(7)-0.5), 0.01*np.ones(7))# + (2*np.random.rand(7)-1)*x0[7:14]
    x00[9:13] = x0[9:13]
    res = opt.minimize(objfun, x00, method='Nelder-Mead',options=options)
    return res
    
res = lview.map(iter_init, range(R1))

while not res.ready():
    time.sleep(1)
    clear_output(wait=True)
    print 'Blocks number:', nBlock,'\n Global opt.', '\n Progress:', np.float32(res.progress)/R1*100, '%'
    sys.stdout.flush()   
    
fun = np.array([res[i]['fun'] for i in range(R1) if res[i]['success']==1 ]) #if res0[i]['success']==1
resX = np.array([res[i]['x'] for i in range(R1) if res[i]['success']==1 ])

xx = resX[fun.argmin(),:]
az0 = az00
sz0 = sz00

llhValue = fun.min()


muzz0 = (xx[0]*freq[2]+xx[1])*xx[4]-(xx[0]*freq[:5]+xx[1])*xx[2:7] 
sigzz0 = np.sqrt(xx[2:7] + xx[4])

muzz1 = (xx[0]*freq[4]+xx[1])*xx[6]-(xx[0]*freq[2:]+xx[1])*xx[4:9] 
sigzz1 =  np.sqrt(xx[4:9] + xx[6])

    
elapsed = time.time() - t

print 'Time (s):', elapsed
    