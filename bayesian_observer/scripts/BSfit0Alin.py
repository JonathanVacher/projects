#   Copyright (c) 2016 Jonathan Vacher

nBlock = 5

S1 = psyFun(speedSamp[:,np.newaxis], a1, b1, speedSamp[2]+muzz0, sigzz0)

xxBS = np.zeros((9,REP))
muzz0BS = np.zeros((5,REP))
sigzz0BS = np.zeros((5,REP))
llhValueBS = np.zeros(REP)

t = time.time()

for rep in range(REP):
       
    psych00 = np.zeros((nS,nF,nBlock))
    
    for u in range(nBlock):
        psych00[:,:,u] = np.random.binomial(10, S1, (nS, nF))
        
    dview.push(dict(psych00=psych00[:,:,:nBlock], nBlock=nBlock,\
                    options=options, speedSamp=speedSamp, freq=freq))

    dview.execute('objfun = build_objfun1_Alin(psych00, speedSamp, nBlock)')
    
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
        print 'Rep:', rep,'/',REP, '\n Local opt.', '\n Progress:', p0/(nF*R0)*100, '%'
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
    x0[2:7] = np.abs(sz00)
    x0[7] = xx0[0,:].mean()
    x0[8] = xx0[1,:].mean()

    # sent initial condition to parallel kernels    
    dview.push(dict(x0=x0))

    # define random iteration fitting functions (random around the initial conditions above, empirical parameters)
    def iter_init00(n):
        x00 = np.zeros(9)
        x00[0:2] = x0[0:2] + 0.5*(np.random.rand(2)-0.5)
        x00[2:7] = np.maximum(x0[2:7]+ 0.05*(np.random.rand(5)-0.5), 0.05*np.ones(5))
        x00[7:9] = x0[7:9]
        res = opt.minimize(objfun, x00, method='Nelder-Mead',options=options)
        return res

    # run the computations in parallel    
    res00 = lview.map(iter_init00, range(R1))

    # wait for parallel results
    while not res00.ready():
        time.sleep(1)
        clear_output(wait=True)
        print 'Rep:', rep+1,'/',REP,'\n Global opt.', '\n Progress:',\
        np.float32(res00.progress)/R1*100, '%'
        sys.stdout.flush()   

    # keep the succesful fit    
    fun00 = np.array([res00[i]['fun'] for i in range(R1) if res00[i]['success']==1 ]) #if res0[i]['success']==1
    res00X = np.array([res00[i]['x'] for i in range(R1) if res00[i]['success']==1 ])

    # stock the results in xxx
    xxBS[:,rep] = res00X[fun00.argmin(),:]

    # keep a trace of fit results after step 1
    az0 = az00
    sz0 = sz00

    llhValueBS[rep] = fun00.min()


    muzz0BS[:,rep] = (xxBS[0,rep]*freq[4]+xxBS[1,rep])*xxBS[4,rep]-(xxBS[0,rep]*freq[2:]+xxBS[1,rep])*xxBS[2:7,rep] 
    sigzz0BS[:,rep] = np.sqrt(xxBS[2:7,rep] + xxBS[4,rep])


    elapsed = time.time() - t
    print 'Time (s):', elapsed
