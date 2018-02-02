#   Copyright (c) 2016 Jonathan Vacher

S1 = psyFun(speedSamp[:,np.newaxis], a1, b1, speedSamp[2]+muzz0, sigzz0)
S2 = psyFun(speedSamp[:,np.newaxis], a2, b2, speedSamp[2]+muzz1, sigzz1)

xxBS = np.zeros((18,REP))
muzz0BS = np.zeros((5,REP))
sigzz0BS = np.zeros((5,REP))
muzz1BS = np.zeros((5,REP))
sigzz1BS = np.zeros((5,REP))
llhValueBS = np.zeros(REP)

t = time.time()

for rep in range(REP):
       
    psych00 = np.zeros((nS,nF,nBlock))
    psych11 = np.zeros((nS,nF,nBlock))

    for u in range(nBlock):
        psych00[:,:,u] = np.random.binomial(10, S1, (nS, nF))
        psych11[:,:,u] = np.random.binomial(10, S2, (nS, nF))

    dview.push(dict(psych00=psych00[:,:,:nBlock], psych11=psych11[:,:,:nBlock], nBlock=nBlock,\
                    options=options, speedSamp=speedSamp))

    dview.execute('objfun = build_objfun(psych00, psych11, speedSamp, nBlock)')

    res1 = ()
    res0 = ()
    xx0 = np.zeros((4,nF))
    xx1 = np.zeros((4,nF))

    # compute initial condition
    for u in range(nF):
        dview.push(dict(u=u))
        #objfun0 = build_objfun0(psych0[:,u,:], speedSamp, nBlock)
        dview.execute('objfun0 = build_objfun0(psych00[:,u,:], speedSamp, nBlock)')
        dview.execute('objfun1 = build_objfun0(psych11[:,u,:], speedSamp, nBlock)')    
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
        print 'Rep:', rep+1,'/',REP, '\n Local opt.', '\n Progress:', (p0 + p1)/(2*nF*R0)*100, '%'
        sys.stdout.flush()
        cond0 = np.array([res0[i].ready() for i in range(nF)]) 
        cond1 = np.array([res1[i].ready() for i in range(nF)])
        condP = cond0.prod()*cond1.prod()

    for u in range(nF): 
        fun0 = np.array([res0[u][i]['fun'] for i in range(R0) if res0[u][i]['success']==1 ]) #if res0[i]['success']==1
        fun1 = np.array([res1[u][i]['fun'] for i in range(R0) if res1[u][i]['success']==1 ]) # if res1[i]['success']==1

        xx0[:,u] = res0[u][fun0.argmin()]['x']
        xx1[:,u] = res1[u][fun1.argmin()]['x']

    az00, sz00 = initCond(xx0, xx1, 0.0*freq)


    x0 = np.zeros(18)
    x0[0:7] = az00
    x0[7:14] = np.abs(sz00)
    x0[14] = xx0[0,:].mean()
    x0[15] = xx0[1,:].mean()
    x0[16] = xx1[0,:].mean()
    x0[17] = xx1[1,:].mean()

    dview.push(dict(x0=x0))

    def iter_init(n):
        x00 = np.zeros(18)
        x00[0:7] = x0[0:7] + 0.5*(np.random.rand(7)-0.5)
        x00[7:14] = np.maximum(x0[7:14]+ 0.01*(np.random.rand(7)-0.5), 0.01*np.ones(7))# + (2*np.random.rand(7)-1)*x0[7:14]
        x00[14:18] = x0[14:18]
        res = opt.minimize(objfun, x00, method='Nelder-Mead',options=options)
        return res

    res = lview.map(iter_init, range(R1))

    while not res.ready():
        time.sleep(1)
        clear_output(wait=True)
        print 'Rep:', rep+1,'/',REP,'\n Global opt.', '\n Progress:', np.float32(res.progress)/R1*100, '%'
        sys.stdout.flush()   

    fun = np.array([res[i]['fun'] for i in range(R1) if res[i]['success']==1 ]) #if res0[i]['success']==1
    resX = np.array([res[i]['x'] for i in range(R1) if res[i]['success']==1 ])

    xxBS[:,rep] = resX[fun.argmin(),:]
    az0 = az00
    sz0 = sz00

    llhValueBS[rep] = fun.min()

    muzz0BS[:,rep] = xxBS[2,rep]*xxBS[9,rep]-xxBS[0:5,rep]*xxBS[7:12,rep] 
    sigzz0BS[:,rep] = np.sqrt(xxBS[7:12,rep] + xxBS[9,rep])
    muzz1BS[:,rep] = xxBS[4,rep]*xxBS[11,rep]-xxBS[2:7,rep]*xxBS[9:14,rep] 
    sigzz1BS[:,rep] =  np.sqrt(xxBS[9:14,rep] + xxBS[11,rep])


    elapsed = time.time() - t

    print 'Time (s):', elapsed
