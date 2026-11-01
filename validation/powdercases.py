import os
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=4"
import json
import time
import numpy as np
import pandas as pd
import epraya as epr
CM2MHZ=29979.2458
config=json.load(open('cases.json'))
Conditions=config['defaults']

#Build the system from JSON file
def build(case,backend):
    if backend=='numpy':
        Ham,Exp,_=epr.Start()
    else:
        Ham,Exp,_=epr.Jstart()
    system,units=case['system'],case.get('units',{})
    Ham.S,Ham.I=system['S'],system['I']
    Ham.g=np.asarray(system['g'],float)
    for key in ('A','D','Q'):
        if key in system:
            Ham.__dict__[key]=np.asarray(system[key],float)*(CM2MHZ if units.get(key)=='cm-1' else 1.0)
    Ham.Hpp=system['Hpp']
    expcon={**Conditions['exp'],**case.get('exp',{})}
    Exp.Freq,Exp.Frange,Exp.Points,Exp.Temperature=expcon['Freq'],expcon['Frange'],expcon['Points'],expcon['Temperature']
    return Ham,Exp
    
#For JAX time calculation
def blocktime(out):     
    for x in out:
        if hasattr(x,'block_until_ready'): 
            x.block_until_ready()

#Count first, mean and error time
def timeit(fn,repeats):
    t0=time.perf_counter()
    out=fn()
    blocktime(out)
    #With JIT
    first=time.perf_counter()-t0
    ts=[]
    for _ in range(repeats):
        t0=time.perf_counter() 
        out=fn()
        blocktime(out)
        ts.append(time.perf_counter()-t0)
    return out,first,np.mean(ts),np.std(ts)

#Loads the reference
def loadref(path):
       if path.endswith('.npz'):
           d=np.load(path)
           return {'B': d['B'],'spc': d['spc']}
       a=np.loadtxt(path)                  
       return {'B': a[:,0],'spc': a[:,1]}

#Relative L2 norm 
def km(ref,sim):                      
    k=ref@sim/(sim@sim)
    return k,np.linalg.norm(ref-k*sim)/np.linalg.norm(ref)

rows=[]
for case in config['cases']:
    ref={n: loadref(p) for n, p in case['reference'].items()}
    Ms=Conditions['mesh']['M']
    sims={}
    for backend in Conditions['backends']:
        if backend=='numpy':
            f=epr.Powder  
        else:
            f=epr.JPowder
        for M in Ms:
            Ham, Exp=build(case,backend)
            out,t1,tm,ts=timeit(lambda: f(Ham,Exp,M=M,graph=False),Conditions['timing']['repeats'])
            B,spc=np.asarray(out[0]),np.asarray(out[1])
            assert np.allclose(B,ref['easyspin_ref']['B']),'Different field range'
            sims[(backend,M)]=spc
            rows.append(dict(case=case['id'],group=case['group'],backend=backend,M=M,norient=len(epr.Delaunay(Exp,M)[0]),tfirst=t1,tmean=tm,tstd=ts))
    for r in rows:
        if r['case']!=case['id']: 
            continue
        s=sims[(r['backend'], r['M'])]
        r['k_ref'],r['l2_ref']=km(ref['easyspin_ref']['spc'],s)
        _,r['l2_default']=km(ref['easyspin_default']['spc'],s)
        #Self convergence
        _,r['l2_self']=km(sims[(r['backend'], Ms[-1])], s)
        #Compare numpy with JAX
        if r['backend']=='jax':
            _,r['l2_backend']=km(sims[('numpy',r['M'])],s)
df=pd.DataFrame(rows)
df.to_csv('validation1.csv', index=False)