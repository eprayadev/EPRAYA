import json, time, numpy as np
import epraya as epr

CM2MHZ=29979.2458
base=json.load(open('cases.json'))
cr=json.load(open('crystal_cases.json'))
defaults=base['defaults']

def directions(o):
    """B0 (Fdirection) and B1 (Mwdirection) in the sample frame."""
    t,p,c=np.radians([o['theta'],o['phi'],o['chi']])
    n=np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    et=np.array([np.cos(t)*np.cos(p),np.cos(t)*np.sin(p),-np.sin(t)])
    ep=np.array([-np.sin(p),np.cos(p),0.0])
    return n,np.cos(c)*et+np.sin(c)*ep

def build(case,nucl,ori,backend):
    Ham,Exp,_=epr.Start() if backend=='numpy' else epr.Jstart()
    s,u=case['system'],case.get('units',{})
    Ham.S,Ham.I=s['S'],s.get('I',0)
    Ham.g=np.asarray(s['g'],float)
    for k in ('A','D','Q'):
        if k in s:
            setattr(Ham,k,np.asarray(s[k],float)*(CM2MHZ if u.get(k)=='cm-1' else 1.0))
    b4=s.get('Bk4',case.get('Bk4'))
    if b4 is not None:
        Ham.Bk4=list(map(float,b4))
    Ham.Hpp=list(map(float,s['Hpp']))
    if backend=='numpy' and nucl:
        Ham.Nucl=nucl                                  
    e={**defaults['exp'],**case.get('exp',{})}
    Exp.Freq,Exp.Frange,Exp.Points,Exp.Temperature=e['Freq'],e['Frange'],e['Points'],e['Temperature']
    Exp.Sampleframe = [ori['phi'],ori['theta'],ori['chi']]  
    Exp.Fdirection, Exp.Mwdirection = [0,0,1],[1,0,0]
    return Ham,Exp

def km(ref,sim):
    k=ref@sim/(sim@sim)
    return k,np.linalg.norm(ref-k*sim)/np.linalg.norm(ref)

def simula(case,nucl,ori,backend):
    Ham,Exp=build(case,nucl,ori,backend)
    t0=time.perf_counter()
    if backend=='numpy':
        B,s=epr.Eresonant(Ham,Exp,graph=False,table=False)
    else:
        out=epr.Calresonant(Ham, Exp, nucl or 'None')
        B,s=out[0],out[1]
    return np.asarray(B),np.asarray(s),time.perf_counter()-t0

def cases():
    cal=cr['calibration']
    yield cal, None, cal['orientations']
    for c in base['cases']:
        yield c,cr['nucl'].get(c['id']),cr['orientations']

if __name__=='__main__':
    for case,nucl,oris in cases():
        for o in oris:
            tag=f"{case['id'].replace(' ', '_')}_{o['id']}"
            ref=np.loadtxt(f'cr_{tag}.txt')
            for backend in ('numpy','jax'):
                B,s,t=simula(case,nucl,o,backend)
                assert np.allclose(B,ref[:,0]),f'{tag}: Different field grid'
                k,l2=km(ref[:,1],s)
                np.save(f'sim_{tag}_{backend}.npy', s)
                row=dict(case=case['id'],ori=o['id'],backend=backend,k=float(k),l2_ref=float(l2), t=t)
                with open('validation_cristal2.jsonl', 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(row)