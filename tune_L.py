import numpy as np, torch
from common import *; from forecasters import *
D=load(); Z=D['Z']; s=D['split']; sd0,mu0=D['sd'][0],D['mu'][0]
tr=np.arange(*s['train']); va=np.arange(*s['val']); te=np.arange(*s['test'])
for L in [24,72,144]:
  for name,mk in [('dapm',lambda: DAPM(5,L,H_FC)),('cpuf',lambda: CPUForecaster(5,H_FC))]:
    res=[]
    for seed in [0,1]:
      m,_,bv=train_model(mk(),Z,tr,va,L,H_FC,seed=seed)
      mu,var=predict(m,Z,va,L,H_FC); _,Y,M=windows(Z,va,L,H_FC)
      res.append((bv, np.mean(np.abs(mu[:,0]-Y.numpy()[:,0]))*sd0))
    print(L,name,'valNLL %.4f valMAE %.3f'%tuple(np.mean(res,0)),flush=True)
