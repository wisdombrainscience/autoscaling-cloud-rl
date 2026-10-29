import time, numpy as np, torch
from common import *; from forecasters import *
D=load(); Z=D['Z']; s=D['split']
tr=np.arange(*s['train']); va=np.arange(*s['val'])
for name,m in [('dapm',DAPM(5,L_HIST,H_FC)),('cpuf',CPUForecaster(5,H_FC))]:
    t=time.time(); train_model(m,Z,tr,va,L_HIST,H_FC,epochs=2,patience=99); print(name,(time.time()-t)/2,'s/epoch')
