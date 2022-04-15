
import numpy as np
import random
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import pickle

import os
os.chdir('/.../code')



from required_func import * 
from main_func import * 



### Simulated Data

# kap = 0.5
Data1 = DataGenerate_HighCond(1000,100,1e-3,0.5,50)
Data2 = DataGenerate_HighCoher(1000,100,1e-3,0.5,50)

# Plot Coherence
U1, _, _ = np.linalg.svd(Data1.Dat, full_matrices=False)      
U2, _, _ = np.linalg.svd(Data2.Dat, full_matrices=False)        
f = plt.figure()
saveFigpath = os.getcwd()+ '/Figure/'+'Coherence.png'
plt.plot(np.sort(np.linalg.norm(U1,2,axis=1)**2),label='low coherence generation',color='red',linestyle=(0,(5,1)))
plt.plot(np.sort(np.linalg.norm(U2,2,axis=1)**2),label='high coherence generation',color='blue',linestyle=(0,()))
# plt.legend(bbox_to_anchor=(0.1,1.05), loc='lower left', borderaxespad=0, fontsize=8,ncol=2)
plt.xlabel(r'$i$',fontsize=25)
plt.ylabel(r'$||U_{(i)}||_2^2$',fontsize=25)
f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

Sketch_Main(Data1)
Sketch_Main(Data2)



# kap = 1
Data3 = DataGenerate_HighCond(1000,100,1e-3,1,50)
Data4 = DataGenerate_HighCoher(1000,100,1e-3,1,50)
Sketch_Main(Data3)
Sketch_Main(Data4)


# kap = 1.5
Data5 = DataGenerate_HighCond(1000,100,1e-3,1.5,50)
Data6 = DataGenerate_HighCoher(1000,100,1e-3,1.5,50)
Sketch_Main(Data5)
Sketch_Main(Data6)







            






