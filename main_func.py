
import numpy as np
import pickle
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from itertools import cycle


import os
os.chdir('/.../code')

from required_func import * 


def Sketch_Main(Data):
    n, d = Data.Dat.shape[0], Data.Dat.shape[1]
    lambd, Rep = Data.lambd, Data.Rep
    LogReg = LogisticRegression(Data.Dat,Data.Resp,Data.lambd)
    _, _ = LogReg.solve_exactly()
    # CREATE FOLDER
    if Data.IdReal == 'real':
        Path = os.getcwd()+ '/Figure/'+Data.name+'/Sketch'
    else:
        if Data.IdCond == 'true':
            Path = os.getcwd()+ '/Figure/'+ 'N'+str(n)+'D'+str(d)+'L'+str(lambd)+'K'+str(Data.kap)+'D/Sketch'
        else:
            Path = os.getcwd()+ '/Figure/'+ 'N'+str(n)+'D'+str(d)+'L'+str(lambd)+'K'+str(Data.kap)+'H/Sketch'
    if not os.path.exists(Path):
        os.makedirs(Path)

    ## SETUP
    Count_EPS, Sket_Size = [1e-6,1e-7], list(map(int,[0.25*d,0.5*d,d,2*d,3*d,5*d]))
    nnz, power = [0.1], [1]
    sketch_func = ['Gaussian','Subsampled','CountSketch','LESS-uniform']

    # Implement BFGS
    BFGS_Err,_,_,_ = LogReg.BFGS()
    BFGS_Log_Err = np.log10(BFGS_Err)
    BFGS_R_Err = BFGS_Err[1:]/BFGS_Err[:-1]    
    BFGS_C_Err1 = sum(BFGS_Err >= Count_EPS[0])
    BFGS_C_Err2 = sum(BFGS_Err >= Count_EPS[1])

    ### Loop over different sketch methods
    for SF in sketch_func:
        ### Loop over sketch size and fraction of non zero entries
        if SF != 'LESS-uniform':
            generator = ((SS_size,NZ) for SS_size in Sket_Size for NZ in [None])
        else:
            generator = ((SS_size,NZ) for SS_size in Sket_Size for NZ in nnz)
        for SS_size, NZ in generator:
            E_Err = [[] for j in range(2+len(power))]
            E_ErrStd = [[] for j in range(2+len(power))]
            Log_Err = [[] for j in range(2+len(power))]
            Log_ErrStd = [[] for j in range(2+len(power))]
            R_Err = [[] for j in range(2+len(power))]
            R_ErrStd = [[] for j in range(2+len(power))]
            C_Err1 = [[] for j in range(2+len(power))]
            C_ErrStd1 = [[] for j in range(2+len(power))]
            C_Err2 = [[] for j in range(2+len(power))]
            C_ErrStd2 = [[] for j in range(2+len(power))]
            # Standard Sketching (without averaging)
            noise_err, noise_log_err, noise_Rerr, Len1, Len2 = [], [], [], 0, 10**3
            for rep in range(Rep):
                print(SF+',size='+str(SS_size)+',NZ='+str(NZ)+',rep='+str(rep))
                Err,_,_,_ = LogReg.sketch_Newton(SS_size,SF,NZ)
                Len1 = max(Len1,len(Err))
                Len2 = min(Len2,len(Err))
                noise_err.append(Err)
                noise_log_err.append(np.log10(Err))
                noise_Rerr.append(Err[1:]/Err[:-1])                                         
            for kk in range(len(noise_err)):
                noise_err[kk] = np.concatenate((noise_err[kk],[noise_err[kk][-1]]*(Len1-len(noise_err[kk]))))
                noise_log_err[kk] = np.concatenate((noise_log_err[kk],[noise_log_err[kk][-1]]*(Len1-len(noise_log_err[kk]))))                
                noise_Rerr[kk] = noise_Rerr[kk][:Len2-1]
            noise_err = np.vstack(noise_err)
            noise_log_err = np.vstack(noise_log_err)
            noise_Rerr = np.vstack(noise_Rerr)
            E_Err[0] = np.median(noise_err, axis=0)
            E_ErrStd[0] = np.std(noise_err, axis=0)
            Log_Err[0] = np.median(noise_log_err, axis=0)
            Log_ErrStd[0] = np.std(noise_log_err, axis=0)
            R_Err[0] = np.mean(noise_Rerr, axis=0)
            R_ErrStd[0] = np.std(noise_Rerr, axis=0)           
            C_Err1[0] = np.median(np.sum(noise_err>=Count_EPS[0],1))
            C_ErrStd1[0] = np.std(np.sum(noise_err>=Count_EPS[0],1))
            C_Err2[0] = np.median(np.sum(noise_err>=Count_EPS[1],1))
            C_ErrStd2[0] = np.std(np.sum(noise_err>=Count_EPS[1],1))

                
            # Sketching with power averaging sequence
            for ii, pp in enumerate(power):
                noise_err, noise_log_err, noise_Rerr, Len1, Len2 = [], [], [], 0, 10**3
                for rep in range(Rep):
                    print(SF+',size='+str(SS_size)+',NZ='+str(NZ)+',power'\
                          +',p='+str(pp)+',rep='+str(rep))
                    Err,_,_,_ = LogReg.sto_weight_Sket_Newton(SS_size,'power',pp,SF,NZ)
                    Len1 = max(Len1,len(Err))
                    Len2 = min(Len2,len(Err))
                    noise_err.append(Err)
                    noise_log_err.append(np.log10(Err))
                    noise_Rerr.append(Err[1:]/Err[:-1])
                for kk in range(len(noise_err)):
                    noise_err[kk] = np.concatenate((noise_err[kk],[noise_err[kk][-1]]*(Len1-len(noise_err[kk]))))
                    noise_log_err[kk] = np.concatenate((noise_log_err[kk],[noise_log_err[kk][-1]]*(Len1-len(noise_log_err[kk]))))                
                    noise_Rerr[kk] = noise_Rerr[kk][:Len2-1]
                noise_err = np.vstack(noise_err)
                noise_log_err = np.vstack(noise_log_err)
                noise_Rerr = np.vstack(noise_Rerr)
                E_Err[ii+1] = np.median(noise_err, axis=0)
                E_ErrStd[ii+1] = np.std(noise_err, axis=0)
                Log_Err[ii+1] = np.median(noise_log_err, axis=0)
                Log_ErrStd[ii+1] = np.std(noise_log_err, axis=0)
                R_Err[ii+1] = np.mean(noise_Rerr, axis=0)
                R_ErrStd[ii+1] = np.std(noise_Rerr, axis=0)           
                C_Err1[ii+1] = np.median(np.sum(noise_err>=Count_EPS[0],1))
                C_ErrStd1[ii+1] = np.std(np.sum(noise_err>=Count_EPS[0],1))
                C_Err2[ii+1] = np.median(np.sum(noise_err>=Count_EPS[1],1))
                C_ErrStd2[ii+1] = np.std(np.sum(noise_err>=Count_EPS[1],1))

            # Sketching with log_power averaging sequence
            noise_err, noise_log_err, noise_Rerr, Len1, Len2 = [], [], [], 0, 10**3
            for rep in range(Rep):
                print(SF+',size='+str(SS_size)+',NZ='+str(NZ)+',log_power'\
                      +',rep='+str(rep))
                Err,_,_,_ = LogReg.sto_weight_Sket_Newton(SS_size,'log_power',1,SF,NZ)
                Len1 = max(Len1,len(Err))
                Len2 = min(Len2,len(Err))
                noise_err.append(Err)
                noise_log_err.append(np.log10(Err))
                noise_Rerr.append(Err[1:]/Err[:-1])                                         
            for kk in range(len(noise_err)):
                noise_err[kk] = np.concatenate((noise_err[kk],[noise_err[kk][-1]]*(Len1-len(noise_err[kk]))))
                noise_log_err[kk] = np.concatenate((noise_log_err[kk],[noise_log_err[kk][-1]]*(Len1-len(noise_log_err[kk]))))                
                noise_Rerr[kk] = noise_Rerr[kk][:Len2-1]
            noise_err = np.vstack(noise_err)
            noise_log_err = np.vstack(noise_log_err)
            noise_Rerr = np.vstack(noise_Rerr)
            E_Err[-1] = np.median(noise_err, axis=0)
            E_ErrStd[-1] = np.std(noise_err, axis=0)
            Log_Err[-1] = np.median(noise_log_err, axis=0)
            Log_ErrStd[-1] = np.std(noise_log_err, axis=0)
            R_Err[-1] = np.mean(noise_Rerr, axis=0)
            R_ErrStd[-1] = np.std(noise_Rerr, axis=0)           
            C_Err1[-1] = np.median(np.sum(noise_err>=Count_EPS[0],1))
            C_ErrStd1[-1] = np.std(np.sum(noise_err>=Count_EPS[0],1))
            C_Err2[-1] = np.median(np.sum(noise_err>=Count_EPS[1],1))
            C_ErrStd2[-1] = np.std(np.sum(noise_err>=Count_EPS[1],1))

            # Save Result
            if not os.path.exists(Path+'/'+SF+'/Result'):
                os.makedirs(Path+'/'+SF+'/Result')
                os.makedirs(Path+'/'+SF+'/Conv')
                os.makedirs(Path+'/'+SF+'/ConvT')
                os.makedirs(Path+'/'+SF+'/ConvNL')
                os.makedirs(Path+'/'+SF+'/ConvTNL')
                os.makedirs(Path+'/'+SF+'/Ratio')
                os.makedirs(Path+'/'+SF+'/RatioT')
                os.makedirs(Path+'/'+SF+'/RatioNL')
                os.makedirs(Path+'/'+SF+'/RatioTNL')
                os.makedirs(Path+'/'+SF+'/Count')
            with open(Path+'/'+SF+'/Result'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Result.pkl',"wb") as f:
                pickle.dump([BFGS_Err, BFGS_Log_Err, BFGS_R_Err, BFGS_C_Err1, BFGS_C_Err2],f)
                pickle.dump([E_Err, Log_Err, R_Err, C_Err1, C_Err2],f)
                pickle.dump([E_ErrStd, Log_ErrStd, R_ErrStd, C_ErrStd1, C_ErrStd2],f)
    
            ### Plot the result
            # CONVERGENCE plot
            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
#            color = iter(cm.rainbow(np.linspace(0,1,len(power)+3)))
#            linecycler = cycle(["-","--","-.",":"])
            saveFigpath = Path+'/'+SF+'/ConvT'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            lll = min(sum(BFGS_Err>=1e-6)+1, len(BFGS_Err))
            plt.errorbar(range(lll),BFGS_Err[:lll],ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            lll = min(sum(E_Err[0]>=1e-6)+1, len(E_Err[0]))
            plt.errorbar(range(lll),E_Err[0][:lll],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            lll = min(sum(E_Err[1]>=1e-6)+1, len(E_Err[1]))
            plt.errorbar(range(lll),E_Err[1][:lll],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                lll = min(sum(E_Err[ii+2]>=1e-6)+1, len(E_Err[ii+2]))
                plt.errorbar(range(lll),E_Err[ii+2][:lll],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            lll = min(sum(E_Err[-1]>=1e-6)+1, len(E_Err[-1]))          
            plt.errorbar(range(lll),E_Err[-1][:lll],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.legend(bbox_to_anchor=(0.1,1.05), loc='lower left', borderaxespad=0, fontsize=8,ncol=len(power)+3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            plt.xlim(0, min(20,len(BFGS_Log_Err)+3))            
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+SF+'/ConvTNL'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            lll = min(sum(BFGS_Err>=1e-6)+1, len(BFGS_Err))
            plt.errorbar(range(lll),BFGS_Err[:lll],ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            lll = min(sum(E_Err[0]>=1e-6)+1, len(E_Err[0]))
            plt.errorbar(range(lll),E_Err[0][:lll],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            lll = min(sum(E_Err[1]>=1e-6)+1, len(E_Err[1]))
            plt.errorbar(range(lll),E_Err[1][:lll],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                lll = min(sum(E_Err[ii+2]>=1e-6)+1, len(E_Err[ii+2]))
                plt.errorbar(range(lll),E_Err[ii+2][:lll],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            lll = min(sum(E_Err[-1]>=1e-6)+1, len(E_Err[-1]))          
            plt.errorbar(range(lll),E_Err[-1][:lll],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            plt.xlim(0, min(20,len(BFGS_Log_Err)+3))            
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+SF+'/Conv'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            lll = min(sum(BFGS_Err>=1e-6)+1, len(BFGS_Err))
            plt.errorbar(range(lll),BFGS_Err[:lll],ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            lll = min(sum(E_Err[0]>=1e-6)+1, len(E_Err[0]))
            plt.errorbar(range(lll),E_Err[0][:lll],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            lll = min(sum(E_Err[1]>=1e-6)+1, len(E_Err[1]))
            plt.errorbar(range(lll),E_Err[1][:lll],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                lll = min(sum(E_Err[ii+2]>=1e-6)+1, len(E_Err[ii+2]))
                plt.errorbar(range(lll),E_Err[ii+2][:lll],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            lll = min(sum(E_Err[-1]>=1e-6)+1, len(E_Err[-1]))          
            plt.errorbar(range(lll),E_Err[-1][:lll],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.legend(bbox_to_anchor=(0.1,1.05), loc='lower left', borderaxespad=0, fontsize=8,ncol=len(power)+3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+SF+'/ConvNL'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            lll = min(sum(BFGS_Err>=1e-6)+1, len(BFGS_Err))
            plt.errorbar(range(lll),BFGS_Err[:lll],ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            lll = min(sum(E_Err[0]>=1e-6)+1, len(E_Err[0]))
            plt.errorbar(range(lll),E_Err[0][:lll],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            lll = min(sum(E_Err[1]>=1e-6)+1, len(E_Err[1]))
            plt.errorbar(range(lll),E_Err[1][:lll],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                lll = min(sum(E_Err[ii+2]>=1e-6)+1, len(E_Err[ii+2]))
                plt.errorbar(range(lll),E_Err[ii+2][:lll],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            lll = min(sum(E_Err[-1]>=1e-6)+1, len(E_Err[-1]))          
            plt.errorbar(range(lll),E_Err[-1][:lll],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")


            # RATE plot
            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+SF+'/RatioT'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Rate.png'
            plt.errorbar(range(len(BFGS_R_Err)),BFGS_R_Err,ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[0])),R_Err[0],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[1])),R_Err[1],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                plt.errorbar(range(len(R_Err[ii+2])),R_Err[ii+2],yerr=R_ErrStd[ii+2],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[-1])),R_Err[-1],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.legend(bbox_to_anchor=(0.1,1.05), loc='lower left', borderaxespad=0, fontsize=8,ncol=len(power)+3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\frac{\|\|x_{t+1} - x^\star\|\|_{H^\star}}{\|\|x_{t} - x^\star\|\|_{H^\star}}$',fontsize=25)
            plt.xlim(0, min(20,len(BFGS_Log_Err)+2))
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+SF+'/RatioTNL'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Rate.png'
            plt.errorbar(range(len(BFGS_R_Err)),BFGS_R_Err,ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[0])),R_Err[0],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[1])),R_Err[1],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                plt.errorbar(range(len(R_Err[ii+2])),R_Err[ii+2],yerr=R_ErrStd[ii+2],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[-1])),R_Err[-1],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\frac{\|\|x_{t+1} - x^\star\|\|_{H^\star}}{\|\|x_{t} - x^\star\|\|_{H^\star}}$',fontsize=25)
            plt.xlim(0, min(20,len(BFGS_Log_Err)+2))
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+SF+'/Ratio'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Rate.png'
            plt.errorbar(range(len(BFGS_R_Err)),BFGS_R_Err,ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[0])),R_Err[0],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[1])),R_Err[1],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                plt.errorbar(range(len(R_Err[ii+2])),R_Err[ii+2],yerr=R_ErrStd[ii+2],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[-1])),R_Err[-1],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.legend(bbox_to_anchor=(0.1,1.05), loc='lower left', borderaxespad=0, fontsize=8,ncol=len(power)+3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\frac{\|\|x_{t+1} - x^\star\|\|_{H^\star}}{\|\|x_{t} - x^\star\|\|_{H^\star}}$',fontsize=25)
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            f = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+SF+'/RatioNL'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Rate.png'
            plt.errorbar(range(len(BFGS_R_Err)),BFGS_R_Err,ls=next(linecycler),label='BFGS',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[0])),R_Err[0],ls=next(linecycler),label='NoAvg',color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[1])),R_Err[1],ls=next(linecycler),label=r'UnifAvg',color=next(color),linewidth=3)
            for ii, pp in enumerate(power[1:]):
                lab='$\omega_t=(t+1)^{'+str(pp)+'}$'
                plt.errorbar(range(len(R_Err[ii+2])),R_Err[ii+2],yerr=R_ErrStd[ii+2],ls=next(linecycler),label=r''+lab,color=next(color),linewidth=3)
            plt.errorbar(range(len(R_Err[-1])),R_Err[-1],ls=next(linecycler),label=r'WeightAvg',color=next(color),linewidth=3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\frac{\|\|x_{t+1} - x^\star\|\|_{H^\star}}{\|\|x_{t} - x^\star\|\|_{H^\star}}$',fontsize=25)
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            # BAR plot
            f = plt.figure()
            saveFigpath = Path+'/'+SF+'/Count'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Count1.png'
            plt.bar(np.arange(len(power)+3),[BFGS_C_Err1]+C_Err1,capsize=10,color=['magenta','blue','green','red'])
            lab = ['BFGS', 'NoAvg','UnifAvg'] \
                   + ['$(t+1)^{'+str(pp)+'}$' for pp in power[1:]]\
                   + ['WeightAvg']
            plt.xticks(np.arange(len(power)+3),lab,fontsize=12)
            plt.ylabel('$t$',fontsize=25)
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")
            
            f = plt.figure()
            saveFigpath = Path+'/'+SF+'/Count'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Count2.png'
            plt.bar(np.arange(len(power)+3),[BFGS_C_Err2]+C_Err2,capsize=10,color=['magenta','blue','green','red'])
            lab = ['BFGS', 'NoAvg','UnifAvg'] \
                   + ['$(t+1)^{'+str(pp)+'}$' for pp in power[1:]]\
                   + ['WeightAvg']
            plt.xticks(np.arange(len(power)+3),lab,fontsize=12)
            plt.ylabel('$t$',fontsize=25)
            f.savefig(saveFigpath,dpi=300,bbox_inches="tight")

    
    ### Plot Sketch matrix on the same figure
    Averg_Seq = ['NoAvg','UnifAvg'] \
                +['power'+str(pp) for pp in power[1:]] + ['WeightAvg']
    if Data.IdReal == 'real':
        Path = os.getcwd()+ '/Figure/'+Data.name+'/AvgSeq'
    else:
        if Data.IdCond == 'true':
            Path = os.getcwd()+ '/Figure/'+ 'N'+str(n)+'D'+str(d)+'L'+str(lambd)+'K'+str(Data.kap)+'D/AvgSeq'
        else:
            Path = os.getcwd()+ '/Figure/'+ 'N'+str(n)+'D'+str(d)+'L'+str(lambd)+'K'+str(Data.kap)+'H/AvgSeq'
    if not os.path.exists(Path):
        os.makedirs(Path)
                
    for ii, AS in enumerate(Averg_Seq):
        if not os.path.exists(Path+'/'+AS+'/Conv'):
            os.makedirs(Path+'/'+AS+'/Conv')
            os.makedirs(Path+'/'+AS+'/ConvNL')
            os.makedirs(Path+'/'+AS+'/ConvT')
            os.makedirs(Path+'/'+AS+'/ConvTNL')
        generator = ((SS_size,NZ) for SS_size in Sket_Size for NZ in nnz)
        for SS_size, NZ in generator:
            g = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+AS+'/ConvT'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            for SF in sketch_func:
                if SF != 'LESS-uniform':
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZNone'+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                else:
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                lll = min(sum(A1[ii]>=1e-6)+1,len(A1[ii]))  
                plt.errorbar(range(lll),A1[ii][:lll],ls=next(linecycler),label=SF,color=next(color),linewidth=3)
            plt.legend(bbox_to_anchor=(0,1.05), loc='lower left', borderaxespad=0, fontsize=8,ncol=len(sketch_func))
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            plt.xlim(0,20)
            g.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            g = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+AS+'/ConvTNL'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            for SF in sketch_func:
                if SF != 'LESS-uniform':
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZNone'+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                else:
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                lll = min(sum(A1[ii]>=1e-6)+1,len(A1[ii]))  
                plt.errorbar(range(lll),A1[ii][:lll],ls=next(linecycler),label=SF,color=next(color),linewidth=3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            plt.xlim(0,20)
            g.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            g = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+AS+'/Conv'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            for SF in sketch_func:
                if SF != 'LESS-uniform':
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZNone'+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                else:
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                lll = min(sum(A1[ii]>=1e-6)+1,len(A1[ii]))  
                plt.errorbar(range(lll),A1[ii][:lll],ls=next(linecycler),label=SF,color=next(color),linewidth=3)
            plt.legend(bbox_to_anchor=(0,1.05), loc='lower left', borderaxespad=0, fontsize=8,ncol=len(sketch_func))
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            g.savefig(saveFigpath,dpi=300,bbox_inches="tight")

            g = plt.figure()
            color = iter(['magenta','blue','green','red'])
            linecycler = iter([(0, ()),(0, (5,1)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))])
            saveFigpath = Path+'/'+AS+'/ConvNL'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Con.png'
            for SF in sketch_func:
                if SF != 'LESS-uniform':
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZNone'+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                else:
                    with open(Path+'/../Sketch/'+SF+'/Result'+'/SS'+str(SS_size)+'NZ'+str(NZ).replace(".","_")+'Result.pkl',"rb") as f:
                        pickle.load(f)
                        A1,_,_,_,_ = pickle.load(f)
                lll = min(sum(A1[ii]>=1e-6)+1,len(A1[ii]))  
                plt.errorbar(range(lll),A1[ii][:lll],ls=next(linecycler),label=SF,color=next(color),linewidth=3)
            plt.xlabel('$t$',fontsize=25)
            plt.ylabel(r'$\|\|x_t - x^\star\|\|_{H^\star}$',fontsize=25)
            plt.yscale('log')
            g.savefig(saveFigpath,dpi=300,bbox_inches="tight")




