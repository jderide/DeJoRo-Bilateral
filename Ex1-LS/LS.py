from BTE import *
from copy import deepcopy
import pandas as pd
import gc

from time import perf_counter, strftime,localtime
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def CDLS():
    n = 100
    I = 10
    beta_aux = np.random.rand(I,n)
    beta = 0.9*beta_aux/np.sum(beta_aux,axis=1)[:,None]
    e = 100*np.random.rand(I,n)
    Econ = Economy(n,I)
    Econ.alpha = beta
    Econ.e = e
    Econ.allocations = e
    Econ.pag = Econ.evalp(Econ.allocations)
    delta = 1e-1*np.ones_like(Econ.pag)
    Econ.delta = delta
    return Econ


if __name__ == "__main__":

    np.random.seed(10)

    exfig = 'pdf'
    Ec_aux = CDLS()
    p_we, x_we = Ec_aux.Walraseq()
    Econ = {}
    ntrials = 2
    EvPag ={}
    EvAlloc = {}
    EvDelta = {}
    TimesChange = {}
    K = np.zeros(ntrials, dtype='i')
    eq_status = np.zeros(ntrials)
    ExTimes = np.zeros(ntrials)
    EvSD = np.zeros(ntrials)
    Evpbar = np.ones((Ec_aux.n, ntrials))
    Walras_prices = np.zeros((Ec_aux.n, ntrials))
    Walras_alloc = {}
    BPT = np.zeros((ntrials,Ec_aux.n-1))
    Prices_wal = np.zeros((ntrials,Ec_aux.n))
    Wealth_bte = np.zeros((ntrials,Ec_aux.I))
    Utilities_bte = np.zeros((ntrials,Ec_aux.I))
    Wealth_wal = np.zeros((ntrials,Ec_aux.I))
    Utilities_wal = np.zeros((ntrials,Ec_aux.I))

    writer = pd.ExcelWriter('Params.xlsx')  #Key 2, create an excel sheet named hhh
    for i in range(Ec_aux.I):
        XX = np.zeros((Ec_aux.n,3))
        for j in range(Ec_aux.n):
            XX[j,:] = np.array([j,Ec_aux.e[i,j],Ec_aux.alpha[i,j]])
        strpg = 'Agent_{}'.format(i)
        data_df = pd.DataFrame(XX,columns=['j', 'x_i^0', '\beta_{ij}'])
        data_df.to_excel(writer,strpg,float_format='%.8f')  #Key 3, float_format controls the accuracy, write data_df to the first page of the hhh form. If there are multiple files, you can write in page_2
    writer.save()

    for k in range(ntrials):
        print('Trial number {}'.format(k))
        Econ[k] = deepcopy(Ec_aux)
        Walras_prices[:,k], Walras_alloc[k] = Econ[k].Walraseq()
        t = time.time()
        EvPag[k], EvAlloc[k], EvDelta[k], TimesChange[k], K[k], eq_status[k] = Econ[k].Bilateral(eps_prices = 1e-4, MAXIT = 250000, lbda = 0.998, delta_tol = 1e-18, inspection='ran')
        ExTimes[k] = time.time()-t
        print('Total time {}[s]'.format(ExTimes[k]))
        Evpbar[1:,k] = np.max(Econ[k].pag,axis=0)
        EvSD[k] = np.max(np.std(Econ[k].pag,axis=0))
        print('Max_i Std of p_ij of {}'.format(EvSD[k]))

        for i in range(Econ[k].I):
            Z = np.zeros((Econ[k].I,K[k]))
            for kk in range(K[k]):
                for ii in range(Econ[k].I):
                    Z[ii,kk] = EvAlloc[k][kk][ii,0]
                    for nn in range(Econ[k].n-1):
                        Z[ii,kk] += (EvAlloc[k][kk][ii,nn+1])*(EvPag[k][kk][ii,nn])
            plt.figure()
            for ii in range(Econ[k].I):
                plt.plot(Z[ii,:],label='Agent {}'.format(ii+1))
            plt.title(r'Wealth evolution for each agent $\{x_{i0}^{\nu}+\sum_{j=1}^n x_{ij}^{\nu}p_{ij}^{\nu}\}$')
            plt.xlabel(r'Iteration $\nu$')
            plt.ylabel(r'$b$')
            plt.legend(loc='upper left')
            plt.savefig('Wealth_trial'+str(k)+'.'+exfig)
            plt.close()
            #
            #
        # Computation: Prices, Wealth and Utilities
        #
        #
        BPT[k,:] = np.max(Econ[k].pag,axis=0)
        for i in range(Econ[k].I):
            Wealth_bte[k,i] = np.sum(Evpbar[:,k]*Econ[k].allocations[i,:])
            Utilities_bte[k,i] = np.prod(Econ[k].allocations[i,:]**Econ[k].alpha[i,:])
            Wealth_wal[k,i] = np.sum(Walras_prices[:,k]*Walras_alloc[k][i,:])
            Utilities_wal[k,i] = np.prod(Walras_alloc[k][i,:]**Econ[k].alpha[i,:])
        gc.collect()

    print('Number of eqs {} out of {}'.format(np.sum(eq_status),ntrials))
    print("Median of K {}".format(np.median(K)))
    print("Median of Time {}".format(np.median(ExTimes)))

    plt.figure()
    plt.boxplot(BPT, showfliers=False)
    plt.title(r'Boxplot for equilibrium price thresholds $\bar{p}_{\cdot j}$')
    plt.xlabel(r'Goods $j$')
    plt.ylabel(r'$p$')
    plt.savefig('Pag.'+exfig)
    plt.close()

    plt.figure()
    pmean = Evpbar.mean(axis=1)
    diff = np.zeros((ntrials,ntrials))
    for i in range(ntrials):
        for j in range(i):
            diff[i,j] = np.max(np.abs(Evpbar[:,i]- Evpbar[:,j]))
    i0,i1 = np.unravel_index(np.argmax(diff), diff.shape)
    plt.plot(Evpbar[:,i0],label='Trial {}'.format(i0))
    plt.plot(Evpbar[:,i1],label='Trial {}'.format(i1))
    print(diff)
    diff[i0,:] *= 0.0
    diff[:,i0] *= 0.0
    diff[i1,:] *= 0.0
    diff[:,i1] *= 0.0
    print(diff)
    i2,i3 = np.unravel_index(np.argmax(diff), diff.shape)
    plt.plot(Evpbar[:,i2],label='Trial {}'.format(i2))
    plt.plot(Evpbar[:,i3],label='Trial {}'.format(i3))
    plt.legend(loc='lower right')
    plt.savefig('Pbardiffs.'+exfig)
    plt.close()

    print(i0,i1,i2,i3)
    print('j&{}&{}&{}&{}&Wal'.format(i0,i1,i2,i3))
    for n in range(Ec_aux.n):
        print('{}&{}&{}&{}&{}&{}'.format(n,Evpbar[n,i0],Evpbar[n,i1],Evpbar[n,i2],Evpbar[n,i3],p_we[n]))
    #aux_str=strftime("%Y%m%d_%H%M", localtime())
    #pname='Results_'+aux_str
    #os.system('mkdir '+pname)
    #os.system('mv *.'+exfig+' '+pname)
    #os.system('cp BTE.py '+pname)
    #os.system('cp Ex1.py '+pname)
