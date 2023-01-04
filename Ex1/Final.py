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


def Ex1():
    n = 10
    I = 5
    #beta = np.array([[0.06,0.015,0.015,0.19,0.21,0.05,0.18,0.1,0.04,0.04],
    beta = np.array([[0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09],
                        [0.05,0.1,0.17,0.02,0.16,0.1,0.16,0.07,0.03,0.04],
                        [0.06,0.05,0.09,0.15,0.07,0.08,0.14,0.02,0.11,0.13],
                        [0.01,0.15,0.01,0.11,0.11,0.16,0.03,0.14,0.09,0.09],
                        [0.03,0.13,0.05,0.16,0.16,0.07,0.08,0.1,0.08,0.04]])
    e = 1.0*np.array([[59,76,10,37,54,99,73,30,25,20],
                [14,40,63,57,69,39,34,86,10,56],
                [19,57,43,65,78,40,9,82,71,82],
                [10,65,35,43,63,74,79,38,20,27],
                [37,70,40,94,83,15,34,97,35,34]])
    Econ = Economy(n,I)
    Econ.alpha = beta
    Econ.e = e
    Econ.allocations = e
    Econ.pag = Econ.evalp(Econ.allocations)
    delta = 1e-1*np.ones_like(Econ.pag)
    Econ.delta = delta
    return Econ

np.random.seed(10)

exfig = 'pdf'
Ec_aux = Ex1()
p_we, x_we = Ec_aux.Walraseqrtr()
Econ = {}
ntrials = 10
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

for k in range(ntrials):
    print('Trial number {}'.format(k))
    Econ[k] = Ex1()
    Walras_prices[:,k], Walras_alloc[k] = Econ[k].Walraseqrtr()
    t = time.time()
    EvPag[k], EvAlloc[k], EvDelta[k], TimesChange[k], K[k], eq_status[k] = Econ[k].Bilateral(eps_prices = 1e-6, MAXIT = 250000, lbda = 0.975, delta_tol = 1e-18, inspection='ran')
    ExTimes[k] = time.time()-t
    print('Total time {}[s]'.format(ExTimes[k]))
    Evpbar[1:,k] = np.max(Econ[k].pag,axis=0)
    EvSD[k] = np.max(np.std(Econ[k].pag,axis=0))
    print('Max_i Std of p_ij of {}'.format(EvSD[k]))

    plt.figure()
    Z = np.zeros((Econ[k].I,K[k]))
    for ii in range(Econ[k].I):
        for kk in range(K[k]):
            Z[ii,kk] = np.prod(EvAlloc[k][kk][ii,:]**Econ[k].alpha[ii,:])
        plt.plot(Z[ii,:],label='Agent {}'.format(ii+1))
#    plt.title(r'Utilities evolution for each agent $\{\Pi_{j=0}^n x_{ij}^{\nu,\alpha_i}\}$')
    plt.xlabel(r'Iteration $\nu$')
    plt.ylabel(r'$u(x_{i\cdot}^{\nu})$')
    plt.legend(loc='upper left')
    plt.savefig('Ut_trial'+str(k)+'.'+exfig)
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
#print("Median of trade operations {}".format(np.median(TimesChange)))

plt.figure()
plt.plot(Evpbar)
plt.title(r'Boxplot for equilibrium price thresholds $\bar{p}_{\cdot j}$')
plt.xlabel(r'Goods $j$')
plt.ylabel(r'$p$')
plt.savefig('Pbar.'+exfig)
plt.close()

'''
for k in range(ntrials):
    fig, ax = plt.subplots()
    x = np.arange(Ec_aux.I)+1
    width = 0.35
    rects1 = ax.bar(x-width/2, Wealth_bte[k,:], width, label='BTE')
    rects2 = ax.bar(x+width/2, Wealth_wal[k,:], width, label='Wal')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Wealth')
    ax.set_title('Wealth')
    ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.savefig('WealthBTEWal_trial'+str(k)+'.'+exfig)
    plt.close()
    fig, ax = plt.subplots()
    x = np.arange(Ec_aux.I)+1
    width = 0.35
    rects1 = ax.bar(x-width/2, Utilities_bte[k,:], width, label='BTE')
    rects2 = ax.bar(x+width/2, Utilities_wal[k,:], width, label='Wal')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Utilities')
    ax.set_title('Utility functions')
    ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.savefig('UtilityBTEWal_trial'+str(k)+'.'+exfig)
    plt.close()
'''

plt.figure()
x = np.linspace(np.amin([Utilities_wal,Utilities_bte]),np.amax([Utilities_wal,Utilities_bte]),100)
for k in range(ntrials):
    plt.scatter(Utilities_bte[k,:],Utilities_wal[k,:],label='Trial {}'.format(k))
plt.plot(x,x,linestyle='--')
plt.xlabel('BTE Utility')
plt.ylabel('Walras Utility')
plt.legend(loc='lower right')
plt.savefig('Utilities.'+exfig)

plt.figure()
x = np.linspace(np.amin([Wealth_wal,Wealth_bte]),np.amax([Wealth_wal,Wealth_bte]),100)
for k in range(ntrials):
    plt.scatter(Wealth_bte[k,:],Wealth_wal[k,:],label='Trial {}'.format(k))
plt.plot(x,x,linestyle='--')
plt.xlabel('BTE Wealth')
plt.ylabel('Walras Wealth')
plt.legend(loc='lower right')
plt.savefig('Wealth.'+exfig)


Gini_bte = np.zeros(ntrials)
Gini_wal = np.zeros(ntrials)
for k in range(ntrials):
    Gini_bte[k] = gini(Wealth_bte[k,:])
    Gini_wal[k] = gini(Wealth_wal[k,:])

plt.figure()
x = np.linspace(np.amin([Gini_bte, Gini_wal]),np.amax([Gini_bte, Gini_wal]),100)
for k in range(ntrials):
    plt.scatter(Gini_bte[k],Gini_wal[k],label='Trial {}'.format(k))
plt.plot(x,x,linestyle='--')
plt.xlabel('Gini BTE')
plt.ylabel('Gini Walras')
plt.legend(loc='lower right')
plt.savefig('Gini.'+exfig)

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

PShow = np.column_stack([Evpbar[:,[i0,i1,i2,i3]], p_we])#, axis=1)
idags = np.asarray(['$j=0$','$j=1$', '$j=2$','$j=3$','$j=4$','$j=5$', '$j=6$', '$j=7$', '$j=8$', '$j=9$'])
labls = ['$\overline p^{}$'.format(i0),'$\overline p^{}$'.format(i1), '$\overline p^{}$'.format(i2), '$\overline p^{}$'.format(i3), '$\overline p^{W}$']
df = pd.DataFrame(PShow,columns=labls, index=idags)
df.T.to_latex('Ex1_results.tex', float_format="{:0.4f}".format, escape=False)


p_we, x_wertr = Ec_aux.Walraseqrtr()
p_wejd, x_we = Ec_aux.Walraseq()

idxs = np.array([i0,i1,i2,i3])
print('j&{}&{}&{}&{}&Wal'.format(i0,i1,i2,i3))
for jj in range(4):
    ids = idxs[jj]
    print('{}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}'.format(ids,Econ[ids].allocations[0,0],Econ[ids].allocations[1,0],Econ[ids].allocations[2,0],Econ[ids].allocations[0,1],Econ[ids].allocations[1,1],Econ[ids].allocations[2,1],Econ[ids].allocations[0,2],Econ[ids].allocations[1,2],Econ[ids].allocations[2,2]))
print('Wal&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}'.format(x_we[0,0],x_we[1,0],x_we[2,0],x_we[0,1],x_we[1,1],x_we[2,1],x_we[0,2],x_we[1,2],x_we[2,2]))
