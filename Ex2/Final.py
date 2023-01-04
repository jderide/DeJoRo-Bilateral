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
    n = 3
    I = 3
    beta = np.array([[0.6,0.15,0.15],
                        [0.01,0.85,0.04],
                        [0.01,0.09,0.8]])
    e = 1.0*np.array([[10,10,10],
                        [2,8,80],
                        [2,80,8],])
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
p_we, x_we = Ec_aux.Walraseq()
p_wertr, x_wertr = Ec_aux.Walraseqrtr()
Econ = {}
ntrials = 20
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
Wealth_bte = np.zeros((ntrials+1,Ec_aux.I))
Utilities_bte = np.zeros((ntrials+1,Ec_aux.I))
Wealth_wal = np.zeros((ntrials,Ec_aux.I))
Utilities_wal = np.zeros((ntrials,Ec_aux.I))
Prices_final = np.zeros((ntrials,Ec_aux.n-1))
insp = {}
l_order = {0:[0,1,2],
     1:[0,2,1],
     2:[1,0,2],
     3:[1,2,0],
     4:[2,0,1],
     5:[2,1,0]}
g_order = {0:[0,1], 1:[1,0]}

u0 = np.ones(Ec_aux.I)
w0 = np.ones(Ec_aux.I)
for i in range(Ec_aux.I):
    p0 = np.append(1.0, Ec_aux.pag[i,:] )
    print(p0)
    w0[i] = np.dot(p0,Ec_aux.e[i,:])
    u0[i] = np.prod(Ec_aux.allocations[i,:]**Ec_aux.alpha[i,:])

for ku in range(ntrials):
    if ku < 12:
        for io in range(6):
            for go in range(2):
                insp[2*io+go] = ['fixed',l_order[io], g_order[go]]
    else:
        insp[ku] = ['ran',{},{}]
print(insp)

for k in range(ntrials):
    print('Trial number {}'.format(k))
    Econ[k] = Ex1()
    Walras_prices[:,k], Walras_alloc[k] = Econ[k].Walraseqrtr()
    t = time.time()
#    insp[k] = ['ran',{},{}]
    print(insp[k])
    EvPag[k], EvAlloc[k], EvDelta[k], TimesChange[k], K[k], eq_status[k] = Econ[k].Bilateral(eps_prices = 1e-6, MAXIT = 250000, lbda = 0.975, delta_tol = 1e-18, inspection=insp[k])
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
    BPT[k,:] = np.max(Econ[k].pag,axis=0)
    for i in range(Econ[k].I):
        Wealth_bte[k,i] = np.sum(Evpbar[:,k]*Econ[k].allocations[i,:])
        Utilities_bte[k,i] = np.prod(Econ[k].allocations[i,:]**Econ[k].alpha[i,:])
        Wealth_wal[k,i] = np.sum(Walras_prices[:,k]*Walras_alloc[k][i,:])
        Utilities_wal[k,i] = np.prod(Walras_alloc[k][i,:]**Econ[k].alpha[i,:])
#    plt.figure()
#    Z = np.zeros((Econ[k].I,K[k]))
#    for ii in range(Econ[k].I):
#        for kk in range(K[k]):
#            Z[ii,kk] = np.prod(EvAlloc[k][kk][ii,:]**Econ[k].alpha[ii,:])
#        plt.plot(Z[ii,:],label='Agent {}'.format(ii+1))
#    #    plt.title(r'Utilities evolution for each agent $\{\Pi_{j=0}^n x_{ij}^{\nu,\alpha_i}\}$')
#    plt.xlabel(r'Iteration $\nu$')
#    plt.ylabel(r'$u(x_{i\cdot}^{\nu})$')
#    plt.legend(loc='upper left')
#    plt.savefig('Ut_trial'+str(k)+'.'+exfig)
#    plt.close()
    gc.collect()

Wealth_bte[-1,:] = Wealth_wal[0,:]
Utilities_bte[-1,:] = Utilities_wal[0,:]

print('Number of eqs {} out of {}'.format(np.sum(eq_status),ntrials))
print("Median of K {}".format(np.median(K)))
print("Median of Time {}".format(np.median(ExTimes)))
#print("Median of trade operations {}".format(np.median(TimesChange)))

soc_ut_bte = np.sum(Utilities_bte[:-1,:],axis=1)
soc_ut_wal = np.sum(Utilities_wal,axis=1)
wealth_bte = np.sum(Wealth_bte,axis=1)

plt.figure()
plt.plot(soc_ut_bte, label='BTE')
plt.plot(soc_ut_wal, label='Wal')
plt.title(r'Social utility $\displaystyle \sum_{i\in I} u_i(x_i)$')
plt.xlabel(r'Trials $\nu$')
plt.ylabel(r'$U$')
plt.savefig('SocUt.'+exfig)
plt.close()

plt.figure()
labels = range(ntrials+1)
for i in range(Ec_aux.I):
    plt.bar(labels, Wealth_bte[:,i], bottom=np.sum(Wealth_bte[:,:i],axis=1))
plt.title(r'Wealth ${\bar p}\cdot x_i^{\nu}$')
plt.axvline(x = ntrials-0.25, linestyle = '-.')
plt.savefig('Wealth_bte.'+exfig)
plt.close()

plt.figure()
labels = range(ntrials+1)
for i in range(Ec_aux.I):
    plt.bar(labels, Utilities_bte[:,i], bottom=np.sum(Utilities_bte[:,:i],axis=1))
plt.title(r'Utility $ u_i(x_i^{\nu})$')
plt.axvline(x = ntrials-0.25, linestyle = '-.')
plt.savefig('Utilities_bte.'+exfig)
plt.close()


plt.figure()
plt.scatter(Evpbar[1,:],Evpbar[2,:])
plt.scatter(p_wertr[1],p_wertr[2], color='red')
plt.title(r'Final BTE equilibrium price thresholds $\bar{p}_{\cdot j}$')
plt.xlabel(r'Goods $j$')
plt.ylabel(r'$p$')
plt.savefig('Pbar.'+exfig)
plt.close()

plt.figure()
plt.boxplot(BPT, showfliers=False)
plt.title(r'Boxplot for equilibrium price thresholds $\bar{p}_{\cdot j}$')
plt.xlabel(r'Goods $j$')
plt.ylabel(r'$p$')
plt.savefig('Pag.'+exfig)
plt.close()

#'''
for k in range(ntrials):
    lnst = ['solid', 'dotted', 'dashed']*ntrials
    cols = [(130/235,36/235,51/235),(0,50/235,90/235),(130/235,120/235,111/235),(0/235,173/235,208/235),(17/235,28/235,36/235),(220/235,220/235,220/235),(0,128/235,128/235)]
    delta = 0.0025
    s = np.sum(Ec_aux.e,axis=0) # Total suppy
    fig, ax = plt.subplots()
    for k in range(ntrials):
        XX = np.zeros(K[k])
        YY = np.zeros(K[k])
        for kk in range(K[k]):
            XX[kk] = EvAlloc[k][kk][0,1]
            YY[kk] = EvAlloc[k][kk][0,2]
        plt.plot(XX, YY, label=("Example {}".format(k+1)),linewidth=1.5,linestyle=lnst[k])
        plt.annotate(r'$x^{0}$',xy=(XX[0],YY[0]))
        plt.annotate(r'$\bar{x}$',xy=(XX[-1],YY[-1]-0.2))
    plt.xlabel('Good 1')
    plt.ylabel('Good 2')
    ax.set_title('Edgeworth box')
    ax.set_xlim(0.0, s[1])
    ax.set_ylim(0.0, s[2])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False)
    plt.savefig(str(k)+'Edgebox12.'+exfig)
    plt.close()
    #
    fig, ax = plt.subplots()
    ax.set_title('Edgeworth box')
    for k in range(ntrials):
        XX = np.zeros(K[k])
        YY = np.zeros(K[k])
        for kk in range(K[k]):
            XX[kk] = EvAlloc[k][kk][0,0]
            YY[kk] = EvAlloc[k][kk][0,1]
        plt.plot(XX, YY, label=("Example {}".format(k+1)),linewidth=1.5,linestyle=lnst[k])
        plt.annotate(r'$x^{0}$',xy=(XX[0],YY[0]))
        plt.annotate(r'$\bar{x}$',xy=(XX[-1],YY[-1]-0.2))
    plt.xlabel('Good 0')
    plt.ylabel('Good 1')
    ax.set_xlim(0.0, s[0])
    ax.set_ylim(0.0, s[1])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False)
    plt.savefig(str(k)+'Edgebox01.'+exfig)
    plt.close()
    #
    fig, ax = plt.subplots()
    ax.set_title('Edgeworth box')
    for k in range(ntrials):
        XX = np.zeros(K[k])
        YY = np.zeros(K[k])
        for kk in range(K[k]):
            XX[kk] = EvAlloc[k][kk][0,0]
            YY[kk] = EvAlloc[k][kk][0,2]
        plt.plot(XX, YY, label=("Example {}".format(k+1)),linewidth=1.5,linestyle=lnst[k])
        plt.annotate(r'$x^{0}$',xy=(XX[0],YY[0]))
        plt.annotate(r'$\bar{x}$',xy=(XX[-1],YY[-1]-0.2))
    plt.xlabel('Good 0')
    plt.ylabel('Good 2')
    ax.set_xlim(0.0, s[0])
    ax.set_ylim(0.0, s[2])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,  frameon=False)
    plt.savefig(str(k)+'Edgebox02.'+exfig)
    plt.close()
#'''

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

writer = pd.ExcelWriter('Output.xlsx')  #Key 2, create an excel sheet named hhh
for k in range(ntrials):
    XX = np.zeros((K[k],11))
    for j in range(K[k]):
        a = np.array([j,EvPag[k][j][0,0],EvPag[k][j][0,1],EvPag[k][j][1,0],EvPag[k][j][1,1],EvAlloc[k][j][0,0],EvAlloc[k][j][0,1],EvAlloc[k][j][0,2],EvAlloc[k][j][1,0],EvAlloc[k][j][1,1],EvAlloc[k][j][1,2]])
        XX[j,:] = a
    strpg = 'page_{}'.format(k)
    data_df = pd.DataFrame(XX)
    data_df.to_excel(writer,strpg,float_format='%.8f')  #Key 3, float_format controls the accuracy, write data_df to the first page of the hhh form. If there are multiple files, you can write in page_2
data_fd = pd.DataFrame(Walras_alloc[0])
data_fd.to_excel(writer,'Walras_alloc',float_format='%.8f')
data_fd = pd.DataFrame(Evpbar)
data_fd.to_excel(writer,'Evpbar',float_format='%.8f')
writer.save()


plt.figure()
x = np.linspace(np.amin([Utilities_wal,Utilities_bte[:-1,:]]),np.amax([Utilities_wal,Utilities_bte[:-1,:]]),100)
for k in range(ntrials):
    plt.scatter(Utilities_wal[k,:],Utilities_bte[k,:],label='Trial {}'.format(k))
plt.plot(x,x,linestyle='--')
plt.ylabel('BTE Utility')
plt.xlabel('Walras Utility')
plt.legend(loc='lower right')
plt.savefig('Utilities.'+exfig)
plt.close()

print(p_wertr)
#verificar el random de los agentes check
#Revisar tabla de dotaciones de equilibrio (por el orden randomizado)


#Tab4
idx = [0, 4, 10, 13, 14]
idags = np.asarray(['$j=1$','$j=2$'])
#labls = ['$\overline p^{}$'.format(i0),'$\overline p^{}$'.format(i1), '$\overline p^{}$'.format(i2), '$\overline p^{}$'.format(i3), '$\overline p^{W}$']
labls = ['$\overline p^{1}$', '$\overline p^{2}$', '$\overline p^{3}$', '$\overline p^{4}$', '$\overline p^{5}$', '$\overline p^{W}$']
T4 = np.column_stack([Evpbar[1:,idx],p_we[1:]])
df = pd.DataFrame(T4,columns=labls, index=idags)
df.to_latex('Tab4_results.tex', float_format="{:0.4f}".format, escape=False)


Tab5 = np.zeros((6,Ec_aux.n*Ec_aux.I))
for nn in range(Ec_aux.n):
    for ii in range(Ec_aux.I):
        for jj in range(5):
            Tab5[jj,3*nn+ii] = Econ[idx[jj]].allocations[ii,nn]
        Tab5[5,3*nn+ii] = x_we[ii,nn]

idags5 = np.asarray(['$\nu=1$','$\nu=2$','$\nu=3$','$\nu=4$','$\nu=5$','Wal'])
labls5 = ['$\overline x^{\nu}_{10}$','$\overline x^{\nu}_{20}$','$\overline x^{\nu}_{30}$','$\overline x^{\nu}_{11}$','$\overline x^{\nu}_{21}$','$\overline x^{\nu}_{31}$','$\overline x^{\nu}_{12}$','$\overline x^{\nu}_{22}$','$\overline x^{\nu}_{32}$']
df5 = pd.DataFrame(Tab5,columns=labls5, index=idags5)
df5.to_latex('Tab5_results.tex', float_format="{:0.2f}".format, escape=False)
