from BTEQuad import *
import pandas as pd
import gc

from time import perf_counter, strftime,localtime
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

import pandas as pd

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def makeArrow(ax,pos,function,direction):
    delta = 0.0001 if direction >= 0 else -0.0001
    ax.arrow(pos,function(pos),pos+delta,function(pos+delta),head_width=0.05,head_length=0.1)


def Ex2():
    n = 3
    I = 2
    alpha = np.array([0.5, 0.5])
    a = np.array([[5.0,5.0],[6.0,6.0]])
    b = np.array([[0.4,0.2],[0.2,0.4]])
    eps = 0.1
    e = np.array([[10-eps,0,8],[eps,10,2]])
    Econ = Economy(n,I)
    Econ.alpha = alpha
    Econ.a = a
    Econ.b = b
    Econ.allocations = e
    Econ.e = e
    Econ.pag = Econ.evalp(Econ.allocations)
    #delta = 5e0*np.random.rand(Econ.n-1,Econ.I)
    delta = 5e0*np.ones_like(Econ.pag)
    Econ.delta = delta
    return Econ


if __name__ == "__main__":
    np.random.seed(10)

    exfig = 'pdf'
    Ec_aux = Ex2()
    #p_we, x_we = Ec_aux.Walraseq()
    Econ = {}
    ES = {}
    xopt = {}
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
    Col_c = ['b', 'r', 'g']
    BPT = np.zeros((ntrials,Ec_aux.n-1))
    Prices_wal = np.zeros((ntrials,Ec_aux.n))
    Wealth_bte = np.zeros((ntrials,Ec_aux.I))
    Utilities_bte = np.zeros((ntrials,Ec_aux.I))
    Wealth_wal = np.zeros((ntrials,Ec_aux.I))
    Utilities_wal = np.zeros((ntrials,Ec_aux.I))
    p0 = np.ones(Ec_aux.n)
    p0[1:] = np.max(Ec_aux.pag, axis=0)
#    print(p0)
#    p_wal = Ec_aux.Walrasdyn(p0)
#    print(p_wal,Ec_aux.Walrascheck(p_wal))

    for k in range(ntrials):
        print('Trial number {}'.format(k))
        Econ[k] = Ex2()
        Econ[k].e[0,2] -= 6.0*k
        Econ[k].e[1,2] += 6.0*k
        Econ[k].allocations = Econ[k].e
        Econ[k].pag = Econ[k].evalp(Econ[k].allocations)
        #Walras_prices[:,k], Walras_alloc[k] = Econ[k].Walraseq()
        print(Econ[k].e)
        Walras_alloc[k] = np.zeros_like(Econ[k].allocations)
        t = time.time()
        EvPag[k], EvAlloc[k], EvDelta[k], TimesChange[k], K[k], eq_status[k] = Econ[k].Bilateral(eps_prices = 1e-4, MAXIT = 1000, lbda = 0.5, delta_tol = 1e-18, inspection='det')
        ExTimes[k] = time.time()-t
        print('Total time {}[s]'.format(ExTimes[k]))
        Evpbar[1:,k] = np.max(Econ[k].pag,axis=0)
        EvSD[k] = np.max(np.std(Econ[k].pag,axis=0))
        print('Max_i Std of p_ij of {}'.format(EvSD[k]))
        ES[k], xopt[k] = Econ[k].Walrascheck(Evpbar[:,k])#,Econ[k].allocations)

        for i in range(Econ[k].I):
            Z = np.zeros((Econ[k].I,K[k]))
            plt.figure()
            for kk in range(K[k]):
                for ii in range(Econ[k].I):
                    Z[ii,kk] = EvAlloc[k][kk][ii,0]
                    for nn in range(Econ[k].n-1):
                        Z[ii,kk] += (EvAlloc[k][kk][ii,nn+1])*(EvPag[k][kk][ii,nn])
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
            Utilities_bte[k,i] = Econ[k].allocations[i,0]**Econ[k].alpha[i] + np.sum(Econ[k].a[i,:]*Econ[k].allocations[i,1:]-0.5*Econ[k].b[i,:]*Econ[k].allocations[i,1:]**2)
            Wealth_wal[k,i] = np.sum(Walras_prices[:,k]*Walras_alloc[k][i,:])
            Utilities_wal[k,i] = Walras_alloc[k][i,0]**Econ[k].alpha[i] + np.sum(Econ[k].a[i,:]*Walras_alloc[k][i,1:]-0.5*Econ[k].b[i,:]*Walras_alloc[k][i,1:]**2)
        gc.collect()

    print('Number of eqs {} out of {}'.format(np.sum(eq_status),ntrials))
    print("Median of K {}".format(np.median(K)))
    print("Median of Time {}".format(np.median(ExTimes)))
    #print("Median of trade operations {}".format(np.median(TimesChange)))

    plt.figure()
    plt.boxplot(BPT, showfliers=False)
    plt.title(r'Boxplot for equilibrium price thresholds $\bar{p}_{\cdot j}$')
    plt.xlabel(r'Goods $j$')
    plt.ylabel(r'$p$')
    plt.savefig('Pag.'+exfig)
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

    plt.figure()
    x = np.linspace(np.amin([Utilities_wal,Utilities_bte]),np.amax([Utilities_wal,Utilities_bte]),100)
    for k in range(ntrials):
        plt.scatter(Utilities_bte[k,:],Utilities_wal[k,:],label='Trial {}'.format(k+1))
    plt.plot(x,x,linestyle='--')
    plt.xlabel('BTE Utility')
    plt.ylabel('Walras Utility')
    plt.legend(loc='upper left')
    plt.savefig('Utilities.'+exfig)

    plt.figure()
    x = np.linspace(np.amin([Wealth_wal,Wealth_bte]),np.amax([Wealth_wal,Wealth_bte]),100)
    for k in range(ntrials):
        plt.scatter(Wealth_bte[k,:],Wealth_wal[k,:],label='Trial {}'.format(k))
    plt.plot(x,x,linestyle='--')
    plt.xlabel('BTE Wealth')
    plt.ylabel('Walras Wealth')
    plt.legend(loc='upper left')
    plt.savefig('Wealth.'+exfig)
    plt.close()


    Gini_bte = np.zeros(ntrials)
    Gini_wal = np.zeros(ntrials)
    #for k in range(ntrials):
    #    Gini_bte[k] = gini(Wealth_bte[k,:])
    #    Gini_wal[k] = gini(Wealth_wal[k,:])

    plt.figure()
    x = np.linspace(np.amin([Gini_bte, Gini_wal]),np.amax([Gini_bte, Gini_wal]),100)
    for k in range(ntrials):
        plt.scatter(Gini_bte[k],Gini_wal[k],label='Trial {}'.format(k))
    plt.plot(x,x,linestyle='--')
    plt.xlabel('Gini BTE')
    plt.ylabel('Gini Walras')
    plt.legend(loc='lower right')
    plt.savefig('Gini.'+exfig)
    plt.close()
    #
    #
    #
    #
    '''
    #lnst = ntrials*['-']
    #lnst = ['-', '--', '..', '-.', ':','-', '--', '-.', ':','-', '--', '-.'	, ':','-', '--', '-.', ':']
    lnst = ['solid', 'dotted', 'dashed']*ntrials
    #cols = [[(130/235,36/235,51/235)],[(0,50/235,90/235)],[(130/235,120/235,111/235)],[(0/235,173/235,208/235)],[(17/235,28/235,36/235)],[(220/235,220/235,220/235)],[(0,128/235,128/235)]]
    cols = [(130/235,36/235,51/235),(0,50/235,90/235),(130/235,120/235,111/235),(0/235,173/235,208/235),(17/235,28/235,36/235),(220/235,220/235,220/235),(0,128/235,128/235)]
    #
    delta = 0.0025
    s = np.sum(Ec_aux.e,axis=0) # Total suppy
    x0 = np.arange(0, s[1], delta)
    x1 = np.arange(0, s[2], delta)
    X0, X1 = np.meshgrid(x0, x1)
    fig, ax = plt.subplots()
    #plt.style.use('seaborn-white')
    Z1 =  (Ec_aux.e[0,0])**(Ec_aux.alpha[0]) +  Ec_aux.a[0,0]*X0 +  Ec_aux.a[0,1]*X1 - 0.5*Ec_aux.b[0,0]*X0**2 - 0.5*Ec_aux.b[0,1]*X1**2
    Z2 =  (Ec_aux.e[1,0])**(Ec_aux.alpha[1]) +  Ec_aux.a[1,0]*(s[1]-X0) +  Ec_aux.a[1,1]*(s[2]-X1) - 0.5*Ec_aux.b[1,0]*(s[1]-X0)**2 - 0.5*Ec_aux.b[1,1]*(s[2]-X1)**2
    #CS1 = ax.contour(X0, X1, Z1, colors='b', linewidths=0.05)
    #CS2 = ax.contour(X0, X1, Z2, colors='k', linewidths=0.05)
    #ax.clabel(CS1, inline=1, fontsize=6)
    #ax.clabel(CS2, inline=1, fontsize=6)
    for k in range(ntrials):
        XX = np.zeros(K[k])
        YY = np.zeros(K[k])
        for kk in range(K[k]):
            XX[kk] = EvAlloc[k][kk][0,1]
            YY[kk] = EvAlloc[k][kk][0,2]
        plt.plot(XX, YY, label=(r'Trial $\nu={}$'.format(k)),linewidth=1.5,linestyle=lnst[k],color=str((k+1)/(ntrials+2)))
        lbl0 = r'$x^{0,'+'{}'.format(k)+'}$'
        lblbx = r'$\bar{x}^{'+'{}'.format(k)+'}$'
        plt.annotate(lbl0,xy=(XX[0]+0.01,YY[0]+0.2*(1-2*k)))
        plt.annotate(lblbx,xy=(XX[-1],YY[-1]-0.2*(1-2*k)))
        for kd in range(1,K[k]-100,K[k]//15):
            plt.arrow(XX[kd], YY[kd], (XX[kd]-XX[kd-1])/10, (YY[kd]-YY[kd-1])/10, shape='full', lw=0, length_includes_head=True, head_width=.1, color=str((k+1)/(ntrials+2)))
    plt.annotate(r'Agent 1',xy=(0.01,1+0.1))
    plt.annotate(r'Agent 2',xy=(3-0.3,9-0.3))
    plt.annotate(r'Trial 0',xy=(0.01,7.5))
    plt.annotate(r'Trial 1',xy=(0.01,3))
    plt.xlabel(r'Good 1')
    plt.ylabel(r'Good 2')
    ax.set_title(r'Edgeworth box, goods 1 and 2')
#    ax.set_xlim(0.0, s[1])
#    ax.set_ylim(0.0, s[2])
    ax.set_xlim(0.0, 3)
    ax.set_ylim(1.0, 9)
    handles, labels = ax.get_legend_handles_labels()
#    ax.legend(handles, labels, frameon=False)
    ax.set_facecolor('#ebecf0')
    plt.grid(color = '#fdfdfe', linestyle = '--', linewidth = 0.05)
    plt.tight_layout()
    plt.savefig('Edgebox12.'+exfig)
    plt.close()
    #
    #
    #
    #
    #
    '''
    x0 = np.arange(0, s[0], delta)
    x1 = np.arange(0, s[1], delta)
    X0, X1 = np.meshgrid(x0, x1)
    Z1 =  (X0)**(Ec_aux.alpha[0]) +  Ec_aux.a[0,0]*X1 +  Ec_aux.a[0,1]*Ec_aux.e[0,2] - 0.5*Ec_aux.b[0,0]*X1**2 - 0.5*Ec_aux.b[0,1]*Ec_aux.e[0,2]**2
    Z2 =  (s[0]-1)**(Ec_aux.alpha[1]) +  Ec_aux.a[1,0]*(s[1]-X1) +  Ec_aux.a[1,1]*Ec_aux.e[1,2] - 0.5*Ec_aux.b[1,0]*(s[1]-X1)**2 - 0.5*Ec_aux.b[1,1]*(Ec_aux.e[1,2])**2
    fig, ax = plt.subplots()
    #CS1 = ax.contour(X0, X1, Z1, colors='b', linewidths=0.05)
    #CS2 = ax.contour(X0, X1, Z2, colors='k', linewidths=0.05)
    #ax.clabel(CS1, inline=1, fontsize=6)
    #ax.clabel(CS2, inline=1, fontsize=6)
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
    plt.savefig('Edgebox01.'+exfig)
    plt.close()
    #
    #
    #
    #
    #
    x0 = np.arange(0, s[0], delta)
    x1 = np.arange(0, s[2], delta)
    X0, X1 = np.meshgrid(x0, x1)
    Z1 =  (X0)**(Ec_aux.alpha[0]) +  Ec_aux.a[0,0]*Ec_aux.e[0,1] +  Ec_aux.a[0,1]*X1 - 0.5*Ec_aux.b[0,0]*Ec_aux.e[0,1]**2 - 0.5*Ec_aux.b[0,1]*X1**2
    Z2 =  (s[0]-1)**(Ec_aux.alpha[1]) +  Ec_aux.a[1,0]*Ec_aux.e[1,1] +  Ec_aux.a[1,1]*(s[1]-X1) - 0.5*Ec_aux.b[1,0]*(Ec_aux.e[1,1])**2 - 0.5*Ec_aux.b[1,1]*((s[1]-X1))**2

    fig, ax = plt.subplots()
    #plt.style.use('seaborn-white')
    #CS1 = ax.contour(X0, X1, Z1, colors='b', linewidths=0.05)
    #CS2 = ax.contour(X0, X1, Z2, colors='k', linewidths=0.05)
    #ax.clabel(CS1, inline=1, fontsize=6)
    #ax.clabel(CS2, inline=1, fontsize=6)
    ax.set_title('Edgeworth box')
    for k in range(ntrials):
        XX = np.zeros(K[k])
        YY = np.zeros(K[k])
        for kk in range(K[k]):
            XX[kk] = EvAlloc[k][kk][0,0]
            YY[kk] = EvAlloc[k][kk][0,1]
            #plt.scatter(EvAlloc[kk][jj][0,1],EvAlloc[kk][jj][0,2])
        plt.plot(XX, YY, label=("Example {}".format(k+1)),linewidth=1.5,linestyle=lnst[k])
        plt.annotate(r'$x^{0}$',xy=(XX[0],YY[0]))
        plt.annotate(r'$\bar{x}$',xy=(XX[-1],YY[-1]-0.2))
    plt.xlabel('Good 0')
    plt.ylabel('Good 2')
    ax.set_xlim(0.0, s[0])
    ax.set_ylim(0.0, s[2])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,  frameon=False)
    plt.savefig('Edgebox02.'+exfig)
    plt.close()
    '''

    plt.figure()
    for k in range(ntrials):
        plt.scatter(Evpbar[1,k],Evpbar[2,k],label='Trial {}'.format(k+1))
    plt.xlabel(r'$p_1$')
    plt.ylabel(r'$p_2$')
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig('Pbar.'+exfig)

    fig, ax = plt.subplots()
    for k in range(ntrials):
        XX = np.zeros((Econ[k].I,K[k]))
        YY = np.zeros((Econ[k].I,K[k]))
        for kk in range(K[k]):
            XX[:,kk] = EvPag[k][kk][:,0]
            YY[:,kk] = EvPag[k][kk][:,1]
            #plt.scatter(EvAlloc[kk][jj][0,1],EvAlloc[kk][jj][0,2])
        for i in range(Econ[k].I):
            line = plt.plot(XX[i,:], YY[i,:], label=("Trial {} Agent {}".format(k,i+1)),linewidth=1,linestyle=lnst[k], color=str((k+1)/(ntrials+2)))#,marker='o')
            for kd in range(1,K[k]-150*(1+2*k),5*(1+5*k)):
                plt.arrow(XX[i,kd], YY[i,kd], XX[i,kd]-XX[i,kd-1], YY[i,kd]-YY[i,kd-1], shape='full', lw=0, length_includes_head=True, head_width=.1, color=str((k+1)/(ntrials+2)))
            annstr = r'$p^{0,%i}_{%i}$'%(k,i+1)
            plt.annotate(annstr,xy=(XX[i,0]-1.0,YY[i,0]+2*i*(k-1)),fontsize=8)#,color=cols[i])
            plt.scatter(XX[i,0],YY[i,0],marker='o',color=str((k+1)/(ntrials+2)))
            if eq_status[k] == 1:
                if i ==0 :
                    plt.scatter(XX[i,-1],YY[i,-1],marker='+',color=str((k+1)/(ntrials+2)))
                    annstr = r'${\bar p}^{%i}$'%(k)
                    #annstr = r'Equilibrium trial ${%i}$'%(k)
                    plt.annotate(annstr,xy=(XX[i,-1]-0.25,YY[i,-1]-0.4),fontsize=11)
            else:
                plt.scatter(XX[i,-1],YY[i,-1],marker='x',color=str((k+1)/(ntrials+2)))
                annstr = r'$\bar p^{%i}_{%i}$'%(k,i+1)
                plt.annotate(annstr,xy=(XX[i,-1]-0.25*(1-i),YY[i,-1]-0.4),fontsize=11)
    plt.annotate('Trial 0, agent 1',xy=(16,21),fontsize=8)#,color=cols[i])
    plt.annotate('Trial 0, agent 2',xy=(22.5,16),fontsize=8)#,color=cols[i])
    plt.annotate('Trial 1, agent 1',xy=(17,12.2),fontsize=8)#,color=cols[i])
    plt.annotate('Trial 1, agent 2',xy=(22.5,22),fontsize=8)#,color=cols[i])
    plt.title(r'Price thresholds  $\{(p^{\nu,k}_{i1},p^{\nu,k}_{i2})\}_k$, agent $i$, trial $\nu$')
    plt.xlabel(r'Price threshold for good $l=1$ ($p_1$)')
    plt.ylabel(r'Price threshold for good $l=2$ ($p_2$)')
#    plt.legend(loc='upper right',prop={"size":8}, frameon=False)
    ax.set_xlim(16.0, 24)
    ax.set_ylim(12.0, 24)
    plt.tight_layout()
    ax.set_facecolor('#ebecf0')
    plt.grid(color = '#fdfdfe', linestyle = '--', linewidth = 0.05)
#    plt.style.use('grayscale')
    plt.savefig('Evbarp.'+exfig)
    plt.close()

    writer = pd.ExcelWriter('Evs.xlsx')  #Key 2, create an excel sheet named hhh
    for k in range(ntrials):
        XX = np.zeros((K[k],11))
        for j in range(K[k]):
            a = np.array([j,EvPag[k][j][0,0],EvPag[k][j][0,1],EvPag[k][j][1,0],EvPag[k][j][1,1],EvAlloc[k][j][0,0],EvAlloc[k][j][0,1],EvAlloc[k][j][0,2],EvAlloc[k][j][1,0],EvAlloc[k][j][1,1],EvAlloc[k][j][1,2]])
            XX[j,:] = a
        strpg = 'page_{}'.format(k)
        data_df = pd.DataFrame(XX)
        data_df.to_excel(writer,strpg,float_format='%.8f')  #Key 3, float_format controls the accuracy, write data_df to the first page of the hhh form. If there are multiple files, you can write in page_2
    writer.save()


    aux_str=strftime("%Y%m%d_%H%M", localtime())
    pname='Results_'+aux_str
    #os.system('mkdir '+pname)
    #os.system('mv *.'+exfig+' '+pname)
    #os.system('mv *.xlsx '+pname)
    #os.system('cp BTEQuad.py '+pname)
    #os.system('cp Ex2.py '+pname)

    Tab7 = np.zeros((Ec_aux.I*ntrials,2*(Ec_aux.n-1)))
    for kk in range(ntrials):
        for ii in range(Ec_aux.I):
            Tab7[Ec_aux.I*kk+ii,0] = EvPag[kk][0][ii,0]
            Tab7[Ec_aux.I*kk+ii,1] = EvPag[kk][K[kk]-1][ii,0]
            Tab7[Ec_aux.I*kk+ii,2] = EvPag[kk][0][ii,1]
            Tab7[Ec_aux.I*kk+ii,3] = EvPag[kk][K[kk]-1][ii,1]

    idags7 = np.asarray(['$\nu=0,i=1$','$\nu=0,i=2$','$\nu=1,i=1$','$\nu=1,i=2$'])
    labls7 = ['$ p^{0,\nu}_{i1}$','$\overline p^{\nu}_{i1}$','$ p^{0,\nu}_{i2}$','$\overline p^{\nu}_{i2}$']
    df7 = pd.DataFrame(Tab7,columns=labls7, index=idags7)
    df7.to_latex('Tab7_results.tex', float_format="{:0.4f}".format, escape=False)
