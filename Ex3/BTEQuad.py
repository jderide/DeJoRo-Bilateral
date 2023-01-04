from __future__ import division
import numpy as np
from scipy import optimize
import random
import time
from pyomo.environ import *
from pyomo.opt import SolverFactory,SolverStatus,TerminationCondition
from pyutilib.misc.timing import tic,toc

def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

nlsol = 'knitroampl'
#This routine implements the Bilateral trades algorithm for economies with #
#Cobb-Douglas-type utility functions

class Economy:
    def __init__(self,n,I):
        Economy.n = n
        Economy.I = I
        Economy.pag = np.zeros((self.I,self.n-1))
        #Economy.prices = np.zeros(self.n-1)
        Economy.delta = np.zeros((self.I,self.n-1))
        Economy.allocations = np.zeros((self.I,self.n))
        Economy.e = np.zeros((self.I,self.n))
        Economy.a = np.zeros((self.I,self.n-1))
        Economy.b = np.zeros((self.I,self.n-1))
        Economy.alpha = np.zeros(self.I)

    def evalp(self,x):
        A = np.zeros((self.I,self.n-1))
        for i in range(self.I):
            for j in range(self.n-1):
                if x[i,0] < 1e-18:
                    print('xi0 too small')
                    num = np.copy(self.a[i,j] - self.b[i,j]*x[i,j+1])
                    den = np.copy(self.alpha[i]*(1e-18)**(self.alpha[i]-1.0))
                    A[i,j] = num / den
                else:
                    num = np.copy(self.a[i,j] - self.b[i,j]*x[i,j+1])
                    den = np.copy(self.alpha[i]*(x[i,0])**(self.alpha[i]-1.0))
                    A[i,j] = num / den
        return A

    # xshort: \xi^-
    def xshort(self,i,j,pit):
        ut = lambda xi: -1.0*((self.allocations[i,0] - pit*xi)**(self.alpha[i]) + self.a[i,j]*(self.allocations[i,j+1] + xi) - 0.5*self.b[i,j]*(self.allocations[i,j+1] + xi)**2)
        LB = 0.0
        UB = (np.copy(self.allocations[i,0])/pit)
        result = optimize.minimize_scalar(ut, bounds=(LB,UB), method='bounded')
        #print(result.x)
        return result.x

    # xlong: \xi^+
    def xlong(self,i,j,pit):
        ut = lambda xi: -1.0*((self.allocations[i,0] + pit*xi)**(self.alpha[i]) + self.a[i,j]*(self.allocations[i,j+1] - xi) - 0.5*self.b[i,j]*(self.allocations[i,j+1] - xi)**2)
        LB = 0.0
        UB = np.copy(self.allocations[i,j+1])
        result = optimize.minimize_scalar(ut, bounds=(LB,UB), method='bounded')
        #print(result.x)
        return result.x

    def Bilateral(self,eps_prices,MAXIT,lbda,delta_tol,inspection):
        EvPag = {}
        EvAlloc = {}
        EvDelta = {}
        TimesChange = {}
        KK = 0
        eq_status = 0
        EvPag[0] = np.copy(self.pag)
        EvAlloc[0] = np.copy(self.allocations)
        EvDelta[0] = np.copy(self.delta)
        for K in range(MAXIT):
            trade_aux = 0
            if np.max(np.std(self.pag,axis=0)) < eps_prices:
                print('Equilibrium found! The max std of p_ij is {}, less than the given tolerance of {}'.format(np.max(np.std(self.pag,axis=0)),eps_prices))
                eq_status = 1
                break
            elif np.max(self.delta) < delta_tol:
                print('Delta (max delta ={}) less than the given tolerance of {}'.format(np.max(self.delta),delta_tol))
                break
            else:
                if inspection =='ran':
                    #i1 = random.choice(range(self.I))
                    agents_insp1 = randomly(range(self.I))
                    agents_insp2 = randomly(range(self.I))
                    goods_insp = randomly(range(self.n-1))
                elif inspection =='det':
                    #i1 = 0
                    agents_insp1 = (range(self.I))
                    agents_insp2 = (range(self.I))
                    goods_insp = (range(self.n-1))
                else:
                    agents_insp1 = (range(self.I))
                    agents_insp2 = (range(self.I))
                    goods_insp = (range(self.n-1))
                l_aux = 0.5
                for i1 in agents_insp1:
                    for i2 in agents_insp2:
                        for j in goods_insp:
                            if i1 != i2:
                                pip = self.pag + self.delta
                                pin = self.pag - self.delta
                                pit = np.copy(np.min([pip[i1,j],pin[i2,j]])+l_aux*np.abs(pip[i1,j]-pin[i2,j]))
                                xin = self.xshort(i2,j,pit)
                                xip = self.xlong(i1,j,pit)
                                xij = np.minimum(xin,xip)
                                if pip[i1,j]-pin[i2,j] <= -1e-18 \
                                    and self.allocations[i1,j+1] > 1e-18 \
                                    and xij > 1e-18:
                                    #print('Trade!')
                                    trade_aux += 1
                                    self.allocations[i1,0] += xij*pit
                                    self.allocations[i1,j+1] += -xij
                                    self.allocations[i2,0] += -xij*pit
                                    self.allocations[i2,j+1] += xij
                                    self.pag = np.copy(self.evalp(self.allocations))
                if trade_aux == 0:
                    self.delta = np.copy(lbda*self.delta)
                else:
                    TimesChange[K] = trade_aux
#                else:
#                    print('Trade of {}'.format(trade_aux))
            EvPag[K+1] = np.copy(self.pag)
            EvAlloc[K+1] = np.copy(self.allocations)
            EvDelta[K+1] = np.copy(self.delta)
            if K == (MAXIT-1):
                print('Number of iterations hits the maximum given threshold of {}'.format(MAXIT))
                print(self.pag, self.delta)
        KK = K+1
        return EvPag, EvAlloc, EvDelta, TimesChange, KK, eq_status

    def Walrascheck(self,p):
        x = np.zeros_like(self.e)
        for ii in range(self.I):
            model = ConcreteModel('Agent problem')
            model.n = Set(initialize=range(self.n))
            def alpha_init(model):
                return self.alpha[ii]
            model.alpha = Param(initialize=alpha_init)
            def a_init(model,j):
                if j ==0:
                    return 0.0
                else:
                    return self.a[ii,j-1]
            model.a = Param(model.n, initialize=a_init)
            def b_init(model,j):
                if j ==0:
                    return 0.0
                else:
                    return self.b[ii,j-1]
            model.b = Param(model.n, initialize=b_init)
            def e_init(model,j):
                return self.e[ii,j]
            model.e = Param(model.n, initialize=e_init)
            def p_init(model,j):
                if j ==0:
                    return 1.0
                else:
                    return p[j-1]
            model.p = Param(model.n,initialize=p_init)
            def _bounds_rule(model, j):
                if j == 0:
                    return (1e-20,None)
                else:
                    return (0.0,None)
            def x_init(model,j):
                #print(self.allocations[ii,j])
                return self.allocations[ii,j]
            model.x = Var(model.n, bounds=_bounds_rule, initialize=x_init)
            def obj_rule(model):
                exp1 = (model.x[0])**model.alpha
                exp2 = sum( model.a[j]*model.x[j] - 0.5*model.b[j]*model.x[j]**2 for j in model.n )
                return exp2 + exp1
                return prod( (model.x[j])**model.alpha[j] for j in model.n )
            model.obj = Objective(rule=obj_rule, sense=maximize)
            def bc_rule(model):
                return sum( model.p[j]*(model.x[j]-model.e[j]) for j in model.n) <= 0
            model.bc = Constraint(rule=bc_rule)
            opt = SolverFactory('ipopt')
            opt.solve(model)#,tee=True, keepfiles=True)
            for j in range(self.n):
                x[ii,j] = value(model.x[j])
        ES = np.sum(self.e-x,axis=0)
        if np.max(np.abs(ES)) < 0.01*np.max(np.sum(self.e,axis=0)):
            print('These prices form a Walras equilibrium')
        else:
            print('These prices do not form a Walras equilibrium')
        #print(ES)
        return ES, x

    def Walraseq(self):
        x = np.zeros_like(self.allocations)
        p = np.ones(self.n)
        B = np.sum(self.alpha,axis=1)
        E = np.sum(self.e, axis=0)
        A = np.zeros((self.n,self.n))
        b = np.zeros(self.n)
        for j in range(self.n):
            for l in range(self.n):
                if j == l:
                    A[j,l] = E[j]-np.sum(self.alpha[:,j]*self.e[:,l]/B[:])
                else:
                    A[j,l] = -np.sum(self.alpha[:,j]*self.e[:,l]/B[:])
            b[j] = np.sum(self.alpha[:,j]*self.e[:,0]/B[:])
        p = np.linalg.solve(A,b)
        p = p/p[0]
        for i in range(self.I):
            ul = sum(self.allocations[i,:]*p)/B[i]
            for j in range(self.n):
                x[i,j] = ul*self.alpha[i,j]/p[j]
        return p, x

    def Walrasdyn(self,p0):
        p = np.copy(p0)
        ES, x = self.Walrascheck(p)
        ld = 0.01
        MAXIT = 1000
        for k in range(MAXIT):
            if np.min(ES) >= -1e-3:
                print('Walras equilibrium')
                return p
            else:
                p = p + ld*ES
                ES, x = self.Walrascheck(p)
        if k == MAXIT:
            print('Maximum amount of iterations')
        return p

def gini(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
