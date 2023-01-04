from __future__ import division
import numpy as np
import random
import time
from pyomo.environ import *
from pyomo.opt import SolverFactory,SolverStatus,TerminationCondition
from pyutilib.misc.timing import tic,toc

def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

#This routine implements the Bilateral trades algorithm for economies with #
#Cobb-Douglas-type utility functions

class Economy:
    def __init__(self,n,I):
        #        The Spline element is iniatilized with the "skeleton": the lower limit (vector),
        #        upper limit (vector), and the mesh size in each dimension
        Economy.n = n
        Economy.I = I
        Economy.pag = np.zeros((self.I,self.n-1))
        #Economy.prices = np.zeros(self.n-1)
        Economy.delta = np.zeros((self.I,self.n-1))
        Economy.allocations = np.zeros((self.I,self.n))
        Economy.e = np.zeros((self.I,self.n))
        Economy.alpha = np.zeros((self.I,self.n))

    def evalp(self,x):
        A = np.zeros((self.I,self.n-1))
        for i in range(self.I):
            for j in range(self.n-1):
                A[i,j] = (self.alpha[i,j+1]/self.alpha[i,0])*(x[i,0]/x[i,j+1])
        return A

    #\xi_j^{-}
    def xshort(self,i,j,pit):
        num = self.alpha[i,j+1]*self.allocations[i,0]-pit*self.alpha[i,0]*self.allocations[i,j+1]
        den = pit*(self.alpha[i,j+1]+self.alpha[i,0])
        xin = num/den
        ut = lambda xi: np.prod( (self.allocations[i,:] + xi)**self.alpha[i,:] )
        xi = np.array([0.0,xin])
        pij = np.zeros_like(self.allocations[i,:])
        pij[0] = -pit
        pij[j+1] = 1.0
        uts = np.array([ut(pij*xi[0]),ut(pij*xi[1])])
        return xi[np.argmax(uts)]

    #\xi_j^{+}
    def xlong(self,i,j,pit):
        num = self.alpha[i,0]*pit*self.allocations[i,j+1]-self.alpha[i,j+1]*self.allocations[i,0]
        den = pit*(self.alpha[i,j+1]+self.alpha[i,0])
        xip = num/den
        ut = lambda xi: np.prod( (self.allocations[i,:] + xi)**self.alpha[i,:] )
        xi = np.array([0.0,xip,self.allocations[i,j+1]])
        pij = np.zeros_like(self.allocations[i,:])
        pij[0] = pit
        pij[j+1] = -1.0
        uts = np.array([ut(pij*xi[0]),ut(pij*xi[1]),ut(pij*xi[2])])
        return xi[np.argmax(uts)]

    def Bilateral(self,eps_prices,MAXIT,lbda,delta_tol,inspection):
        EvPag = {}
        EvAlloc = {}
        EvDelta = {}
        TimesChange = {}
        KK = 0
        eq_status = 0
        for K in range(MAXIT):
            KK = K
            trade_aux = 0
            if np.max(np.std(self.pag,axis=0)) < eps_prices:
                print('Equilibrium found! The max std of p_ij is {}, less than the given tolerance of {}'.format(np.max(np.std(self.pag,axis=0)),eps_prices))
                eq_status = 1
                break
            elif np.max(self.delta) < delta_tol:
                print('Delta less than the given tolerance of {}'.format(delta_tol))
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
                                pit = np.min([pip[i1,j],pin[i2,j]])+l_aux*np.abs(pip[i1,j]-pin[i2,j])
                                xin = self.xshort(i2,j,pit)
                                xip = self.xlong(i1,j,pit)
                                xij = np.minimum(xin,xip)
                                if pip[i1,j]-pin[i2,j] <= -1e-12 \
                                    and self.allocations[i1,j+1] > 1e-12 \
                                    and xij > 1e-12:
                                    #print('Trade!')
                                    trade_aux += 1
                                    self.allocations[i1,0] += xij*pit
                                    self.allocations[i1,j+1] += -xij
                                    self.allocations[i2,0] += -xij*pit
                                    self.allocations[i2,j+1] += xij
                                    self.pag = self.evalp(self.allocations)
                if trade_aux == 0:
                    self.delta = lbda*self.delta
                else:
                    TimesChange[KK] = trade_aux
#                else:
#                    print('Trade of {}'.format(trade_aux))
            if KK == (MAXIT-1):
                print('Number of iterations hits the maximum given threshold of {}'.format(MAXIT))
            EvPag[KK] = np.copy(self.pag)
            EvAlloc[KK] = np.copy(self.allocations)
            EvDelta[KK] = np.copy(self.delta)
        return EvPag, EvAlloc, EvDelta, TimesChange, KK, eq_status

    def Walrascheck(self,p):
        x = np.zeros_like(self.e)
        for ii in range(self.I):
            model = ConcreteModel('Agent problem')
            model.n = Set(initialize=range(self.n))
            def alpha_init(model,j):
                return self.alpha[ii,j]
            model.alpha = Param(model.n, initialize=alpha_init)
            def e_init(model,j):
                return self.e[ii,j]
            model.e = Param(model.n,initialize=e_init)
            def p_init(model,j):
                return p[j]
            model.p = Param(model.n,initialize=p_init)
            model.x = Var(model.n, within=NonNegativeReals,initialize=e_init)
            def obj_rule(model):
#                return sum( model.alpha[j]*log(model.x[j]) for j in model.n )
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

    def Walraseqrtr(self):
        nn = self.I+self.n-1
        xx = np.zeros_like(self.e)
        p = np.ones(self.n)
        B = np.sum(self.alpha,axis=1)
        E = np.sum(self.e, axis=0)
        A = np.zeros((nn,nn))
        b = np.zeros(nn)
        A11 = np.diag(B)
        A22 = np.diag(-E[1:])
        A12 = -self.e[:,1:]
        A21 = self.alpha[:,1:].T
        A = np.block([[A11, A12],[A21, A22]])
        b[:self.I] = self.e[:,0]
        xsol = np.linalg.solve(A,b)
        #print(xsol)
        p = np.block([1.0,xsol[self.I:]])
        for i in range(self.I):
            for j in range(self.n):
                if j == 0:
                    xx[i,j] = xsol[i]
                else:
                    xx[i,j] = self.alpha[i,j]*xx[i,0]/(self.alpha[i,0]*p[j])
        #print(np.sum(xx,axis=0))
        return p, xx


def gini(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
