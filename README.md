# DeJoRo-Bilateral
Official repository for the paper (link)
## Study Objective:
This repository seeks to implement and thoroughly analyze the bilateral negotiation-based market equilibrium model proposed in the article *"Reaching an Equilibrium of Prices and Holdings of Goods through Direct Buying and Selling"* (Deride et al., 2023), with three main axes: (1) **theoretical validation** of the convergence toward price and holdings equilibria through decentralized transactions, contrasting the results with traditional Walrasian models; (2) **detailed empirical analysis** of Examples 1, 2, and 3 of the article, exploring how variables such as randomness in the trading order, the structure of utility functions (Cobb-Douglas vs. non-Cobb-Douglas), and the initial distribution of goods (including cases with zero holdings) affect the dynamics and stability of the equilibrium; and (3) **model extension** to overcome identified limitations (such as the assumption of positive holdings across all assets) by proposing algorithmic adjustments to handle realistic scenarios involving non-core assets. The repository will include replicable simulations, interactive visualizations of price and holdings developments, and technical documentation linking the results to decentralized equilibrium economic theories, thus providing a practical tool for studying markets based on direct interactions between agents.
### 1. Theoretical validation
According to the supporting paper "Reaching an equilibrium of prices and holdings of goods through direct buying and selling", a theoretical comparison is presented that demonstrates the superiority of decentralized methods over the classical Walrasian model in terms of economic realism, practical implementation, and convergence to stable equilibria.
#### Walrasian Model
There are three criticisms of the Walrasian model, which has fundamental problems:
- **Unrealistic centralization:** Requires a "ian auctioneer" that sets prices abstractly, without real market mechanisms (section 1 of the paper)
- **Absence of money and bilateral transactions:** Prices are relative (non-monetary), and transfers occur in a "large simultaneous exchange," ignoring the dynamics of buying and selling with money.
- **Lack of dynamic guarantees:** price adjustment may not converge, except under restrictive conditions.
- **Relative, non-monetary prices:** In Walras, prices are exchange ratios between goods (e.g., 1 apple = 2 oranges), but they do not incorporate money as a universal means of payment. Therefore, they cannot explain phenomena such as liquidity or the role of money as a store of value.
- **Static vs. Dynamic:** Walras describes a static equilibrium (without history or trajectory), while the decentralized approach captures the dynamic evolution of prices and holdings.
- **Frictionless Markets:** Walras assumes no transaction costs, perfect information, and infinitely divisible goods. This is incompatible with real markets, where trading involves costs.
#### BTE Model
The model proposed in the paper solves these Walras limitations by:
- **Bilateral money negotiations:** Agents exchange one good at a time for money, reflecting real markets. Prices arise from thresholds based on marginal utility, not from a central auctioneer.
- **Proven convergence to equilibrium:** Theorem 3 shows that, under general conditions, iterative transactions lead to a price and holdings equilibrium as a real limit (not just an accumulation point). In contrast, the Walrasian model does not guarantee that tatonnement will converge without strong assumptions.
- **Decentralization and self-organization:** Agents act in their immediate self-interest (maximizing utility in each transaction), without the need for central coordination.
Equilibrium emerges organically, even with random trading orders.
+ **Empirical Evidence and Key Results:**
    + Example 1: With Cobb-Douglas utilities, the bilateral method converges to multiple equilibria (depending on the trading order), all distinct from Walrasian prices. This reflects the path dependence typical of real markets.
    + Example 3: By relaxing the assumption of positive holdings (non-core goods), the model still achieves equilibria in some cases, while the Walrasian approach cannot handle these situations without ad hoc adjustments.
+ **Limitations of the decentralized approach:**

  The paper acknowledges that:
    + Convergence requires sufficiently regular utilities (strong concavity).
    + In extreme cases, the absence of initial holdings of an asset can stall the process, suggesting the need for additional rules (reactivation of dormant assets).
#### In summary
The decentralized model outperforms the Walrasian model in:
- **Realism:** Dynamics based on bilateral money transactions.
- **Robustness:** Convergence tested under general conditions.
- **Flexibility:** Adaptability to scenarios with zero holdings or heterogeneous preferences. While Walras is useful as a theoretical abstraction, the decentralized approach provides a computationally and economically viable framework for studying real markets.
### 2. Detailed empirical analysis
In this section, the results of each of the three examples will be evaluated, as well as a comparison between the results of the BTE and Walras algorithms. Conclusions will then be drawn based on the graphs of how both algorithms behave. First, each code for each example will be documented to explain how the algorithm works in the code for each example and what its methodology will be. After this, libraries will be installed to run the code in Jupyter Notebook on a Windows 11 system for each of the three codes. After this, the meaning of each of the graphs produced by each of the three codes will be explained, and how the BTE method varied based on Walras will be explained, in order to reach final conclusions for each of the three examples.
### Example 1
#### Explanation
Example 1 presents an economy with 9 non-monetary goods (plus money as good 0) and 5 agents, where each agent has a Cobb-Douglas utility function defined as $u_i(x_i) = \prod_{j=0}^n x_{ij}^{\beta_{ij}}$, with $0 < \beta_{ij} < 1$ and $\sum_{j=0}^n \beta_{ij} < 1$. The parameters $\beta_{ij}$ and the initial holdings $x_{ij}^0$ are detailed in a table, showing varied distributions of preferences and resources. A Bilateral Trading Scheme was implemented with a random inspection order, and run 50 times. The algorithm converged to a market equilibrium in all runs, with a stopping criterion of $\varepsilon_p = 10^{-6}$, achieving median times of 2.12 seconds and 3093 iterations. Due to randomness, equilibrium prices varied between runs, and results for the four runs with the largest differences are reported, comparing them with Walrasian prices to highlight the conceptual differences between the two approaches. This illustrates how decentralized bilateral trade can reach equilibria without the need for a centralized entity.

#### Code documentation
Here the documentation for example 1 will be placed to know what each part of the code does with its respective commands. we started.
We document BTE first since it is the code needed for example 1 to run.
#### BTE documentation
**line 1 to 7**
```python   
from __future__ import division
import numpy as np
import random
import time
from pyomo.environ import *
from pyomo.opt import SolverFactory,SolverStatus,TerminationCondition
from pyutilib.misc.timing import tic,toc
 ```   
First, import the libraries described in the paper:
- Numpy (used for matrix operations)
- random (used for random processing of matrices)
- Pyomo (used to solve optimization problems, using walrascheck to verify market equilibrium)
- pyutilib (to see the times that have passed for each iteration)

**lines 9 to 12**
```python
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)
```
Implement the random inspection of agents and goods mentioned in part 5 of the paper (random inspection strategy)

**lines 17 to 28**
```python
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
```
The economy class is defined, where an economy is initialized with n goods and i agents, where in the paper model there are n+1 goods plus money. 
The following matrices are initialized with respect to the class created above:
- pag: Prices (related to threshold prices p_ij in the paper)
- delta: Price premium delta_ij to find an equilibrium
- allocations: Current allocations, which can also be seen as the current iteration x_i (holdings)
- e: Initial allocations, not yet iterated (initial holdings x_ij^0)
- alpha: Utility parameter (beta_ij in the paper for Cobb Douglas functions)

**lines 30 to 35**
```python
def evalp(self,x):
        A = np.zeros((self.I,self.n-1))
        for i in range(self.I):
            for j in range(self.n-1):
                A[i,j] = (self.alpha[i,j+1]/self.alpha[i,0])*(x[i,0]/x[i,j+1])
        return A
```
The evalp function is defined, where the agent's threshold prices p_ij(x_i) are calculated. Threshold price is understood as the agent's purchases and sales through an initial vector of goods or initial allocations. It is measured in units of the agent's utility per unit of price. It implements $p_{ij} = (\alpha_{ij}/\alpha_{i0}) \times (x_{i0}/x_{ij})$ for j>0 (equation 2.8 of the paper)

**lines 38 to 48**
```python
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
```
It optimizes utilities by updating the buyers' holdings, iterating their allocations with the definition of xshort by a maximum quantity that an agent i is willing to buy of good j at a price pit. The buyer through his preferences sees the optimal quantity of good j that he buys and if it is better not to buy, with the option not to buy, or 0. Generating the new variable xi, if the buyer through a price pit and his preferences prefers to buy or it is better not to, to update the holdings, and with this the utility with a Cobb-Douglas function and evaluate the different scenarios, until the utility can not be improved more. The buyer makes monetary transactions and receives the goods as can be seen in pij, where he loses money, but receives goods, thus improving his utility depending on his preferences, until they can not be improved more (equation 2.20 of the paper).

**lines 51 to 61**
```python
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
```
It optimizes utilities by updating the sellers' holdings, iterating their allocations with the definition of xlong as a maximum quantity that an agent i is willing to sell of good j at a price pit. In the case of the seller, they will earn money by exchanging their products and this exchange will cause their goods to decrease. The seller sells, excuse the redundancy, according to their preferences. In this way, an xip is generated, which is the optimal quantity at which they will sell their products at a pit price. Now, unlike buyers, with sellers three scenarios are evaluated: not selling anything, selling the optimal quantity, or selling their entire stock. These three options are evaluated with xi and then added to the holdings, to have the new holdings of the seller, and in this way, utility can be calculated and evaluated according to these three scenarios to find the maximum level of utility that the sellers find according to their preferences. Once no more selling opportunities are found, the delta premiums are updated (equation 2.14 of the paper).

**lines 63 to 127**
```python
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
```
The Bilateral method implements a bilateral trade algorithm for economies with Cobb-Douglas utility functions, seeking a price equilibrium where agents exchange goods until the standard deviation of prices (pag) is less than eps_prices or price premiums (delta) fall below delta_tol, with at most MAXIT iterations; it uses random (ran) or sequential (det) inspection to match agents and goods, calculates an intermediate price (pit) between buyer and seller, determines quantities to trade (xij) according to preferences, updates allocations if price and stock conditions are met, reduces delta by a factor lbda if there are no trades, and saves the history of prices, allocations, and premiums to analyze convergence, with random inspection being more efficient and sequential inspection more accurate near equilibrium.

**lines 129 to 161**
```python
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
```
The `Walrascheck` method checks whether a price vector `p` forms a Walras equilibrium for an economy with Cobb-Douglas utility functions. It initializes a matrix `x` to store the optimal demands of agents, solving an optimization problem for each agent using Pyomo's `ConcreteModel`. It defines the goods, parameters (preferences `alpha`, initial endowments `e`, prices `p`), and decision variables (optimal allocations `x`, non-negative). It maximizes Cobb-Douglas utility subject to a budget constraint that ensures that expenditures do not exceed income. It solves the nonlinear problem with the `ipopt` solver, stores the optimal allocations in `x`, and computes excess demand (`ES`) as the difference between initial endowments and optimal demand. If the maximum absolute value of `ES` is less than 1% of the total endowment, it declares a Walras equilibrium; Otherwise, it indicates that the prices do not form it. Returns `ES` and `x` for analysis.

**lines 163 to 183**
```python
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
```
The `Walraseq` method algebraically computes Walras-optimal allocations and equilibrium prices for an economy with Cobb-Douglas utility functions, ensuring that aggregate demand equals aggregate supply. It initializes matrices for allocations (`x`), prices (`p`), aggregate preferences per agent (`B`, sum of `alpha`), and aggregate endowments per good (`E`, sum of `e`). It constructs a matrix `A` where the diagonal represents the net supply of each good less the dependent demand for that good, and the off-diagonal elements capture the impact of endowments of other goods on demand. The vector `b` reflects the dependent demand for money. It solves the linear system `A*p = b` to obtain equilibrium prices (`p`), normalized with respect to the price of money. It then computes the optimal allocations (`x`) for each agent, where the demand for each good (`x[i,j]`) is proportional to their total revenue adjusted by their preferences (`alpha`) and prices. It returns `p` and `x`, satisfying the Walras equilibrium conditions described in section 5 of the article.
[Falta documentar]

**Lines 212 to 216**
```python
def gini(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
```
Finally, the code defines the Gini equation, which calculates absolute differences by iterating over each agent i, calculating the sum of the absolute differences between xi and all subsequent agents j>i. After this, it is normalized by dividing diffsum in n squared by x dash to obtain the Gini coefficient. Interestingly, the Gini coefficient decreases with each iteration, implying that a convergence is being reached across the iterations. It is also useful for evaluating inequality in allocations, that is, to see how equitably the goods between agents are distributed in the results of the bilateral algorithm vs. the Walrasian equilibrium.

Example 1 will now be documented, put in the repository as final 1
#### Final1 documentation
**Lines 1 to 11**
```python
from BTE import *
from copy import deepcopy
import pandas as pd
import gc

from time import perf_counter, strftime,localtime
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
```
The following libraries are imported:
- From BTE import: Imports the economy class and auxiliary functions from BTE.py as bilateral trading.
- From copy import deepcopy: To copy objects without sharing references.
- Import pandas as pd: To export data to Excel.
- Import gc: Garbage Collector (optimizes memory).
- From time: Measures time and handles date formats.
- Import os: System operations, for example, creating folders.
- Import matplotlib: To generate graphs.
- From mpl_toolkits: Support for 3D graphics (not used)

**Lines 13 to 16**
```python
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
```
Styles are configured for charts:
- rc(“font”..): Serves as a font for charts, controlling typographical aspects, setting global parameters, and setting the preferred font for chart letters.
- rc(“text”:::): Enables text rendering using LaTeX, allowing the use of mathematical commands and professional typography.
- plt.rc(“text”..): Reinforces the previous configuration using the pyplot interface. It is redundant but ensures that LaTeX works.
- plt.rc(“font”..): Sets the serif font family for mathematical elements such as Times New Roman.

**Lines 19 to 42**
```python
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
```
We begin by defining example 1 incorporated in the paper, starting with the quantity of initial goods, which in this case are 10 (j = 0, money and j = 1 to 9, goods other than money). Then, a 5x10 matrix of the alpha parameters is placed, which are the preferences of the agents of the Cobb Douglas utility functions. Then, the initial endowments of the agents are placed in a 5x10 matrix with the function e. Then, the Econ class is created, where the economy of n agents and I goods, including money, is created. Econ.alpha assigns the preferences of the agents. Econ.e assigns the initial endowments of the agents. Econ.allocations allocates the agent's assets after iterating the initial endowments for the first time, and subsequent iterations as well. Econ.pag returns the result of each agent's prices after the iterations. Delta updates the delta values ​​when trades are no longer possible, decreasing their value by 90% until it is zero or negligible. Econ.delta allocates the deltas according to the current iteration. The result for the economy in question is then returned. np.random.seed is then created to generate random results between pairs of agents, ensuring they are not ordered.

**Lines 44 to 65**
```python
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
```
This code section initializes an economic simulation to analyze a Walras equilibrium and the bilateral trade algorithm (BTE). It sets the figure output format to PDF (`exfig = 'pdf'`) and creates a trial economy (`Ec_aux`) with 10 goods and 5 agents using the `Ex1()` function. It computes the theoretical Walras equilibrium (`p_we`, `x_we`) with `Walraseqrtr()`. It stores 10 simulations (`ntrials = 10`) in `Econ`, saving prices (`EvPag`), allocations (`EvAlloc`), premiums (`EvDelta`), and number of trades (`TimesChange`) per iteration. It records the total number of iterations (`K`), equilibrium state (`eq_status`, 1 if converged, 0 otherwise), running times (`ExTimes`), maximum price standard deviation (`EvSD`), simulated equilibrium prices (`Evpbar`), Walras prices and allocations (`Walras_prices`, `Walras_alloc`), and maximum prices per good (`BPT`). It calculates agents' wealth (`Wealth_bte`, `Wealth_wal`) and utility (`Utilities_bte`, `Utilities_wal`) for BTE and Walras, generating a 10x5 matrix to compare results. Boxplots show the dispersion of threshold prices, allowing to analyze the convergence and distribution of results between methods.

**Lines 67 to 77**
```python
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
```
This section of code runs a `for` loop that performs `ntrials` simulations to compare the Walras method and the bilateral trade algorithm (BTE). For each `k` iteration, it prints the trial number, creates an independent economy (`Econ[k]`) with `Ex1()`, and calculates the Walras equilibrium prices (`Walras_prices`) and allocations (`Walras_alloc`) with `Walraseqrtr()`. It measures the runtime of the bilateral algorithm (`Bilateral`) with specific parameters: `eps_prices=1e-6` (price convergence tolerance), `MAXIT=250000` (maximum number of iterations), `lbda=0.975` (premium reduction factor), `delta_tol=1e-18` (minimum premium tolerance), and `inspection='ran`` (random selection of agents and goods). Stores results in `EvPag[k]` (price history), `EvAlloc[k]` (allocations), `EvDelta[k]` (premiums), `TimesChange[k]` (number of transactions), `K[k]` (total iterations), and `eq_status[k]` (1 if converged, 0 otherwise). Computes the running time (`ExTimes[k]`), stores equilibrium prices (`Evpbar`) by taking the maximum per good, and computes the maximum standard deviation of prices across agents (`EvSD[k]`) to evaluate convergence.

**Lines 79 to 102**
```python
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
```
This section of the code generates plots and calculates metrics to compare bilateral (BTE) and Walras equilibria. It creates a `Z` matrix (agents × iterations) to store each agent's utility at each iteration of the bilateral algorithm, calculated with the Cobb-Douglas function (`np.prod(EvAlloc[k][kk][ii,:]**Econ[k].alpha[ii,:])`). It plots the evolution of these utilities per agent with `plt.plot`, labeling axes (iterations in x, utility in y), adding a legend, and saving the plot as `Ut_trial{k}.pdf` before closing it with `plt.close()`. It then calculates comparative metrics: `BPT[k,:]` stores the maximum prices per good (`np.max(Econ[k].pag)`), reflecting price convergence (section 3.5). `Wealth_bte[k,i]` computes each agent's wealth as the product of equilibrium prices (`Evpbar[:,k]`) and final allocations (`Econ[k].allocations[i,:]`) (section 3.6). `Utilities_bte[k,i]` computes the Cobb-Douglas utility of bilateral allocations (section 5, equation 5.2). Similarly, `Wealth_wal[k,i]` and `Utilities_wal[k,i]` compute wealth and utility with Walras prices and allocations (`Walras_prices`, `Walras_alloc`). Finally, `gc.collect()` frees memory. This section is key to visualize and compare the evolution of utilities and outcomes between both methods.

**Lines 104 to 161**
```python
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

plt.figure()
x = np.linspace(np.amin([Utilities_wal,Utilities_bte]),np.amax([Utilities_wal,Utilities_bte]),100)
for k in range(ntrials):
    plt.scatter(Utilities_bte[k,:],Utilities_wal[k,:],label='Trial {}'.format(k))
plt.plot(x,x,linestyle='--')
plt.xlabel('BTE Utility')
plt.ylabel('Walras Utility')
plt.legend(loc='lower right')
plt.savefig('Utilities.'+exfig)
```
This section of code generates statistics and visualizations to validate the paper's results. It prints the number of simulations that reached equilibrium (`np.sum(eq_status)` of `ntrials`), the median iterations (`np.median(K)`), and the running time (`np.median(ExTimes)`). It creates a line chart with `plt.plot` to display equilibrium prices (`Evpbar`) per good (excluding money), with labeled axes (goods in x, normalized prices in y) and a title in LaTeX notation, saving it as `Pbar.pdf`. It then generates a scatterplot comparing BTE (`Utilities_bte`) and Walras (`Utilities_wal`) utilities for each agent in each simulation, using `np.linspace` to define a 100-point range between the minimum and maximum utilities. Plot an `x=y` line to evaluate the efficiency of BTE versus the Walras optimum, where points above indicate greater welfare in BTE and points below indicate loss. The axes are labeled BTE (x) and Walras (y) utilities, with a legend identifying the 10 simulations. The graph is saved as `Utilities.pdf`. This section evaluates convergence and compares the efficiency of bilateral and Walras equilibria.

**Lines 163 to 171**
```python
plt.figure()
x = np.linspace(np.amin([Wealth_wal,Wealth_bte]),np.amax([Wealth_wal,Wealth_bte]),100)
for k in range(ntrials):
    plt.scatter(Wealth_bte[k,:],Wealth_wal[k,:],label='Trial {}'.format(k))
plt.plot(x,x,linestyle='--')
plt.xlabel('BTE Wealth')
plt.ylabel('Walras Wealth')
plt.legend(loc='lower right')
plt.savefig('Wealth.'+exfig)
```
This section of the code generates a scatter plot to compare agents' wealth in bilateral equilibrium (BTE) and Walras equilibrium. It creates a figure with `plt.figure()` and defines 100 equidistant points (`x`) with `np.linspace`, from the minimum (`np.amin`) to the maximum (`np.amax`) of the BTE (`Wealth_bte`) and Walras (`Wealth_wal`) wealth. It iterates over `ntrials` simulations with a `for` loop, plotting a scatter plot (`plt.scatter`) of `Wealth_bte[k,:]` (BTE wealth) against `Wealth_wal[k,:]` (Walras wealth) for each agent in simulation `k`. Plot a dashed `x=y` line (`linestyle='--'`) to evaluate efficiency: points above the line indicate higher BTE wealth, while points below indicate lower wealth. Label the axes with `plt.xlabel` (BTE wealth) and `plt.ylabel` (Walras wealth), add a legend with `plt.legend` to identify the 10 simulations, and save the plot as `Wealth.pdf` with `plt.savefig`. This visualization evaluates the efficiency of the bilateral equilibrium versus the Walras optimum in terms of wealth.

**Lines 174 to 196**
```python
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
```
This section of the code calculates and visualizes inequality in wealth distribution between bilateral (BTE) and Walrasian equilibria using the Gini coefficient, and generates a boxplot of the equilibrium prices. It initializes `Gini_bte` and `Gini_wal` arrays to store the Gini coefficients for each simulation (`ntrials=10`), calculating them with the `gini()` function on the wealth vectors `Wealth_bte[k,:]` and `Wealth_wal[k,:]` to measure inequality (0=perfect equality, 1=maximum inequality). It creates a scatter plot with `plt.scatter` comparing `Gini_bte` (x-axis) and `Gini_wal` (y-axis) for each simulation, using `np.linspace` for 100 points between the minimum and maximum of the coefficients. Plot a dashed `x=y` line to assess whether BTE reduces (points below) or increases (points above) inequality relative to Walras. Label the axes, add a legend for the 10 simulations, and save the graph as `Gini.pdf`. Then generate a boxplot with `plt.boxplot` for equilibrium prices (`BPT`, 10 simulations × 9 non-monetary goods), excluding outliers (`showfliers=False`), showing the dispersion of threshold prices by good (tight boxes indicate agreement, wide boxes reflect sensitivity). Title the graph in LaTeX, label the axes (goods in x, normalized prices in y), and save the figure as `Page.pdf`, closing it with `plt.close` to free memory.

**Lines 198 to 212**
```python
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
```
This section of the code generates a comparison chart of the equilibrium prices across simulations and analyzes the largest differences between them. It creates a figure with `plt.figure()` and calculates `pmean`, a 10-element vector that averages the equilibrium prices (`Evpbar`) per good across the 10 simulations (`ntrials`). It constructs a lower triangular `diff` (10x10) matrix with `np.zeros` to store the maximum differences between the equilibrium prices of each pair of simulations, using two `for` loops (over `i` and `j<i`) to avoid redundancies. It calculates `diff[i,j]` as the maximum of the absolute differences (`np.abs(Evpbar[:,i]-Evpbar[:,j])`) between the prices of simulations `i` and `j`. Identify the simulations with the largest difference (`i0`, `i1`) using `np.unravel_index(np.argmax(diff), diff.shape)`. Plot the equilibrium prices from these simulations (`Evpbar[:,i0]` and `Evpbar[:,i1]`) with `plt.plot`, labeling each with its simulation number. Print the `diff` matrix to show the differences, then unravel the rows and columns of `i0` and `i1` in `diff` to avoid reuse, allowing new combinations to be identified.

**Lines 213 to 229**
```python
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
```
This section of the code generates a graph and table comparing the simulations with the largest differences in equilibrium prices and the Walras prices. It identifies the second most divergent pair of simulations (`i2`, `i3`) using `np.unravel_index(np.argmax(diff), diff.shape)` over the `diff` difference matrix. It plots the equilibrium prices from these simulations (`Evpbar[:,i2]`, `Evpbar[:,i3]`) with `plt.plot`, assigning legend labels to each, placing the legend in the lower right corner, and saving the plot as `Pbardiffs.pdf` before closing it with `plt.close`. Prints the indices `i0`, `i1`, `i2`, `i3` of the selected simulations and generates a table in LaTeX format with a `for` loop over the goods (`Ec_aux.n`), showing the equilibrium prices of the four most divergent simulations (`Evpbar[n,i0]`, etc.) next to the Walras price (`p_we[n]`) for each good `n`. Create a `PShow` array with `np.column_stack` combining the prices from simulations `i0`, `i1`, `i2`, `i3` and `p_we`, and use `pd.DataFrame` to generate a LaTeX table (`Ex1_results.tex`) with good labels (`idags`, in LaTeX format `$j=0$` to `$j=9$`) and columns labeled `$\overline p^{i}$` for the simulations and `$\overline p^W$` for Walras, showing prices to four decimal places.

**Lines 232 to 240**
```python
p_we, x_wertr = Ec_aux.Walraseqrtr()
p_wejd, x_we = Ec_aux.Walraseq()

idxs = np.array([i0,i1,i2,i3])
print('j&{}&{}&{}&{}&Wal'.format(i0,i1,i2,i3))
for jj in range(4):
    ids = idxs[jj]
    print('{}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}'.format(ids,Econ[ids].allocations[0,0],Econ[ids].allocations[1,0],Econ[ids].allocations[2,0],Econ[ids].allocations[0,1],Econ[ids].allocations[1,1],Econ[ids].allocations[2,1],Econ[ids].allocations[0,2],Econ[ids].allocations[1,2],Econ[ids].allocations[2,2]))
print('Wal&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}&{:.2f}'.format(x_we[0,0],x_we[1,0],x_we[2,0],x_we[0,1],x_we[1,1],x_we[2,1],x_we[0,2],x_we[1,2],x_we[2,2]))
```
Finally, two variables p_we and x_wertr are created, which represent the equilibrium price vectors and the transformed goods allocation matrix respectively. All this data is used to calculate the Walrasian equilibrium with square root transformation using Walraseqrtr(). Then, another method is used, the normal calculation of the Walrasian equilibrium using Walraseq() and with the variables p_wejd which are the prices and x_we which are the final allocations of goods (creating these last two variables as before). Then, a variable inxs is created that packs the indices of the 4 most divergent simulations, which facilitates iterative access to these special cases. Then, a print is created with the same latex notation as above with the same format. Then, a for loop is created that iterates over the 4 selected simulations, that's why in range(4). Then, this for loop contains the variable ids which obtains the numeric ID of the current simulation. A LaTeX table is created with all this data in the same table format, including the allocations of the agents who had the greatest deviations from their equilibrium prices, including money as a good, as it is the only one that did not change since it is normalized to 1. All this is done with Econ[ids].allocations[i,j] where i is the agent and j is the good. This shows the agents who were most disparate in their equilibrium prices for different goods. The same table is then modeled after Walrasian equilibrium simulations to see which one showed the greatest variation. The same table was also created in LaTeX format.


#### Installing libraries and configurations to run the code
- A. First you have to open the Anaconda prompt, searching for it from the search bar of the PC. The Anaconda prompt can be found by going to the search option in the Windows taskbar. 

<img width="959" alt="Image" src="https://github.com/user-attachments/assets/51afe7d7-aa57-4f14-940d-97a3f79f322b" />

- B. Then search for "Anaconda Prompt" in the taskbar search box and click

<img width="702" alt="Image" src="https://github.com/user-attachments/assets/65d2c4cc-37aa-4c81-9303-1a7eaca9c466" />

- C. The following tab will open once you click on the anaconda prompt

<img width="905" alt="Image" src="https://github.com/user-attachments/assets/8f1d603d-6558-45cb-89bd-b43fac80bef1" />

- D. Once in the Anaconda prompt, follow these steps to install the libraries to run the code in Jupyter.
    - D.1. First, you need to create a Python 3.11 environment with the following command:

      <img width="884" alt="Image" src="https://github.com/user-attachments/assets/f83a21a7-3d04-441e-b7b2-c3553df49fb6" />
       
      This is because an error will appear regarding the imp module, which consists of the fact that this module was marked        as obsolete for Python 3.4 and was already removed in Python 3.12, so Python 3.11 is the most modern library that            contains the imp module for the pyutilib library to work.

    - D.2. After this, the created environment is activated with the following command:
 
      <img width="894" alt="Image" src="https://github.com/user-attachments/assets/6328dfc9-021a-4ea0-9666-7f5bae1c51c2" />

    - D.3. Once inside the py311 environment, the following channel is installed, which has a larger collection of software:
 
      <img width="902" alt="Image" src="https://github.com/user-attachments/assets/d9131fbc-e108-47f5-83a3-ecd07ce79366" />

    - D.4. Once Forge is installed, the libraries are installed as follows:
 
      <img width="912" alt="Image" src="https://github.com/user-attachments/assets/ee4e3b96-16eb-4874-a7a9-9769bef3864a" />

      Pyomo is used to formulate and model optimization models used in the BTE.py code, specifically the verification and calculation of Walrasian equilibria in the context of the centralized economic model that this traditional equilibrium has. These optimization models are designed to calculate Cobb-Douglas utility maximization subject to a budget constraint and its corresponding price equilibrium. As the paper states, Walras solves linear equation problems derived from first-order conditions for economies with Cobb-Douglas utilities. Pyomo has an extension called ipopt that is used to solve non-linear optimization problems such as the Walrascheck method, which maximizes Cobb-Douglas utility by producing the sub ij allocations of the agents raised to their preferences, distributed according to each agent and good. This utility maximization, which is the objective function, has a budget restriction consisting of the sum of the prices of each good per agent multiplied by the difference between the sub ij allocations of the agents           minus their sub ij initial endowments, which must be equal to or less than 0 to ensure that each agent starts with a         certain amount of money and initial allocations so that they can make the trades. Ipopt helps calculate this convex          non-linear optimization problem. After this, it verifies that the excess demand is close to 0 for all goods j, which         indicates a Walrasian equilibrium. Ipopt is invoked from pyomo and is used to find local maxima and find the x sub ij        that allow maximizing utilities, allocations and wealth through the prices p of each good. The pyutilib library is           used to measure the time taken to reach equilibrium using the bilateral market exchange method in each simulation. It        also complements and is efficient with the pyomo library.

    - D.5. Verifying package installation:

      <img width="906" alt="Image" src="https://github.com/user-attachments/assets/f3843785-0eb5-4f2b-bdb8-57a9ba76d244" />

- E. Once the library installations have been verified, proceed to install the newly created environment in the Jupyter kernel.
    - E.1. Install ipykernel first:

      <img width="894" alt="Image" src="https://github.com/user-attachments/assets/f8558af0-0650-4a1a-b587-4f9a5f1a55b0" />

    - E.2. Register the environment in the Jupyter kernel after installing ipykernel as follows:

      <img width="898" alt="Image" src="https://github.com/user-attachments/assets/7d2c23d8-313d-485d-8662-3a851904fb2b" />

- F. As a final step, install MikTeX from the following website <https://miktex.org/download>

<img width="960" alt="Image" src="https://github.com/user-attachments/assets/8c407a5f-17c2-4d6c-98c6-2237e8037dd5" />

- G. Follow the steps below for the executor:
    - G.1. Accept the terms and conditions and press next:

      <img width="409" alt="Image" src="https://github.com/user-attachments/assets/72c52e4c-3188-424a-bf2b-f8072e18bf11" />

    - G.2. This section is optional (whether you want to download it for all users or just for yourself). Click Next:
 
      <img width="401" alt="Image" src="https://github.com/user-attachments/assets/2a2a46b9-513e-4637-8b83-00c0a364af89" />

    - G.3. The path where the mikTeX installation files will be saved is shown here. Click Next:
 
      <img width="410" alt="Image" src="https://github.com/user-attachments/assets/2de067a5-e502-49de-a390-ffc19dbf7e63" />

    - G.4. Select Yes for the second option to automatically install any missing packages the code may require. Click Next:
 
      <img width="415" alt="Image" src="https://github.com/user-attachments/assets/3362558a-6815-4487-83de-8958861b78fe" />

    - G.5. Start the installation:
 
      <img width="393" alt="Image" src="https://github.com/user-attachments/assets/c297336f-84ba-4479-ba49-37d742367390" />
  
  Once these steps are completed, the first part of installing libraries and configurations to run example 1 is complete. Now we will show how to place the example in Jupyter and how to select the environment created previously in the kernel.

- A. Open Anaconda by searching for it in the taskbar search:

<img width="274" alt="Image" src="https://github.com/user-attachments/assets/7fc2d247-6f5b-49f0-83fb-c51cc8be4ef6" />

- B. Open Jupyter from Anaconda once inside, by pressing "launch":

<img width="959" alt="Image" src="https://github.com/user-attachments/assets/16a1cbbc-e8b7-4c98-b0ce-b84cfbb3b872" />

- C. Download BTE and Final1 from the repository in the folder that says Ex1:

<img width="873" alt="Image" src="https://github.com/user-attachments/assets/beff71c9-23cb-4850-8547-6fef2ae5fafc" />

-D. Press BTE once inside:

<img width="960" alt="Image" src="https://github.com/user-attachments/assets/1563d5a2-2260-4b95-853a-a96d7f27c747" />

-E. Download BTE:

<img width="960" alt="Image" src="https://github.com/user-attachments/assets/8326b172-dc44-4219-8f46-145cc46ac927" />

Same process to download Final1.

-F. Once saved in your computer's files, return to Jupyter. Press "Upload" to upload these two .py files:

<img width="960" alt="Image" src="https://github.com/user-attachments/assets/27097dbf-d049-4365-82c8-980c272bf3d1" />

-G. Now the two downloaded files are chosen (BTE and final1):

<img width="514" alt="Image" src="https://github.com/user-attachments/assets/6071ec80-72f2-4947-85ab-46143b66e72f" />

-H. Create a new notebook with the environment created in the New option:

<img width="960" alt="Image" src="https://github.com/user-attachments/assets/22cd77fb-beeb-45bd-9b03-3b9233095142" />

-I. Open Final1.py and copy its entire contents by selecting everything with Ctrl+A and Ctrl+V to copy. Copy it to the new notebook created with the Python 3.11 environment:

Once all these steps have been completed, the code is ready to run and display all its outputs. The next section will show all its outputs and their explanations. It will confirm in detail that trades in bilateral exchanges are better than in Walras exchanges.

#### Outputs

The first output that will be launched are the times that it took to calculate the bilateral equilibrium for the different trials calculating the delta premiums and the optimal allocations of each of the 5 agents at each point for the 10 goods. After this, it launches values ​​of the standard deviation between the prices of each agent for each calculated trial and with their tolerance of the delta premium. After the 10 trials have passed, it can be observed that the console will say that equilibrium has been reached in 10 of the 10 simulations with their respective, with the average of the times that the bilateral algorithm took to find equilibrium and also places the number of iterations that the algorithm took to find an equilibrium in each of the 10 simulations as shown below:

![Image](https://github.com/user-attachments/assets/3f1c5a4f-f0c2-4107-b11b-5e7adcb21b81)


  The times are measured in Final (which are measured with the ExTimes command in Final1) to evaluate the computational efficiency of the BTE algorithm compared to the Walrasian approach. This is important because it demonstrates that it not only converges theoretically, but also does so in a practical time, even for economies with many agents and goods. The time reflects the algorithm's scalability. Scalability refers to the algorithm's ability to maintain its efficiency (in terms of execution time and resource usage) as the problem grows in size and complexity. The paper shows that BTE reaches equilibrium for 5 agents and 9 goods in 2.12 seconds (median), and the problem for 10 agents and 100 goods converges in 711 seconds, which is approximately 12 minutes. While the time increases with the size of the problem, it does so in a manageable and predictable way. The growth in time is also not exponential, making it impractical for large economies. Time grows sustainably, allowing for the study of the effect of more agents and goods participating in the market, with realistic interactions between them. In contrast, the Walrasian system requires solving systems of linear equations (such as Walraseq() in Final1), which can become computationally expensive for many goods and agents. The BTE, on the other hand, avoids this and, being an iterative and decentralized process, avoids the need to solve global systems with linear equations, making it more scalable in practice without expending significant resources. The code was run 10 times and values ​​between 2 seconds and 5 seconds of delay were obtained, which is what is expected or within a range of values ​​indicated in the paper, which makes sense for small economies. After generating the execution times and the standard deviation that the example takes to reach equilibrium, the code launches an output of the number of times the equilibrium was reached and the median of the number of iterations that the algorithm took to reach equilibrium in each of the tests. The paper says that the code has a median of 3093 iterations to reach equilibrium in each of the tests. Measurements from running the code 10 times gave a range between 3050 and 3100 iterations as the median for each of the tests, which supports what the paper says, as well as the efficiency of the BTE algorithm and how quickly each of these iterations are done, taking between 2 to 5 seconds of execution times. It also displays the average time it took the algorithm to reach equilibrium in each of the tests, which in most of the tests performed gave a value between 2 and 5, as previously stated. 


Then the output that the code launches are matrices of 10 x 9, where the rows represent the iterations that have passed in the algorithm and the columns each of the 9 goods (Money is normalized). The content of the matrix represents how the prices of each of the agents change as the iterations pass to reach equilibrium when making bilateral market exchanges. It can be observed that as the iterations pass, all the agents reach agreements and their prices change to reach the equilibrium prices of each good. It can be observed that in the BTE algorithm, in each of the iterations, the agents arrange to reach an agreement and balance their holdings to reach an agreed price for those goods. The code was run and it showed that in the first iteration no agents yet make trades, the price deltas are 0. Then for iteration 2, it can be observed that good 1 has a change, having a value of 0.01457994, meaning that its value has a decrease or increase of approximately 1.45%, subject to the change in utility with respect to the change in its allocations, occupying equation 2.8 of the paper. This occurs for the prices of all goods, demonstrating that agents always look for an opportunity to trade randomly without restrictions to seek equilibrium. After this, there is another 10x9 matrix that represents the change in prices, in this case, a Walras matrix, where it can be seen that prices do not always change between agents in the different iterations. This means that it is a very restrictive algorithm subject to linear equations, forcing agents to trade in an orderly manner, without any type of randomness, which is more artificial and less realistic from a market point of view. It can be observed in the generated matrix that a price change occurs in iterations 2 and 3, but in iteration 4, no price changes occur, seeing that the agents under their restrictions did not find any trading opportunities. There were even goods that did not suffer price changes in any of the iterations, seeing that there were no trade opportunities for that good for any of the agents. 

[Imagen]

After this, a table is created where 4 tests of the algorithm (columns) are chosen for the 9 goods (rows) where the prices varied for each good are compared to make the comparison with the Walras prices, which are fixed without iterations. A price variation can be observed for good 1 between 0.9093905048137538 and 0.9473978286534999, coinciding with table 2 of the paper on page 21 of this, as well as the other goods. With respect to the Walras price, a similarity can be seen, indicating that the best way to reach equilibrium is with BTE, seeing that this value can be reached by the different ways of being able to trade between the different agents and goods and not just in one way, as is the restrictive Walrasian model. The BTE algorithm is highly efficient when searching for equilibrium prices, seeing how by establishing a maximum tolerance, prices remain stable in the different trades between agents with the different goods, obeying this slight restriction so as not to create discriminatory prices while maintaining utility between agents, seeking to improve their utility in each trade, quite the opposite of Walras, which is an established algorithm and does not allow free trading. 

[Imagen]

The rows and columns are then inverted, and the change in the allocations of each good for each of the four iterations randomly chosen by the code is observed, comparing the BTE allocations with the Walras allocations. It can be observed that the BTE allocations are very close to those of Walras and vary between very close values. In the example, the BTE allocations for good 1 for the four chosen iterations vary by a maximum of approximately 7 units, between the values ​​58 and 51. This good has a Walras value of approximately 51. It can be seen how agents in the BTE algorithm can correct their allocations to improve their positions jointly rather than individually, reaching a value greater than the Walras allocations. This also speaks to the dynamism of the algorithm in seeking the best allocations for everyone, better than those of Walras, which is an algorithm that makes trades with restrictions and in a single test.

[Imagen]

Now comes the section where you can see the graphics provided by this code.

First, the code outputs the graph of Walras utilities vs. BTE, placing each algorithm on the y and x axes respectively. Then, an x=y function is plotted to observe which of the two algorithms had the highest utility (Points above the function, the Walras utility is better. Points below the function, the BTE utility is better). It can be observed how all the points for each of the iterations show that the Walras utility is better, all being on the dotted function line. This is not necessarily optimal; this means that the Walras utilities for each of the agents are completely unequal. This algorithm, having restrictions, does not allow its agents to want to improve their utilities jointly such that all have a similar utility. It can be seen in the 5 points plotted in the graph that there is a large difference between the agents' utilities for Walras. BTE, on the other hand, has lower utilities but manages to compensate by making everyone's utilities similar, with no inequality between agents.

[Imagen]

The code then generates a wealth graph, which follows the same logic as the first graph. Here, we can conclude the same thing as before. BTE's wealth is similar across agents and worse than in Walras, but Walras has very unequal wealth across its agents.

[Imagen]

A graph of the BTE vs. Walras Gini index is then drawn, indicating the inequality between each method, using the same logic as the other two graphs. As expected, the Walras Gini is greater than the BTE and remains at 0.4, as can be seen in the graph. The BTE Gini grows steadily with each iteration, with no more than 0.10 points on the Gini index, ensuring equality between agents.

[Imagen]

Ten graphs are then generated showing how utilities have evolved across the different trials of a test for each agent. The concave behavior of the curves with positive slope can be observed, as stated in the paper, reaching a limit where agents can no longer improve their utilities. The utility function is a Cobb-Douglas function that is convex, and the paper states on page 6 that when the preference equation is convex, it will always have a concave subset, as demonstrated by the graph. It can be seen that there is a relatively balanced utility among each of the agents in each of the iterations. The utilities are plotted on the y-axis, and the iterations of each test are plotted on the x-axis. This shows roughly at which iteration the utility equilibrium was reached.

[Imagen]

A graph called Page is then generated, which shows how the prices of each good evolved, where each good is placed on the x-axis and the prices are placed on the y-axis. Each good is represented in a boxplot. Boxplots are used to measure dispersion in a data set, showing the median and quartiles, which are data positioning measures, to know exactly where each of them is located. These graphs also consider atypical data, which are those that are far from the others. In this graph, atypical data are excluded. It can be observed in each boxplot that the dispersion is low due to their size; the interquartile range (the difference between the third and first quartiles) is low, presenting a low dispersion of the central data with a median that is approximately in the center of the box. This shows the reliability of the model and that agents are reaching agreements to arrive at a specific price.

[Imagen]

A graph called Pbar is then generated, which is the union of the medians of each boxplot from the previous graph for each iteration. The axes of the previous graph remain unchanged, but the boxplots are removed, demonstrating that in each iteration, prices remain virtually unchanged or the change is negligible.

[Imagen]

Finally, a graph called Pbardiffs is created. It randomly selects four iterations from the previous graph to clearly show the evolution without the other iterations interfering with the measurement. The logic is the same as Pbar.

[Imagen]

### Example 2
#### Explanation
Example 2 from the paper "Reaching an Equilibrium of Prices and Holdings of Goods Through Direct Buying and Selling"  analyzes a small economy with 3 goods (including money, \( j=0 \)) and 3 agents, using Cobb-Douglas utility functions but with highly unbalanced initial endowments to test the bilateral trade algorithm (BTS). Each agent has preferences defined by exponents \(\beta_{ij}\) (see Table 3, p. 22) and initial endowments (\(x_{i j}^0\)) where each non-monetary good is mostly concentrated in one agent (e.g., agent 2 has 80 of good 2, agent 3 has 80 of good 1).

The BTS algorithm, described in Section 5, is run with a random inspection strategy to identify bilateral trade opportunities. The results of two runs (\(\nu=1, \nu=2\)) are reported in Tables 4 and 5 (p. 22). Equilibrium prices (\(\bar{p}^\nu\)) and final allocations (\(\bar{x}^\nu\)) are consistent across runs, with close prices for nonmonetary goods (e.g., \(\bar{p}_1^1 \approx 0.0475\), \(\bar{p}_1^2 \approx 0.0460\)). However, they differ significantly from the Walras equilibrium prices and allocations (\(\bar{p}^W\), \(\bar{x}^W\)), which show higher values ​​(e.g., \(\bar{p}_1^W \approx 0.4921\)).

The example highlights that, despite the initial imbalance, bilateral trade converges to a price and allocation equilibrium, but not to the Walrasian equilibrium, reinforcing the paper's criticism of the Walrasian model's lack of market legitimacy, as it relies on a centralized mechanism rather than direct buying and selling interactions.
#### Code documentation
Here the documentation for example 2 will be placed to know what each part of the code does with its respective commands. we started.
BTE is the same for both example 1 and 2 so we start immediately with the Final2 documentation.
#### Final2 documentation
**Lines 1 to 37**
```python
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
```
The following libraries are imported:
- From BTE import: Imports the economy class and auxiliary functions from BTE.py such as bilateral trading.
- From copy import deepcopy: To copy objects without sharing references.
- Import pandas as pd: To export data to Excel.
- Import gc: Garbage Collector (optimizes memory).
- From time: Measures time and handles date formats.
- Import os: System operations, for example, creating folders.
- Import matplotlib: To generate graphs.
- From mpl_toolkits: Support for 3D graphics (not used)
Styles are configured for graphs:
- rc(“font”...): Serves as a font for graphs, controlling typographical aspects, setting global parameters, and setting the preferred font for graph letters.
- rc(“text”:::): Enables text rendering using LaTeX, allowing the use of mathematical commands and professional typography.
- plt.rc(“text”..): Reinforces the previous configuration using the pyplot interface. It is redundant but ensures that LaTeX works.
- plt.rc(“font”..): Configures the serif font family for mathematical elements such as Times New Roman.
We begin by defining example 2 incorporated in the paper, starting with the number of initial goods, which in this case are 3 (j=0, money, and j=1 to 2, goods other than money). We then insert a 3x3 matrix of the alpha parameters, which are the agents' preferences from the Cobb-Douglas utility functions. The agents' initial endowments are then inserted into a 3x3 matrix using the function e. The Econ class is then created, where the economy of n agents and 1 goods, including money, is created. Econ.alpha assigns the agents' preferences. Econ.e assigns the agents' initial endowments. Econ.allocations allocates the agent's assets after iterating the initial allocations for the first time, and subsequent iterations as well. Econ.pag returns the result of each agent's prices after the iterations. Delta updates the delta values ​​when trades are no longer possible, decreasing their value by 90% until it is zero or negligible. Econ.delta allocates the deltas according to the current iteration. The result for the economy in question is then returned. np.random.seed is then created to generate random results between pairs of agents, ensuring they are not in order.

**Lines 39 to 78**
```python
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
```
exfig for formatting figures. A trial economy is created using Ec_aux. This trial economy is used to calculate the theoretical Walrasian equilibrium by entering p_we and x_we. It is also necessary to save each economy for each trial, so the variable Econ is created. Next, it is necessary to know the number of simulations or trials by entering ntrials (which in this case will be 20). It is also necessary to save the allocations, prices, and deltas for each simulation, so the following variables are created: EvPag, EvAlloc, and EvDelta. The number of trades for each iteration is then saved in the variable TimesChange. Next, a variable K is created that records the total iteration time for each simulation. An equilibrium condition is then created with q_status, where 1 indicates whether the good converges to equilibrium and 0 indicates whether it does not. As discussed above, each good, including money, is equivalent to 1. The execution times must then be calculated, measured using the variable ExTimes. The maximum standard deviation for prices is then set using EvS. Equilibrium prices must also be evaluated through simulation, which is done using Evpbar. Finally, the agents' Walrasian prices and current Walrasian price allocations are evaluated using Walras_prices and Walras_alloc. The equilibrium price information for all goods except money is then stored using the BPT function, generating a 9x10 matrix where the rows represent the simulations and the columns the quantity of non-monetary goods per agent. These boxplots show the dispersion of threshold prices. Wealth_bte calculates agent wealth for each agent i in simulation k to generate a comparison with the Walras method. Utilities_bte is used to calculate the utility of the Cobb-Douglas functions for each agent and compare it with the utilities of the Walras equilibrium. Wealth_wal is used to measure the wealth generated by the Walras equilibrium using Walraseqrtr() prices. Finally, Utilities_wal calculates utility for the Walras equilibrium with its respective allocations. Next, a variable prices_final is created, which initializes the matrix containing the final equilibrium prices with the number of iterations and Ec_aux. An empty dictionary is then created to store the inspection orders for each simulation using insp=. Permutations of fixed inspection orders between goods and agents are then created, creating inspection vectors for each agent, where the first order inspects from agent 0 to agent 2, creating different inspections that will lead to different evaluation results. Then with a g_order= the normal and reverse order of the goods is made. After this, the initial matrices are created with np.ones of the initial wealth and utilities, excuse the redundancy given by the variables w0 and u0 respectively, each one with a size of 3 agents. Then a for loop is created that iterates over the 3 agents of the economy Ec_aux created initially as well as the matrices created previously precisely to calculate the initial utilities and wealth of the agents. After this, the variable p0 is created which is the price vector of each of the agents that with np.append it is possible to modify the initial matrix of the agents, creating a single vector with the initial prices of each of the agents, where the first good in the matrix will always be 1, since it is money and is normalized by norm and with Ex_aux.pag[i, :] what remains of the vector is completed with the personal prices of agent i. Then with a print(p0) the personal prices of agent i are printed. Then the variable w0[i] is created where the initial wealth of the agents is calculated using the dot product with p0 being the initial prices of the agents and their initial endowments, with Ec_aux.e[i, :] being the initial endowments of agent i, calculating the dot product between these two matrices using np.dot. Then the variable u0[i] is created which is to calculate the value of the initial utility of the agents, using np.prod which calculates the product of all the elements of a matrix, in this case Ec_aux.allocations[i, :] which are the initial endowments of agent i and Ec_aux.alpha[i,:] are the preferences of the agents, raising allocations in alpha to calculate the utility using the Cobb Douglas equation.

**Lines 80 to 167**
```python
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
```
Then a for loop is created that iterates over each simulation of the 20 defined above, that is, in range(ntrials)(ku is a variable that is created to iterate through each simulation). Then an if is created that if it goes through the first 12 simulations it configures a fixed inspection order combined with 6 possible orders of agents and 2 possible orders of goods as proposed previously. Then two for loops are created, the first for the fixed order of the agents, which goes through each of them (the 6) (creating the index io) and the second for the fixed order of goods (the 2) (creating the index go) then storing all these values ​​that go through the for loops mentioned above in insp[2*io+go]. This last variable saves the fixed combinations of agents and goods, maintaining a fixed order. Then an else is created for simulations 12 to 19 that establishes a random order with the variable insp[ku]. The "ran" object establishes the random order, with empty spaces used as spaceholders or storage for new data. A printout of the complete inspection configuration is then placed. A for loop then iterates over all the simulations being run, allowing the consistency of the comparison results to be studied. A printout is then placed to determine which part of the simulation evaluation is in order to track the process. The Eco variable then creates an independent copy of the initial economy for each test, which is crucial as it allows each agent to start from the same initial state based on their holdings, and the negotiation processes alter the states of the economy. Walras_prices and Walras_alloc are also created, where the former stores the equilibrium prices at iteration k and the latter stores the equilibrium allocations for all agents, respectively. This will serve as a comparison with the other method. Next, a variable t is created, which measures the current time it takes to execute the bilateral algorithm, which will be measured later. The following lines of code then execute the bilateral algorithm with the following variables:
- eps_prices=1e-4: Threshold for considering prices to have converged (when the standard deviation of prices between agents is less than this value)
- MAXIT=250000: Maximum number of iterations allowed
- lbda=0.998: Price premium reduction factor deltaij
- Delta_tol=1e-18: Minimum value allowed for deltaij
- inspection=insp[k]: Strategy for selecting pairs of agents and goods with the orders established above.
The returned results are:
- EvPag[k]: Price history during the iterative process
- EvAlloc[k]: Allocation history during the iterative process
- EvDelta[k]: History of deltaij premiums during the iterative process
- TimesChange[k]: Number of times transactions were made
- K[k]: Total number of iterations performed
- eq_status[k]: Indicator of whether equilibrium was reached (1) or not (0)
Then, the ExTimes[k] function calculates the time it took to execute the bilateral algorithm for the current test, and then prints the execution time. After this, the final equilibrium prices for the current test are stored as follows:
- Econ[k].pag: Stores the threshold prices for all agents (i.e., the limit when revenues equal costs)
- np.max: Takes the maximum for each good
- Evpbar: Stores all goods starting from position 1.
Then, the maximum standard deviation of prices between agents is calculated for any good:
    - np.std: Calculates the standard deviation per good.
    - np.max: Takes the maximum of these deviations.
    - Measures how close agents are to agreeing on prices (convergence).
      
Then, a for loop is created that iterates over each agent i in economy k with Econ[k].I. Then a matrix z of size number of agents x the number of iterations k is initialized, first creating the variable Z and initializing the matrix with np. zeros and Econ [k]. I and K [k], where there are I agents x K [k] iterations. Then another for loop is created within the previous one, but iterating over each iteration kk of simulation k, which is why the index kk is created. Within this loop, another for loop is created that iterates over each agent ii, creating this index ii, which iterates over each of the agents of economy K. Then a variable Z [ii, kk] is created where the value of money (good 0) is assigned for agent ii in iteration kk, therefore placing EvAlloc of economy k iterated over kk and ii, being 0. This implies the direct monetary component as mentioned above. Then another for loop is created within the previous ones, creating the index nn to iterate over the non-monetary goods by iterating over the economy of K with Econ[k] up to the value n-1. After this, the product quantity of the good x the agent's personalized price is added to the monetary possessions that the agent had at the beginning. For this, a variable Z[ii, kk] is created again that multiplies the agents' allocations by the price of these goods to measure their wealth and add it to the money they initially have in their positions. This is done with EvAlloc[k][kk][ii, nn+1], which is the quantity of good nn+1 of agent ii in iteration kk and with EvPag[k][kk][ii, nn] which is the personalized price (threshold) of agent ii for good nn. After this, agent i has the total value of their basket in a certain iteration kk. Then the variable BPT[k, :] is created, which assigns the maximum equilibrium prices for each good in simulation k, calculating the maximum per column for each good with the np.max function with the matrix Econ[k].pag which are the threshold prices of all agents in simulation k. All this results in a vector of the maximum threshold prices of the agents per good in simulation k. All this results in a final equilibrium of maximum price thresholds for each good in the simulations, especially when there are no more mutually beneficial exchange opportunities. Then a for loop is created that runs through the entire economy Econ[k] with all agents I, which creates an index i that places the current agent being iterated. After this for loop contains Wealth_bte which calculates wealth as the dot product between equilibrium prices and final allocations, where Evpbar[:,k] is a vector of equilibrium prices for all goods in simulation k and Econ[k].allocations[i,:] which is the vector of final allocations of agent i. This is related to part 3.6 of the paper which implements the same as above. Then the function Utilities_bte is created which calculates the utility of each agent with a Cobb-Douglas function of the allocations, where Econ[k].alpha[i,:] are the preference parameters of agent i (exponent of the Cobb-Douglas function). ** is used to raise powers and finally np.prod for the product of all the elements. This part implements the Cobb-Douglas function implemented in section 5, equation 5.2. Then the Walrasian equilibrium is calculated. The variable Wealth_wal is created, which is similar to the bilateral calculation but with Walrasian prices and allocations. Then the same utility function is created but applied to Walrasian allocations with Utilities_wal. Then, memory is freed with gc.collect. Then Walrasian reference values ​​are assigned for comparison. For this, the variable Wealth_bte[-1, :] is created, which assigns the Walrasian wealth as the last point of the BTE series, equating Wealth_bte to Wealth_wal[0, :] where Wealth_bte[-1, :] is the last row of the BTE wealth matrix and Wealth_wal[0, :] is the first row of the Walras matrix, which makes it take the Walrasian wealth values ​​as the final reference for the bilateral exchange. Then, the same procedure as before is done, only with the utilities. This allows a direct comparison of the results of the bilateral process with the Walrasian equilibrium. Then, the statistical results and key visualizations that validate the paper's findings are evaluated. First, a print is generated reporting how many tests reached equilibrium (eq_status=1) vs. those that did not, showing a percentage of those that converged. Then, the number of iterations for each simulation called K is printed, thus also calculating the median, which is the central value. Then there is another print to print the amount of execution time that each iteration took per test. Then the aggregate metrics are calculated. First, the variable soc_ut_bte is created, which sums the BTE utilities for each simulation (excluding the last Walras point). This is done using np.sum with Utilities_bte[:-1,:] -1 to not consider the last Walrasian row. Then the Walrasian utility is summed with the variable soc_ut_wal, which only sums one point per column, that is, the last row, which is the one that contains the Walrasian utilities. All this is done again with np.sum and Utilities_wal in parentheses. Then the total BTE wealth per trial is summed with the variable wealth_bte and np.sum again. Then, a figure is created with plt.figure to visualize the evolution of social utility. With plt.plot a time series of the social utility of BTE is created by placing the BTE label, to differentiate with Walras using the label function. Then the same is done, but with the Walrasian social utility. Then with plt.title a title is created for the graph, which in this case is the Social utility, both Walrasian and BTE, in LaTeX format. Then the axis titles are placed with plt.xlabel and plt.ylabel, which are respectively the simulation number and the aggregate utility, all in LaTeX format. Then with plt.savefig the figure is saved with a predefined format with exfig, that is, in PDF format, with the title of SocUt. Then with plt.close, the graph is closed. Another figure is initialized later with plt.figure where the variable labels are generated for the X axis, including the Walras point, putting the number of simulations, that is why range (ntrials + 1), the range generates the sequence of simulation numbers. Then a for loop is created for each agent putting the index i as the agent that is currently iterating the auxiliary economy (Ec_aux.I). Then a stacked bar chart is created using plt.bar, including the variable labels exposed previously which are the different simulations. With Wealth_bte[:,i] being the height of the bars, being the wealth of agent i in each of the trials. The variable bottom positions the bar above the sum of previous agents, thus creating the stacked effect. Then with np.sum, the accumulated wealth of each of the agents is added with Wealth[:,:1] which is column i of the Wealth matrix, which is the wealth of the agents in all simulations. The purpose of this graph is to be able to visualize the evolution of the agents' wealth. After this, a title is placed on the graph with plt.title in latex format with the title of Wealth. A vertical reference line is then added to the graph with plt.axvline, where the variable x sets the position on the x-axis of the simulations, setting the final simulation at -0.25 for visual centering. The line style is then set with linestyle, which in this case is dashes and dots "-." The color is set to black with the color function, which is already black by default. The implicit line thickness is set with linewidth=1.5, setting the line thickness to 1.5. Finally, the graph is saved with plt.savefig with the file name "Wealth_bte" and in PDF format using exfig. The figure is then closed with plt.close() to free up space.

**Lines 170 to 274**
```python

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
```
In the following 10 lines of code, the same procedure is followed, only with the utilities of each agent. Then, a new figure is created with plt.figure. Then, with plt.scatter, a scatter plot is created with X and Y coordinates, respectively, with the threshold prices of goods 1 and 2, by placing Evpbar[1,:] and Evpbar[2,:]. This scatter plot is made to see the dispersion that exists between the bilateral prices for all simulations. Then, the same is done for the Walrasian prices to make the comparison with the threshold prices of the bilateral equilibrium. Then, with plt.title, the title “Final BTE equilibrium price thresholds…” is created for this scatter plot with LaTeX notation. Then, the names of the X and Y axes are placed with plt.xlabel and plt.ylabel, where each one is called “Goods” and “Prices” respectively, which correspond to goods 1 and 2 respectively, excuse the redundancy. Then the figure is saved with plt.savefig with the name of the PDF file Pbar. Then the graph is closed with plt.close to free memory. Then a new figure is created with plt.figure to then create a boxplot with plt.boxplot containing the maximum price matrix using BPT where its rows represent the simulations and its columns the goods. This boxplot creates a boxplot per good to see how the equilibrium prices evolve for each simulation, showing the robustness of bilateral equilibrium vs. the Walrasian. Then the title is given using plt.title and the name of the boxplot axes using plt.xlabel and plt.ylabel. In this case the X and Y axes are Good index and price threshold respectively, each one representing the goods index and the price scale respectively, excuse the redundancy. Then the figure is saved with plt.savefig with the name of the PDF file Pag. Then the graph is closed to avoid losing memory with plt.close. Next, the variables i0 and i1 are created, which are the pairs with the greatest price differences in simulations i and j for the goods. Diff is the aforementioned square matrix of the simulations performed by the different methods. np.argmax(diff) then finds the maximum value of the price differences between the agents' goods across the different simulations as a flat index. To prevent this from happening, np.unravel_index() is used, which converts the flat index into the exact coordinate of the element in question. Diff.shape provides the matrix indices for the conversion. Another figure is created with plt.figure to compare simulations with significant differences in their equilibria. Then a variable pmean is created to make a vector of 3 elements that are all the goods and make an average between the prices of each of these goods in each simulation, where good j are the rows and simulation k are the columns, all this contained in Evpbar leaving a 3x20 matrix with the quantity of goods (including money) and the number of simulations which are 20. This is done to normalize the prices of the different simulations and see what they are converging towards. Then a variable diff is created which is a matrix to compare the difference between the equilibria of each of the models (BTE and Walras). The square matrix is ​​created with np.zeros, where the rows and columns will be the number of simulations that have been carried out using the algorithms, being lower triangular to avoid redundant calculations. After this, two for loops are created, one that goes through all the components of the matrix, that is, through all the simulations and then another for cycle that compares only with the previous simulations to avoid redundancy, which are i and j respectively. After these two loops, the variable diff[i,j] is placed to see the difference in prices between the simulations of each of the algorithms and to see any large discrepancies that may exist. All this is done with np.abs which takes the absolute difference for each simulation of the different algorithms between Evpbar[:, i] and Evpbar[:, j]. Then with np.max it takes the maximum difference between these values ​​and places them in the lower triangular matrix, thus capturing the worst result between the differences between the algorithms. Then the variables i0, i1 are created, which are the pairs where there was a greater price difference in simulations i and j of the goods, where diff is the aforementioned square matrix of the simulations of the different methods. Then np.argmax(diff) finds the maximum value of the price differences of the agents' goods in the different simulations as a flat index. To prevent this from happening, np.unravel_index() is used, which converts the flat index into the exact coordinate where the element in question is located. Diff.shape provides the matrix indices to then perform the conversion. It then results in i0, i1 being the coordinates (row, column) with the greatest difference. Then the graph of a specific simulation is created, in this case the equilibrium prices of the first simulation of each good are placed, forming a vector of size n, corresponding to the total quantity of goods mentioned above (all this is done with Evpbar[:, i0]). Then, with plt.plot, a line is created with the prices per good where the X axis corresponds to the index of the good, and the Y axis to the equilibrium price for that specific good. Then, with label, a legend is created with a label of the simulation in which it is located, in this case the i0. Then, in the next line of code, the same procedure is followed, only with the second simulation and its equilibrium prices, to select the other simulation with the greatest difference, graphing the same axis. Then, a print (diff) is made where the lower triangular matrix that shows the greatest difference between simulations of the different algorithms is displayed, being useful for visualizing the greatest differences. Then, in the following four lines of code, already processed data is eliminated to find new possible combinations. Then, in the following 3 lines of code, the second most divergent pair is graphed, creating the variables i2, i3, which are the pairs where there was a greater price difference in simulations i and j of the goods, where diff is the aforementioned square matrix of the simulations of the different methods. np.argmax(diff) then finds the maximum value of the price differences between the agents' goods across simulations as a flat index. To prevent this from happening, np.unravel_index() is used, which converts the flat index into the exact coordinate of the element in question. Diff.shape provides the matrix indices to then perform the conversion. Then it results that i2,i3 are the coordinates (row,column) with the greatest difference. Then the graph of a specific simulation is created, in this case the equilibrium prices of the first simulation of each good are placed forming a vector of size n, corresponding to the total quantity of goods mentioned above (all this is done with Evpbar[:, i2]). Then with plt.plot a line is created with the prices per good where the X axis corresponds to the index of the good, and the Y axis to the equilibrium price for that specific good. After this, add a legend with plt.legend, placing it in the lower right corner with loc="lower right." Then, save the graph using plt.savefig with four different price curves to show how far they are from equilibrium. A legend identifies each simulation, generating a PDF file called Pbardiffs.pdf with the exfig function. Then, close the graph with plt.close to save memory.

**Lines 277 to 307**
```python
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
```
Then with a print, the indexes i0, i1, i2 and i3 are printed showing the number of the trials selected for the analysis. Then a print is placed to begin the preparation of the table for the comparisons of the deviations of each of the simulations, where the first line is printed j&{}&{}&{}&{}&Wal(& in latex format separates the columns) which is the format of the table of the columns j which in this case are the goods and the Walrasian column and to print the corresponding numbers of the chosen simulations you put format(i0, i1, i2, i3). Then a for loop is created that iterates over all the goods from 0 to n-1 which are all contained in Ec_aux.n that is why it is put within the in range, to iterate over all the goods. Then, we proceed again with a latex-type table like the one mentioned above, only with all the goods that are in each iteration chosen for the analysis with Evpbar, which is the matrix of the equilibrium prices and the number of simulations. For each good n, a row is printed in the latex format described above. The field for each row is n, that is, each good for each simulation that was chosen that is within the columns, and then at the end, the Walrasian price is entered to see how far it is from the theoretical equilibrium. Then, a variable writer is created that creates an Excel file to save multiple sheets with pd.ExcelWriter, calling the file Output.xlsx. Then, a for loop is created that creates an index k that indicates the current simulation that is iterating out of the 20 simulations (hence range(ntrials)). Then a variable XX is created inside the loop, which initializes an array to store K[k] (rows representing the iterations of simulation k) and 11 columns (including data and metadata), all this with np. zeros, to initialize the array. Then another for loop is created (contained in the previous for loop) with index j, which indicates the current iteration of simulation k (that is why range(K[k]) is placed). Then contained in the for loop, the variable a is created to create an array with np.array, which represents the data packet per iteration (created previously). Placing the prices first and then the assignments of each agent, with an order preset previously, creating a 10x10 matrix, the 10 rows being the orders preset previously and the 10 columns the number of simulations for each order. Then a variable XX[j, :] is created to store the previously created matrix, storing in row j all the values ​​of the vector a of the matrix, which is a 1x10 row. Then all this is exported to excel. First, strpg is placed, which is the name of the sheet, placing format(k) to place a simulation per sheet, in this case there would be 20 sheets. Then the variable data_df is placed to create a table in excel with pd.Dataframe(XX), placing a matrix in each excel sheet. Then the data_df.to_excel command is placed that writes a writer object in excel, the name of the sheet with strpg and with an 8 decimal format using float_format. Then with data_fd a Walras assignment table is created, placing pd.DataFrame(Walras_alloc[0]). Then with data_fd.to_excel separate sheets are created for each of the agents, also containing a writer, the name of the sheet, which in this case is placed directly, is not within a variable (Walras_alloc) and 8 decimal places are placed with float_format. Then the same procedure is carried out but with the threshold prices of each agent per well. Then with writer.save() the excel file is saved and closed. Then, using a variable x, 100 equidistant points are created from each other with the np.linespace function, and then with np.amin the minimum of the utilities of all the agents in both BTE and Walras is found, the same with np.amax, but the maximum of the utilities of all the agents (in this case it is different from example 1 in both amax and amin because it is done using the assignment and price tests already done). This is done to measure the technical efficiency of BTE and Walras, where BTE is the X axis, which represents bilateral exchanges, and Walras is the Y axis, which is the optimal axis, generating a line x = y. It is used to see how far away or close to the theory are the bilateral exchanges vs the Walras, which are the optimal and theoretical ones respectively. Then, using a for loop, iterates and makes a scatter graph to see how the utilities of the bilateral exchange (BTE) vary vs the Walras utilities in a certain iteration K using plt.scatter where Utilities_BTE[k, :] is the utility vector of all agents in the simulation k of bilateral exchanges and utilities_wal[k, :] are the utilities corresponding to the Walrasian equilibrium, where each point in this regression represents an agent in a specific simulation. Plt.plot(x, x, …) graphs the line x=y with a dashed style (==) by setting np.linestyle to make this happen. What we are looking for with this is to see how efficient in terms of utility the bilateral market equilibrium is vs. the Walrasian equilibrium, which is the ideal. When the points on the bilateral equilibrium line move far away from the Walrasian equilibrium in terms of utility, it depends. If the utility line of the bilateral market equilibrium is above the Walrasian line, there is a surplus, therefore the agent has greater well-being. On the contrary, if it is below the Walrasian utility line, this surplus does not exist, therefore there is a loss of well-being. The x-axis is set with plt.xlabel to show the utilities of the decentralized mechanism, that is, bilateral exchanges, and on the other hand, on the y-axis, with plt.ylabel, the utility of the Walrasian equilibrium is set, that is, the artificial centralized equilibrium introduced in the paper. Then, in plt.legend, the legend is placed so that the k simulations can be tracked throughout the iterations due to the random nature of the negotiations. Each legend groups a set of points by the number of simulations there were, which in this case are 10, condensed into a single graph. Then, with plt.savefig, the figure is saved and shows the evolution according to the simulations of each of the agents of the relationship between the Walrasian equilibrium and the bilateral equilibrium to see the efficiency of the latter, as shown in the last figure of the paper on page 26.

**Lines 309 to 334**
```python
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
```
Finally, a print command is used to print the Walras prices to the console in matrix format with the equilibrium prices using the p_wertr variable. Table 4 (price results) is then prepared. The simulations to be displayed are then selected using the idx variable, which in this case will be 0, 4, 10, 13, and 14. The idags variable is then created, which labels the rows containing the goods, using the np.asarray command, which converts data into matrices. In this case, the rows are formatted with Latex $j=1$ and $j=2$. The column headers are then created using the labls variable in Latex format, including the prices of the previously selected trials and comparing them with the Walras prices. Then the variable T4 is created by creating a combined matrix generating a single column with np.column_stack with the threshold prices of goods 1 and 2 in the selected simulations, comparing it with the Walras prices of goods 1 and 2. Then a variable df is created that creates another table but exported through pd.dataframe with the data from T4, which is why it appears at the beginning of the parentheses of the command. Then the names of the columns and rows are placed, respectively with labls and idags. Then it is exported to LaTeX with df.to_latex, placing the file name in tex format. Indicating the number of decimals with float_format, in this case 4. With escape=false it allows LaTeX commands. Then, table 5 is prepared with the variable creating a 6x9 matrix (6x(3x3) for the 3 goods and 3 agents from the beginning). Initializing the matrix with np.zeros, where Ec_aux.n are the goods and Ec_aux.I the agents. Then two for loops are created, where in the first an index nn is created which is the current good in which this loop is iterating, contained within this there is another for loop with the index ii which is the index of the current iterated agent and finally another for loop is created contained within the previous 2 loops which with index jj is the current iteration of the selected simulations, which in this case are 5 in question. Then a variable is created within these for loops called Tab5[jj,3*nn+ii] which are the BTE assignments according to the iteration, the good and the agent in which the for loop is located. Then the same way is carried out with Walras, only outside the loop of the chosen simulations. Then, using the variables idags5 and labls5, the final details are completed, respectively, with the labels for the rows containing the selected simulations plus Walras and the column headers, where the values ​​are for each participating agent. Finally, the final details of the latex tables are completed, such as the number of decimal places each result should have.
#### Installing libraries and configurations to run the code
Once the packages have been installed and the Python environment has been created, you're ready to run the code. Only one more library needs to be installed at the Anaconda prompt. First, install openpyxl by typing "conda install openpyxl" at the prompt. Then, replace writer.save() with writer.close() in the code, updating the openpyxl package. This command is located on line 295 of the code; if not, search for it with Ctrl+F.

#### Outputs

First, the code generates a 3x3 matrix indicating the value of each agent's prices for each good, with the first column representing money, which functions as cash and is standardized.

[Imagen]

Then the inspection method changes with respect to example 1. Now for each good and agent (3 goods and 3 agents) there are different ways to inspect them. The first 12 iterations are fixed, that is, they follow a pre-established order to find the equilibrium and then the next 8 iterations are random where all agents can trade with anyone freely, without an established order. In this case, in the fixed ways of inspecting they will always go in pairs since inspecting in the order [0,1,2] is the same as [0,1,2] only the order in which non-monetary goods are inspected changes and this does not generate differences in the final result of the equilibrium. 0 is good 2 and 1 is good 1 in the case of the inspection order of non-monetary goods. Trade opportunities are generated by inspecting each of the agents in the established order and seeing with whom they can have a trade opportunity. In the random mode (set by ran in the output), agents are free to trade with the first person they have the opportunity to do so and jointly earn utility. The code after the first price matrix generates a dictionary with everything mentioned above to maintain order.

[Imagen]

The same thing happens next as in the previous code, except that it specifies the method for inspecting each trade and how this will be carried out. It specifies the standard deviation of prices that are below the tolerance established in the code, which is correct. Also, since it includes the pyutilib library, it specifies the time it took the algorithm to reach equilibrium, which allows us to draw the same conclusions as in Example 1.

[Imagen]

As before, it also generates the number of balances that were successfully reached, in this case all of them (20/20). It also generates the median of the number of iterations that the inspections took to reach the balance, which does not exceed 1030 iterations, and also the median of the times that the inspections took to reach the same balance, which does not exceed 5 seconds.

[Imagen]

Two matrices of 20 rows and 20 columns are then generated, each representing the number of inspections performed and carried out. The BTE algorithm is run on the first matrix, noting that the first two rows are zeros, and then the third and fourth rows begin to show values. This means that no way was found to improve prices between agents for the first two ways of inspecting (i.e. "fixed, [0,1,2], [0,1] or [1,0]), then with the third way of inspecting the combination of agents with this pre-established order was found to improve the prices of the goods, and as the third way of inspecting is equal to the fourth, it has the same improvement value which in this case is percentage as in example 1. This is so until row 13 where inspections begin randomly. Here you can see that inspections are no longer done in pairs but 1 by 1 (exceptionally row 13 does it in pairs to then complete the matrix) finding more ways to make trades and not with a pre-established order, but randomly, making things easier for the bilateral algorithm, making an improvement in prices in a more progressive and sustained way. Then again another matrix is ​​created of 20 x 20 but this time with the Walras algorithm and you can see that with this inspection method things become much more difficult for the reasons explained above and it is more difficult to find a balance having some row values ​​at 0.

[Imagen]

Then, as before, a price comparison matrix is ​​generated between Walras and BTE. This time, the inspection method is chosen equally at random, that is, two fixed iterations and two random iterations. This allows for a comparison between these two methods, not only comparing the algorithms but also the inspections. A significant difference can be seen between the inspection methods and their prices with Walras. First, the fixed inspection method yields significantly higher prices than the random inspection method, with differences not only for good 1 but also for good 2. This may be due to the restrictive nature of the fixed inspection method imposed on the BTE algorithm. This way, it cannot find the best way to combine agents to optimally improve their utilities, but rather in a forced manner, thus raising prices more indiscriminately. In contrast, random arrangements show that prices rise in a more controlled manner for each agent. Both inspections are far from resembling Walrasian prices.

[Imagen]

At the end, place a price vector that represents the final vector of the value of the prices of each of the non-monetary goods in Walras.

[Imagen]

We now begin by looking at the graph outputs for example 2.

One of the graphs generated, called Utility, is a graph that shows the evolution of agents' utilities in the different methods, both in "fixed" and in "ran". Comparing not only these inspection methods, but also comparing them with the Walras utilities for each inspection, the X axis being the Walras utilities and the Y axis the BTE utilities. Then, an x ​​= y function is plotted to measure the efficiency of the utility, that is, which gives a higher value in utility, if BTE or Walras, as mentioned for example 1. The points in the lower left side represent agent 1 (those above the random inspection and those below the fixed inspection). It can be observed how the utility of agent 1 is better for random inspections, having a higher utility than fixed inspections. Then the points furthest to the right are the utilities of agent 2 (the other way around, just like the next agent, with random inspection at the bottom and fixed inspection at the top). You can see here that random inspections are worse in performance than fixed inspections, where the utilities of agent 2 and 3 are very similar (agent 3, the last points on the right). This is because in fixed inspections it is easier for those with larger positions at the beginning to want to trade among themselves to improve their utilities in a more controlled manner, since agent 1 has much lower utilities with respect to their initial endowments, making it inconvenient for agents 2 and 3 to trade with the latter because their utilities would have to go down and this is not the idea for the BTE algorithm or it would have to be a very slow growth with respect to agents 2 and 3 who have the majority of the endowments of goods 1 and 2. Also, the way of inspecting can greatly affect this value, since the true optimal growths for one agent and another for utility are not found. For this reason, random inspections are better than fixed ones, seeing the example of the graph where the points of agent 1 equal the utilities of agents 2 and 3, having a balance in the utilities of each agent.

[Imagen]

A graph called Utilities_bte is then generated, which is made up of stacked bars, each part of which is color-coded to distinguish the three agents. This graph compares the utilities of each agent across different inspections. The y-axis represents the utility, and the x-axis represents the 20 inspections (12 fixed and 8 random). The last bar in the graph represents the Walras utilities, where the utility equality between agents 2 and 3, who have the largest initial allocations, can be observed. The huge difference with the utilities of agent 1 can also be seen (for clarity, agent 1 = blue, agent 2 = orange, and agent 3 = green). Here we can see the effectiveness of the "Ran" method, which helps the BTE algorithm find better balances and synchronously improve the utilities of each of its agents, unlike the "fixed" method, which causes the utilities of agents 2 and 3 to skyrocket and the utilities of agent 1 to plummet. The Ran inspection method is the best for equalizing the utilities of each of the agents, confirming the previous graph.

[Imagen]

Next, there's a graph similar to the one before, except it shows the agents' wealth for each type of inspection, plotting only the utilities for each agent's wealth using the same colors as before on the y-axis. At the end, a stacked bar representing each agent's Walras wealth is placed for comparison. It's clear that the wealth of agent 1 is greater than that of agents 2 and 3 in both fixed and random inspections. This is because, in trying to equalize their positions with agents 2 and 3, agent 1 tries to improve their positions as quickly as possible to equalize the positions of agents 2 and 3. Since fixed inspections are more complex, they have to improve their positions quite a bit in a small pair of iterations, thus improving their wealth more than agents 2 and 3, who are already balanced in terms of holdings of good 1 and good 2. This improves their wealth among themselves in a more controlled and less abrupt manner. It can also be observed that the wealth in random inspections for agent 1 is much greater than in those of 2 and 3, thus improving their situation and balancing their holdings with agents 2 and 3. Here we can see how random inspections with these three graphs cause the BTE algorithm to have a more balanced growth among all agents in both utilities and wealth. The fixed inspection method has many restrictions, which does not allow the BTE algorithm to act freely so that all agents improve their utility jointly. Finally, we can see the great inequality between the wealth of agents 2 and 3 and that of agent 1.

[Imagen]

Then a social utility graph is generated, which is the sum of the total utilities of all agents (from the Utility_bte graph the full bar. For example, for inspection 0 the sum of all the utilities of all agents is 80). In this graph it can be observed that the Walras social utility is always constant at 100, while the BTE utility is changing and at the beginning has a large value, which would be for the values ​​of the first inspections of the algorithm (fixed), where the utilities of agent 2 and 3 shoot up with respect to the utilities of 1, the utilities being uneven, where the holdings are concentrated in only 2 agents instead of all. Then, when we switch to the random inspection method, we can see that utilities drop significantly, indicating that agent 1, through the bilateral exercise, managed to equalize his holdings with agents 2 and 3, equalizing everyone's utilities and making the distribution of goods better for everyone. This is costly for utility, but equalizing agents 2 and 3. This can be seen in two ways. The increase in social utility is good since there is a society that is satisfied with the goods it has and they satisfy its preferences, but when those goods are concentrated in the hands of a few and there are a large number of agents or people who do not have the same goods, that value is just a facade, a lie hidden under numbers. Social utility worsens, not because those with greater positions produce less, but because agents with smaller positions have fewer restrictions on making their trades and producing in such a way as to equalize the positions of those who have more, having a lower Gini coefficient, and resulting in a more egalitarian society.

[Imagen]

After this, the same graph generated for exercise 1, called Page, is generated. This graph is the same as the previous one, only for two goods. The difference in prices can be seen in the boxplots, and how they vary much more violently than in the previous exercise. The difference in inspection methods greatly favors this, with a certain number of prices for fixed inspections and other prices for random inspections. These two methods, as mentioned above, significantly change the way the BTE algorithm operates. Thus, you can see how the box is quite elongated, indicating a large dispersion of the data between the prices at the bottom and those at the top of the boxplot. You can see how the median is closer to the third quartile, indicating a large dispersion of the data in the first section of the box (for both good 1 and good 2). In this way, the largest values ​​are close to the median, where the data are most concentrated, and the smallest values ​​are farther away and more dispersed. This indicates how different the inspection models are from one model to another, with smaller and more dispersed prices for fixed inspections and, conversely, larger values ​​for random inspections, especially for Agent 1.

[Imagen]

Then another graph is created called Pbar, which is different from the Pbar graph in example 1. Here you can see that the X axis represents the goods and the Y axis the prices. You can see different points distributed around the graph. You can see that the first points that are in blue in the lower left side of the graph are the representation of the random inspection prices. Then there is another set of points that are between 0.1 and 0.2 of each axis that are the prices of fixed inspections, demonstrating that they are better than those of random inspections. This is not necessarily better since it shows that there is a greater inequality between agents, thus not reaching equilibrium prices. Finally, there is a last red point, which is the Walrasian prices. The points that are closer to the theoretical equilibrium have a greater inequality between their utilities, as seen previously. Therefore, fixed inspections are worse than random inspections, which demonstrate that they have a more controlled growth in prices.

[Imagen]

Then another graph is created called Pbardiffs which is similar to the one in example 1. This graph shows that in random inspections there is greater equality of prices between agents and with fixed inspections, they are greater but have a greater dispersion, with unequal growth between agents for the reasons explained above.

[Imagen]

Finally, a new type of graph is created that was not present in the previous example: the edgebox. These edgeboxes plot how each good evolved in relation to the other, starting with the agent with the lowest number of positions (in this case, 1). It can be observed how both positions progressively and controlledly improve their assets 1 and 2 to reach equilibrium with the other agents, in this case, agents 2 and 3, who have the largest number of positions of assets 1 and 2. It can also be seen that with the fixed inspection methods, agent 1 did not reach equilibrium, but with the random method, they perfectly reached equilibrium with the assets of agents 2 and 3. This edgebox confirms the effectiveness of the random inspection method over the fixed one. The same applies to the edgebox that compares monetary positions with assets 1 and 2; they do not reach equilibrium with the fixed inspection method, but they do with the random method.

[Imagen]

### Example 3
#### Explanation
#### Code documentation
Here is where the documentation for example number 3 is placed. BTEQuad will be documented here, since its structure is different from the BTEs used previously.
#### BTEQuad Documentation

**Lines 1 to 8**
```python
from __future__ import division
import numpy as np
from scipy import optimize
import random
import time
from pyomo.environ import *
from pyomo.opt import SolverFactory,SolverStatus,TerminationCondition
from pyutilib.misc.timing import tic,toc
```
First, import the libraries described in the paper:
- Numpy (used for matrix operations)
- Random (used for random processing of matrices)
- Pyomo (used to solve optimization problems, using Walrascheck to verify market equilibrium)

**Lines 10 to 13**
```python
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)
```
Implement the random inspection of agents and goods mentioned in part 5 of the paper (random inspection strategy).

**Lines 19 to 30**
```python
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
```
The economy class is defined, where an economy is initialized with n goods and i agents, where in the paper model there are n+1 goods plus money. The following matrices are initialized with respect to the class created above:
- pag: Prices (related to threshold prices p_ij in the paper)
- delta: Price premium delta_ij to find an equilibrium
- allocations: Current allocations, which can also be seen as the current iteration x_i (holdings)
- e: Initial allocations, not yet iterated (initial holdings x_ij^0)
- alpha: Utility parameter (beta_ij in the paper for Cobb-Douglas functions)
- a: Linear utility coefficients
- b: Quadratic utility coefficients

**Lines 32 to 45**
```python
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
```
The evalp function is defined with the def command to calculate the threshold prices p_ij(x_i), where x is the numpy array (I, n) with the current allocations of goods for each agent. Then, a variable A is created to initialize the results matrix (agents x non-monetary goods). With the np.zeros function, the matrix is ​​created with self.I and self.n-1 so as not to include the non-monetary goods as indicated above. Then, it creates two for loops, the first with index i indicating the current iteration of the agents (hence range(self.I)), which consequently iterates over all agents. The second for loop, contained above the first for loop, has index j, that is, the current iteration over the non-monetary goods (hence range(self.n-1)), iterating over all the non-monetary goods. Then, contained below these two for loops is an if that indicates that if the amount of money is tiny, close to 0, the numerator (created with a variable num) is equal to the derivative of the quadratic part of the utility, subtracting the linear component a, exposed above with the quadratic part of b multiplied by the non-monetary allocations of the agents (the entire equation placed in an np.copy that serves to store independent data from a matrix). The den, on the other hand, remains as the derivative of the Cobb-Douglas equation of agents' preferences, with the alpha index multiplying the monetary allocation and raising it to the preference minus 1, as happens in derivatives when there is an exponent. All of this also within an np.copy within a variable den. Then the variable A[i,j] is created to perform the division between the derivative of the non-monetary allocations and the monetary allocations. That is, num/den. Then an else is created, still within the for loops mentioned above, where the case is placed when the money is positive greater than 0. The num and den are done the same as before, only for this other case, and the matrix A[i,j] is created again. It ends with a return A, which returns the matrix A with the threshold prices of all the agents and goods, which is ultimately what is calculated by dividing the derivatives of the assignments by the monetary assignments.

**Lines 48 to 54**
```python
  def xshort(self,i,j,pit):
        ut = lambda xi: -1.0*((self.allocations[i,0] - pit*xi)**(self.alpha[i]) + self.a[i,j]*(self.allocations[i,j+1] + xi) - 0.5*self.b[i,j]*(self.allocations[i,j+1] + xi)**2)
        LB = 0.0
        UB = (np.copy(self.allocations[i,0])/pit)
        result = optimize.minimize_scalar(ut, bounds=(LB,UB), method='bounded')
        #print(result.x)
        return result.x
```
The function xshort is defined as the optimal amount that agent i can purchase of good j at price pit. For this, the variable ut is created, which is the purchase utility function, where -1 represents an expenditure of money when acquiring new goods, subtracting in the same way the agent's monetary positions with the acquisition price of the good raised to their preferences alpha, which are their preferences. Then, a linear term of the acquisition of the goods is added to this, multiplying the linear term a by the acquired allocations plus those already held for the goods, and then the quadratic term is subtracted, which is the quadratic component of money, multiplying the quadratic factor b by the new acquired goods squared. Then, a variable LB is created, which is equal to 0 to avoid purchasing negative quantities, since it makes no sense. Then, a variable UB is created, which is the limit of the available money by proportioning the agent's monetary positions and the price of the goods, all of this saved in an np.copy. Next, a variable, result, is created to minimize the objective function (a negative utility since it is a purchase). It generates tuples of the lower and upper bounds, which are LB and UB (bounds), respectively. The optimal result is then returned with a return statement.

**Lines 57 to 63**
```python
 def xlong(self,i,j,pit):
        ut = lambda xi: -1.0*((self.allocations[i,0] + pit*xi)**(self.alpha[i]) + self.a[i,j]*(self.allocations[i,j+1] - xi) - 0.5*self.b[i,j]*(self.allocations[i,j+1] - xi)**2)
        LB = 0.0
        UB = np.copy(self.allocations[i,j+1])
        result = optimize.minimize_scalar(ut, bounds=(LB,UB), method='bounded')
        #print(result.x)
        return result.x
```
Then, xlong is defined as the optimal quantity of good j that agent i can sell at a given price, pit. The procedure is the same as before, except that instead of subtracting monetary positions, they are added together, and non-monetary allocations are subtracted.

**Lines 65 to 133**
```python
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
```
The main algorithm of the paper in section 5 is implemented, which also explains sections 3 and 4 of the paper, where the first four objects are defined:
eps_prices: equilibrium tolerance (epsilon p in section 5)
MAXIT: maximum number of iterations
lbda: delta reduction factor (lambda in section 5 of the paper)
delta_tol: delta tolerance
inspection: inspection strategy
After this, structures are generated to store the price history, allocations, deltas, and time, as well as a final iteration counter and the equilibrium state. It then saves independent copies of the individual initial prices in step 0, as well as the initial allocations and initial premiums. Then theorem 1 of the paper is applied where all agents must reach a price agreement, placing a maximum tolerance of the epsilon standard deviation that prices must fluctuate to close the trades and end the iterations reaching a price equilibrium with a maximum tolerance given by eps_prices. After the prices are evaluated, the delta premiums, which must become smaller and smaller so that in each trade there is a tendency towards a convergence between prices, for that there is delta_tol which is a maximum tolerance for the deltas once they are very small, that is, an equilibrium is being reached which prints the current maximum value of the delta_ij premiums through an np.max and the delta_tol which is the minimum tolerance configured for the premiums. If equilibrium is not reached through eps_prices and delta_tol, the following lines of if and elif code are activated. The first three lines of code with the if and random are to make combinations between agents randomly, so that all agents interact with each other and none are left out, so that it is not a predictable trade and no agents or goods are left out (explained on page 20 of the paper). Then there is the deterministic mode that makes combinations in sequential order, instead of random, removing trade opportunities between agents and goods, not obeying a random exchange. Finally there is the else method, which is to support the two previous methods, more similar to the sequential method than the random one. Likewise, the method that will be used the most is the random one, since it does not leave out any trade opportunities, since the sequential method leaves trade opportunities out by placing pairs of agents in order in each iteration, therefore evaluating each good in order agent by agent until an optimal exchange opportunity is found. Furthermore, its execution takes much longer when done in order, with random inspection being much more optimal and faster, except when a solution is being reached, in which case, the sequential method is better, since it performs an exhaustive search for the solution, case by case, so as not to exceed the given tolerances. After all the possible inspection methods, a negotiation parameter is set, equivalent to pi, which appears in section 3.2 of the paper, which is an intermediate price between sellers and buyers, to reach an agreement between them. i1 is set as selling agents, i2 as buying agents, and j as the goods traded on the market. This is where the possible combinations are evaluated with the necessary inspection method (random or sequential). It imposes the restriction that an agent cannot transact with itself. Then, with pip and pin, that is, the sale and purchase prices respectively, they reach an equilibrium with the delta premiums once transactions can no longer be made, activating the delta_tol and eps_price restrictions. There is also the pit, which is the equilibrium price at which agents decide to trade their goods between themselves, where the l_aux is activated, or the intermediate price pit, between two prices to reach the equilibrium of one. In this way, there are three ways to reach an agreement on a price: the buyer proposes it, the seller proposes it, or both prices are averaged using the intermediate price and exchanged at that price. The pip is the minimum price at which agent i1 sells their products, while the pin is the maximum price at which agent i2 buys. Then there are the maximum quantities that both parties are willing to exchange. The xin, in the case of xshort, is the maximum quantities that the buyer is willing to buy, redundantly, while xip in xlong is the maximum quantity that the seller is willing to sell. Both, according to their preferences, then, within an object called xij, store the minimum amount to be exchanged by both parties. After this, it is verified that the buyer's prices are higher than the seller's prices, in addition to always ensuring that the seller has enough stock to sell and also ensuring that the trades are economically viable and that the quantities sold are not small (which is explained in section 3.4 of the paper). Then, in the allocations, the trades are carried out using the restrictions imposed previously. Seller i1 receives the money using xij*pij and transfers their positions using -xij, the reverse for buyers. Finally, the threshold prices are saved in self.pag, evaluating new holdings of the agents and updating this value iteration after iteration. Continuing with the last block of code, if no possible trades are found using the trade_aux==0 function, the delta premium is decreased using lambda to find new trade possibilities, increasing the number of acceptable prices for the next iteration. Then, the else function stores the number of successfully executed trades using TimesChange, which, with KK, stores up to the iteration number reached. Then, the total number of iterations is saved in trade_aux, which can be used for convergence analysis. Then there is the if function, which is a warning if equilibrium was not reached with the maximum number of iterations using the MAXIT-1 function. That is, if the last possible iteration was reached and no possible solution was found, an equilibrium was not found with the precision eps_prices. Finally, with the Ev functions, the assignments, prices, and deltas are saved in each iteration of the algorithm, saving the behavior of the algorithm, that is, the history, which are returned with a return.

**Lines 135 to 192**
```python
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
```
The Walrascheck function is defined using the def command, which verifies whether the prices p form a Walrasian equilibrium. Then, a variable x is created that creates an optimal allocation matrix of (Ixn) with the initial endowments e, creating the matrix with np. zeros_like. Then, a for loop with index ii is created, which means the current iteration of an agent self.I, for all the agents in the matrix self.I, representing the optimization for each agent. Then, within this for loop, the variable model is created, which is an optimization model to do the same for each of the agents. Then, the variable model.n is created, which creates a set with the set command of the indices of the goods (for all goods, that's why range(self.n)). Then within this for loop the alpha_init function is defined through def, which are the agents' preferences, then within this definition the return of the Cobb-Douglas exponents for money is returned through return with self.alpha[ii] which is the alpha value of agent ii from the monetary attributes. Then the variable model.alpha is placed outside the return, but within the definition that creates an alpha parameter in the pyomo model initialized with the alpha_init function. Then the linear coefficients of the utility function are initialized through def. Then the variable j is created which returns 0 for the value of money and the coefficient a_ij for other goods through return and the if to establish the conditions of the problem. Then the variable model.a is created to define the parameter a in the pyomo model with the linear coefficients. Then in the following 6 lines of code the same procedure is followed, only with the Cobb-Douglas quadratic factor. In the following three lines, we proceed in the same way as with alpha, but with the initial endowments and the variable j included. Then, the prices for the model are initialized. An if statement is then created to establish the conditions. If the variable j is equal to 0, it sets the price of money to 1 due to the normalization of money. If not (else), it assigns the given input prices for the other goods. The price parameter is then created in the Pyomo model with the variable model.p. The limits for the decision variables (in this case, the allocation of goods) are then defined. An if statement is again placed to establish the conditions. If the variable j is equal to 0, money must be strictly positive, and otherwise, other goods can be zero or positive. This is because if the agent has no allocations, they must strictly have money; otherwise, they can have 0 and have other allocations with their prices. The function is then initialized for the decision variables, that is, the allocation of goods. It returns the current allocations using them as initial values. A variable called model.x is created, which represents optimization variables with the initialization limits explained above (using bounds). Then, the objective function is defined (using obj_rule), that is, the agent's utility to be maximized. We start with the variable exp1, which calculates the Cobb-Douglas component for money. This is followed by the variable exp2, which is the quadratic component for other goods (all the j's in the matrix corresponding to the linear factor multiplied by the initial allocations minus the quadratic factor times the initial allocations squared are added together using the sum command. By adding all the components of the matrix j). In this way, the total utility is then calculated by adding exp1 and exp2, returning the value with a return. Then, a variable called model.obj is created to maximize the objective function with the objective command and sense=maximize. Then, the budget constraint is defined where the total expenditure is less than or equal to the value of the initial endowment with a bc_rule. Then, a return is executed that places one of the constraints that the current allocations minus the initial endowments by their respective prices must be less than or equal to 0, this because the agent does not spend more than their endowment is worth (making a for loop at the end that goes through all agents j). Then, a variable called model.bc is created that adds the previous constraint to the optimization model with the Constraint command. Then, the variable opt is created, which is the instance of the IPOPT solver for nonlinear optimization. Then, with the opt.solve command, the optimization of the model is executed for the current agent. Then a for loop is created that goes through all the agents in the matrix with index j representing the iteration of the current agent. Below the for loop, the variable x[ii, j] is placed, which stores the optimal solution in the allocation matrix using the value command. Then the variable ES is placed, which represents the calculation of the aggregate excess supply for each good using np.sum, subtracting the initial endowments less the allocations. Then the condition that verifies the equilibrium that exceeds 1% of the total supply is created through the inequality where the left side contains the absolute value with the np.abs command of the ES variable explained above that is less than 1% of the maximum using np.max of the sum using np.sum of the initial endowments. Then it is printed with print that confirms that these prices are a Walras equilibrium and if not (with else), it is not a Walras equilibrium. Then it returns the values ​​of ES and x with a return that are respectively the vector of excesses and the optimal allocation matrix.

**Lines 194 to 230**
```python
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
```
Here the Walraseq function is implemented, which is used to calculate the optimal holdings of agents when there is a surplus in demand, greater than that described above. It is an algebraic method. It begins by entering the final allocations to work, which are stored in x, p for the initial prices, B for the sum of the alphas for each agent, and E for the aggregate endowments of each good. B is to see the agents' returns to scale, that is, how much they produce based on the change in inputs, and E the aggregate supply of each good, that is, the quantity of goods they are willing to exchange for a certain price. Then, the linear Walras problem is solved by calculating the agent's total income divided by the sum of their preferences by the share of the different goods in the agent's utility divided by the price, thus algebraically calculating agent i's demand for good j (this process is algebraic, not iterative). After this, aggregate demand must equal aggregate supply to reach equilibrium with the excess supply. The sum of the xij being the demand and the capital E the sum of the supplies, by summing the initial endowments eij, equating the latter to reach equilibrium. In this part of the code is the alphaij which is the share of good j in the utility of i, eil which is the initial endowment of agent i in good l and Bi, which as mentioned before is the sum of the preferences of agent i. All these terms are entered into a matrix that is divided into two parts, the main diagonal, where the aggregate supply of good j is subtracted and part of the aggregate demand that depends on the same good j, which is the division that was exposed before, between the total income of the agent and the sum of the agent's preferences. And the other part that is outside the diagonal which are all the captures of how the endowments of good l affect the demand for good j. This vector A is multiplied by the price vector and results in vector b which results in the goods that depend only on the provision of money since the price is already fixed by being numerary, that is, cash, which would be the silver left over due to excess supply and demand. After this, p is implemented, which is to solve the previous matrix system and sets money as numerary. Finally, a for loop is created to calculate the equilibrium allocations, obeying what section 5 of the paper says, where agents must spend a fraction alpha of their income on goods, where ul is the total income of agent i adjusted by B [i] and xij as the optimal allocation for good j. Finally, the tatonnement process is defined with Walrasdyn, which consists of using Walrasian methods to reach price equilibrium. The variable p is created, which is the initial copy of the price vector that avoids modifying the original using the np.copy command. Then the variables ES and x are created to calculate the excess supply and allocations of current prices with the command self.Walrascheck (p). Then the variable ld is created which is lambda, which is the price adjustment rate which in this case is 0.01. Then the variable MAXIT is created which is the maximum number of iterations allowed (in this case 1000). Then a for loop is made that iterates over all the iterations with index k. An equilibrium condition is created that is an excess greater than -1e-3 by placing the minimum value of the excesses ES. If this is the case, a success message is printed with print where the Walrasian equilibrium is reached. Then within the if, the equilibrium prices p are also returned. Then the other condition is created with an else with a variable p which is the price adjustment that increases if the excess is greater than 0 and decreases if it is less than 0 (p + ld * ES). The variables ES and x are then recreated to recalculate the new prices, and another if loop is created in case the iterations run out. A warning printout is then placed again, indicating that the maximum number of iterations has been reached, and the prices p found are returned.

**Lines 232 to 236**
```python
def gini(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))  
```
Finally, the code defines the Gini equation, which calculates absolute differences by iterating over each agent i, calculating the sum of the absolute differences between xi and all subsequent agents j>i. After this, it is normalized by dividing diffsum in n squared by x dash to obtain the Gini coefficient. Interestingly, the Gini coefficient decreases with each iteration, implying that a convergence is being reached across the iterations. It is also useful for evaluating inequality in allocations, that is, to see how equitably the goods between agents are distributed in the results of the bilateral algorithm vs. the Walrasian equilibrium.

Example 3 will now be documented, put in the repository as final 3.
#### Final3 documentation

**Lines 1 to 21**
```python
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
```
The following libraries are imported:
- From BTE import: Imports the economy class and auxiliary functions from BTE.py such as bilateral trading.
- From copy import deepcopy: To copy objects without sharing references.
- Import pandas as pd: To export data to Excel.
- Import gc: Garbage Collector (optimizes memory).
- From time: Measures time and handles date formats.
- Import os: System operations, for example, creating folders.
- Import matplotlib: To generate graphs.
- From mpl_toolkits: Support for 3D graphics (not used)
Styles are configured for graphs:
- rc(“font”...): Serves as a font for graphs, controlling typographical aspects, setting global parameters, and setting the preferred font for graph letters.
- rc(“text”:::): Enables text rendering using LaTeX, allowing the use of mathematical commands and professional typography.
- plt.rc(“text”..): Reinforces the previous configuration using the pyplot interface. It is redundant but ensures that LaTeX works.
- plt.rc(“font”..): Sets the serif font family for mathematical elements such as Times New Roman.
Then, the makeArrow function is defined with the def command. This is an auxiliary function for drawing arrows in graphs. A variable delta is created with two conditions using the if and else statements. If delta is 0.0001, there are two options: first, the direction must be greater than or equal to 0, or second, it must be equal to -0.0001. This determines the direction of the arrow in the graph (positive/negative). Then, the ax.arrow command creates an arrow from one point on the graph to another (from (pos, f(pos)) to (pos+delta, f(pos+delta))). Then two other variables are created within the ax.arrow command, which are head_width and head_length, which configure the size of the arrow head.

**Lines 24 to 42**
```python
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
```
We begin by defining example 2 with the def function, which implements an economy of 3 goods (including money) and the number of agents. The variable alpha is created, which represents the exponents of the utility of money, or, if you want to put it more simply, its preferences, and they are placed in a matrix with np.array. Then, the variable a is placed, which are the linear coefficients of the goods. Using the same command as before, a matrix is ​​created to place them. The variable b is created, which represents the quadratic coefficients for the goods, placed in the same way as before, in a matrix with np.array. A variable eps is created, which is a parameter for the initial allocation of money (which is 0.1). The variable e is also created, which is the initial allocation of the agents, as a matrix with np.array. Subtracting the variable eps from the first agent and the second as a constant. Then, the class Econ is created, where the economy of n agents and 1 goods, including money, is created. Econ.alpha assigns the preferences of the agents. Econ.a are the linear coefficients of the goods. Econ.b are the quadratic coefficients of the goods. Econ.e assigns the agents' initial endowments. Econ.allocations assigns the agents' goods after iterating the initial endowments for the first time and subsequent iterations as well. Econ.pag returns the results of each agent's prices after the iterations. Delta updates the delta values ​​when trades are no longer possible, increasing its value by 5 times until it is zero or negligible. Econ.delta assigns the deltas according to the current iteration. The result for the economy in question is then returned with a return.

**Lines 45 to 74**
```python
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
```
Next, a condition is created with an if statement that if it is __name__ it will be executed only when the file is run directly (not imported). This is important since it is the main entry point for the script. Then, np.random.seed is created to generate random results between pairs of agents and that they are not in order, and exfig is created for the format of the figures (PDF in this case). A test economy is created using Ec_aux. It is also necessary to save each economy in each of the tests, therefore the variable Econ is created. Next, it is necessary to know the equilibrium states, therefore, an empty dictionary is created, just like before, to know what this is. The variable xopt is also created, which also creates an empty dictionary to keep track of the optimal allocations. Next, it is necessary to know the number of simulations or tests, placing ntrials (which in this case will be 20). It is also necessary to save the allocations, prices, and deltas for each simulation, so EvPag, EvAlloc, and EvDelta are created. The number of trades for each iteration is then saved in the variable TimesChange. A variable K is then created that records the total iteration time for each simulation. An equilibrium condition is then created with q_status, where 1 indicates convergence to equilibrium and 0 indicates failure to do so, since, as mentioned before, each good, including money, is equivalent to 1. The execution times must then be calculated, measured using the variable ExTimes. The maximum standard deviation for prices is then set using EvSD. Equilibrium prices must also be evaluated per simulation, which is done using Evpbar. Finally, the agents' Walrasian prices and current Walrasian price allocations are evaluated using Walras_prices and Walras_alloc. The Col_c variables are then added for the graph colors. The BPT function then stores the equilibrium price information for all goods except money, generating a 2x3 matrix where the rows represent the simulations and the columns the quantity of non-monetary goods per agent. These boxplots show the dispersion of threshold prices. Wealth_bte calculates agent wealth for each agent i in simulation k to generate a comparison with the Walras method. Utilities_bte calculates the utility of each agent using the Cobb-Douglas functions and compares it with the utilities of the Walras equilibrium. Wealth_wal measures the wealth generated by the Walras equilibrium using Walraseqrtr() prices. Finally, Utilities_wal calculates the utility of the Walras equilibrium with its respective allocations. The variable p0 is created to store the initial prices with a matrix of only 1 using the np.ones command due to the normalization of money. Another variable p0[1:] is then created to store the initial prices of non-monetary goods and summed using np.max.

**Lines 79 to 125**
```python
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
```
Then a for loop is created that iterates over all simulations using index k for the current iteration. Then inside the for loop, the current execution number is printed with print using the .format(k) command. Then a new economy is created with the variable Econ[k] with the base parameters of example 2. Then the variables Econ[k].e[0,2] and Econ[k].e[1,2] are created, which serve to modify the initial endowments to vary the bilateral equilibrium experiments. The first one serves to reduce good 2 of agent 1 by 6*k units and the second the same only for agent 2 by increasing its units. Then the variable Econ[k].allocations is created, which updates the allocations as initial endowments. Then the variable Econ[k].pag is created, which recalculates the price thresholds using the evalp function, explained above for each of the agents' allocations. Next, the variable Walras_alloc[k] is created, which initializes the storage for Walras allocations by creating an array of zeros with the np.zeros_like command. A variable t is then created, which measures the current time taken to execute the bilateral algorithm, which will be measured later. The following lines of code then execute the bilateral algorithm with the following variables:
- eps_prices=1e-4: Threshold for considering prices to have converged (when the standard deviation of prices between agents is less than this value)
- MAXIT=250000: Maximum number of iterations allowed
- lbda=0.998: Price premium reduction factor deltaij
- Delta_tol=1e-18: Minimum value allowed for deltaij
- inspection=insp[k]: Strategy for selecting pairs of agents and goods with the orders established above. The returned results are:
    - EvPag[k]: Price history during the iterative process
    - EvAlloc[k]: Allocation history during the iterative process
    - EvDelta[k]: History of deltaij premiums during the iterative process
    - TimesChange[k]: Number of times transactions were made
    - K[k]: Total number of iterations performed
    - eq_status[k]: Indicator of whether equilibrium was reached (1) or not (0)
Then, the ExTimes[k] function calculates the time it took to execute the bilateral algorithm for the current test, and then prints the execution time. After this, the final equilibrium prices for the current test are stored as follows:
- Econ[k].pag: Stores the threshold prices for all agents (i.e., the limit when revenues equal costs).
- np.max: Takes the maximum for each good.
- Evpbar: Stores all goods starting from position 1.
Then, the maximum standard deviation of prices between agents is calculated for any good:
- np.std: Calculates the standard deviation per good.
- np.max: Takes the maximum of these deviations.
- Allows us to measure how close agents are to agreeing on prices (convergence).

Then the variables ES[k] and xopt[k] are created, which verify the Walrasian equilibrium. The first one stores the equilibrium state and the second one saves the optimal allocations for prices Ebpbar[:,k]. All of the above with the variable Econ[k].Walrascheck. Then a for loop is created that iterates over each agent i in economy k with Econ[k].I. Then a matrix z of size number of agents x the number of iterations k is initialized, first creating the variable Z and initializing the matrix with np.zeros and Econ[k].I and K[k], where there are I agents x K[k] iterations. Then another for loop is created within the previous one, but iterating over each iteration kk of the simulation k, which is why the index kk is created. Within this loop, another for loop is created that iterates over each agent ii, creating this index ii, which iterates over each of the agents in economy K. Then a variable Z[ii, kk] is created where the value of money (good 0) is assigned for agent ii in iteration kk, thus placing EvAlloc of economy k iterated over kk and ii, being 0. This implies the direct monetary component as mentioned above. Then another for loop is created within the previous ones, creating the index nn to iterate over the non-monetary goods, iterating over the economy of K with Econ[k] up to the value n-1. After this, the product quantity of the good x the agent's personalized price is added to the monetary possessions that the agent had at the beginning. For this, a variable Z[ii, kk] is created again that multiplies the agents' allocations by the price of these goods to measure their wealth and add it to the money they initially have in their positions. This is done with EvAlloc[k][kk][ii, nn+1], which is the quantity of good nn+1 of agent ii in iteration kk and with EvPag[k][kk][ii, nn] which is the personalized price (threshold) of agent ii for good nn. After this, agent i has the total value of his basket in a certain iteration kk. Then, a for loop is created that iterates over each agent in the economy, with ii as the index in the current iteration. Then, the plt.plot command is placed, which graphs the temporal evolution of the wealth of agent ii+1. Z[ii,:] contains the historical wealth on the axis and label creates an identifying label for the legend, in this case “agent”. Then a title is placed with the plt.title command that displays a title in latex format with a peso sign and curly brackets that represents Wealth = Money (x0) + summation (goods x prices). The plt.xlabel and plt.ylabel commands are placed that label the axes using mathematical notation (v = iteration, b = wealth). Then the plt.legend command is placed to place the legend on a specific side of the image (in this case in the upper left part of the graph). Then the figure is saved with plt.savefig (saving the image in PDF) and the image is closed with the plt.close command. Then the variable BPT [k,:] is created, which stores the maximum prices observed per good for later analysis (with the np.max command to find the maximum within a matrix). Then the variable Wealth_bte[k,i] is created, which calculates the total wealth as the product of the price point and the allocations of agent i (summing everything up with the command np.sum, where Evpbar[:,k] are the prices of each of the agents and Econ[k].allocations[i,:] are the allocations of agent i]). Then the variable for calculating utilities is created using the variable Utilities_bte[k,i], the first term being the agent's monetary allocations using allocations[i,0], raised to their preferences with alpha[i] being the utility of agent i's money. Then the total utility of goods is calculated with the command np.sum with the quadratic component b of the price of the goods and the linear component a of the price of the goods. Then the same thing is done but with the Walrasian method. It is finished with the gc.collect() command.

**Lines 127 to 138**
```python
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
```
First, a printout is generated reporting how many tests reached equilibrium (eq_status=1) vs. those that did not, showing a percentage of those that converged. Then, the number of iterations per simulation called K is printed, thus also calculating the median, which is the central value. Then, there is another printout to print the amount of execution time that each iteration took per test. Then, with the plt.figure command, a new figure is created for a new graph. With the plt.boxplot command, a box plot is generated with the maximum prices (BPT) without showing outliers. Then, a title is created with the plt.title command using mathematical notation with $ and r at the beginning. The labels for each axis are created with the plt.xlabel and ylabel commands, each representing goods and the other representing the price, respectively. Then, the figure is saved in PDF format with the plt.savefig command and finally, the figure is closed to free memory with plt.close.

**Lines 220 to 347**
```python
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
    plt.figure()
    for k in range(ntrials):
        plt.scatter(Evpbar[1,k],Evpbar[2,k],label='Trial {}'.format(k+1))
    plt.xlabel(r'$p_1$')
    plt.ylabel(r'$p_2$')
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig('Pbar.'+exfig)
```
It starts by creating the variable lnst which are the line styles to differentiate the executions (three styles: solid, dotted and dashed). Then a variable created cols is created to place RGB colors (red, green and blue) normalized between 0 and 1. Then a custom color palette is created, to finally place the last color of the palette. Then a step variable called delta is created to create a coordinate mesh. Then the variable s is created, which calculates the total supplies of each good using the np.sum command with the initial provisions added by column using axis = 0. Then the variable x0 is created, which provides a range of values ​​for good 1 that can be from 0 to s [1] with the variable created previously, delta. In the following line of code the same thing happens but with good 2. Then the variables X0 and X1 are created to create a 2D mesh to evaluate functions in the entire space using the np.meshgrid command that transforms matrix vectors into matrices with the variables x0 and x1 created previously. Then the variables fig and ax are created that initialize the figure and the axes for the Edgeworth box which is a graphical representation that distributes the allocations of two goods between two agents (all this with the plt.subplots command). Then the variable Z1 is created to calculate the utility of agent 1. The first term of the sum ((Ec_aux.e[0,0])** (Ec_aux.alpha[0])) calculates the utility of money for agent 1 using the Cobb-Douglas function. Then the second (Ec_aux.a[0,0] * X0) and the third term of the sum (Ec_aux.a[0,1]* X1) represent the linear part of the utility for goods 1 and 2. Where a[0,0] and a[0,1] are coefficients that measure the initial marginal utility and X0 and X1 are the quantities consumed of goods 1 and 2. Then the terms that are being subtracted in the variable Z1(- 0.5*Ec_aux.b[0,0]* X0** 2 - 0.5*Ec_aux.b[0,1]*X1**2) represent the quadratic part of the utility that represents satiety, that is, it is decreasing. Where b[0,0] and b[0,1] determine the rate of decrease in utility, where 0.5 is a normalization constant. Then the same procedure is followed for agent 1, only subtracting the consumption of agent 1. Then, a for loop is created that iterates over each simulation of the experiment with an index k that represents the current simulation being iterated. Within this for loop, the variable XX is created, which is the initialization of the matrix for X coordinates, that is, good 1. In the following line of code, the same is done with good 2 (both matrices created with the np.zeros command, to initialize a matrix of 0). Then, within the previous for loop, another for loop is created that iterates over each simulation of the previous simulation k, now calling the index kk. Within this cycle, the variable XX[kk] is created, which assigns the quantity of good 1 to agent 1. In the following line of code, the same thing happens but for good 2. Then, with the plt.plot command, the assignment trajectory is graphed, placing the variables XX and YY first, labeling the name of the legend that will be executed with label. With line thickness 1.5. Then the linestyle variable is created where the line style is solid or dotted and each line has a specific color using the color variable and its normalization established at the beginning. Then the lbl0 variable is created that creates text in LaTeX format to label the starting point. lblbx is created to create latex text to label the end point. Then with the plt.annotate command, annotations are added in coordinates (X,Y) of the starting point with dynamic displacement, each coordinate point in each simulation moves according to the equations established by the variable xy. Then, in the next line of code, the same thing is done but with the opposite movement to the initial one to find a balance. Then a for loop is created that iterates from 1 to K[k]-100 in steps of K[k]//15, that is, arrows every 15 iterations. Using the kd index to traverse all the iterations and create the arrows. Then with the plt.arrow command it creates an arrow in (XX [kd], YY [kd]) with direction ((deltaX) / 10, (deltaY) / 10). Then below this for loop the variable shape, lw, length_includes_head and head_width are created that configure the arrow, where each one represents respectively in the arrow configuration, full head (shape = full), without border (lw = 0) and the size of the head (head_width = .1). Then the color of the arrow is created according to the color variable exposed at the beginning with an RGB color palette. Then in the next 4 lines of code with the plt.annotate command (in code line order) it creates an agent 1 position label, an agent 2 position label, an execution 0 legend and an execution 1 legend. Then with an xlabel and ylabel the axes are labeled where each one represents good 1 and good 2 respectively. Then, the title of the graph is created with the ax.set_title command. In the following two lines of code, the limits of the x and y axes are set using the ax.set_xlim and ax.set_ylim commands respectively. Then, the background color of the graph is set with the ax.set_facecolor command. With plt.grid, a grid is placed to better work the arrows created previously and see where the exact points in question are. Then with plt.tight_layout, which is an automatic margin adjustment. Then, the graph is saved in PDF format with the file name Edgebox12.pdf. The image is closed with plt.close. Then, a new figure for a scatter plot is created with plt.figure. A for loop is created that iterates for each simulation with index k. Then, the scatter plot is created using plt.scatter for points p1 and p2 (either 1 or 2). The axes are then titled using the plt.xlabel and plt.ylabel commands, each corresponding to the price of good 1 and the price of good 2. A legend is created using the plt.legend command, located at the top left, without a frame. Margin adjustments are then created using the plt.tight_layout command. Finally, the graph is saved in PDF format using the plt.savefig command and named Pbar.pdf.

**Lines 349 to 422**
```python
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
```
Two variables, fig and ax, are created, which create a new figure and graph axes for price thresholds. Then another for loop is created to iterate over each simulation of the experiment with index k, which represents the current simulation being iterated. Then, a variable XX is created again, this time initializing an array for prices of good 1, just like in the next line, only with good 2. Then another for loop is created within the previous one to iterate through the simulation of the previous simulations with index kk, which represents the simulation of the current simulation being iterated. Then, the variable XX[:,kk] is created, which stores the threshold prices p1 for all agents, the same happens in the next line of code, only for threshold prices p2. Then another for loop is created that iterates over each agent with index i, representing the current agent being iterated. Then a variable called line is created that graphs the price evolution using the plt.plot command, with XX[i,:] and YY[i,:] axes, this represents each agent's good. In the following line of code, the style and unique color of the graph are established. Then in the following 4 lines of code, the same procedure is followed as before with the arrows, only now with the threshold prices. After these 4 lines of code, the annstr variable is created, which is the latex text for the initial point. Then, with the plt.annotate command, annotations of the initial price p0 are added, displaced both on the X and Y axes with font size 8. Then, with the plt.scatter command, a circular point is drawn at initial coordinates (XX[i,0],YY[i,0]) with normalized color. Then, a condition is created with an if that verifies if execution k reached equilibrium. Within this if statement, another if statement is created that only processes the first agent (that's why if i == 0). Again, using the plt.scatter command, the final position is marked with a + symbol of the same color. Then, another annstr variable is created that prepares LaTeX text for the equilibrium price. Then, within the previous if statement, the plt.annotate command is placed, which adds the annotation of the displaced equilibrium price with a larger font (11) than before. Then, the other condition is placed with else, in case the equilibrium is not found. A plt.scatter is created again that indicates the final position with a red x in case the equilibrium has not been reached with the previous indications. Another annstr variable is created that places alternative LaTeX text showing the final price per agent. Then, another plt.annotate command is placed that annotates the adjusted displacement per agent (1-i). Then, in the following 4 lines of code with the plt.annotate command, manual labels are created for (in order), on line 1, execution 0 of agent 1 at position 16,21 with source 8, on line 2, execution 0 of agent 2 at position 22.5,16, on line 3, execution 1 of agent 1 at position 17,12.2, on line 4 the manual label for execution 1 of agent 2 at position (22.5,22). Then a title was created with LaTeX mathematical notation with the plt.title command and its respective axis names with plt.xlabel and plt.ylabel labeling each axis as "Threshold for good 1 and the other Threshold for good 2 (price). Then the limits of the x and y axes are set respectively with ax.set_xlim (16,24) and ax.set_ylim (12,24). Then with the plt.tight_layout command the margins of the graph are automatically adjusted. Then with the ax.set_facecolor command the background of the graph area is colored. With plt.grid the grids are placed to make it easier to interpret the equilibrium. After saving the figure with plt.savefig in PDF format with the file name Evbarp. The graph is finally closed with plt.close. Finally, an excel file is created using the writer variable to save the results. Another for loop is created to Iterate over each simulation of the experiment with index k. Then, the variable XX is created, which initializes an array to store 11 variables per iteration. Then, another for loop is created for the simulations of the simulations with index j. Within this for loop, the variable a is created, which is a matrix containing the iterations, the prices of agent 1 and the prices of agent 2 in each of the three columns respectively. Then, the assignments of agent 1 and agent 2 are placed in the next two columns, leaving a vector for each row. Then, the variable XX[j,:] is created, which stores the data within the matrix of each vector. Then, the variable strpg is used to name the Excel sheet. With the variable data_df, the matrix is ​​converted to a pandas dataframe. Then with the data_df.to_excel command, each data is written in Excel with 8 decimal places. Then with writer.save() the Excel file is saved. Then the variable Tab7 is created, which is the LaTeX table matrix that contains 4 rows and 4 columns (created with np.zeros). Then another for loop is created with index kk to iterate each execution and another for loop within this with index ii to iterate each agent. Within these for loops, the variables Tab7[Ec_aux.I*kk+ii,n] are created, which in the following four lines define (in order) the initial price p1, the final price p1, the initial price p2, and the final price p2. Then in the last 4 lines, the aesthetics of the LaTeX tables (titles and labels) are placed.

#### Installing libraries and configurations to run the code

For this example, you don't need to install any additional libraries; they're all already installed, and the Python 3.11 environment is ready to use. Go directly to the outputs.

#### Outputs 

First, as in the previous two examples, it generates written outputs within the same console. First, the code prints the iteration number it is in. It first prints iteration 0 with the agents' main allocations before making the trades. It can now be seen how one of the agents starts with non-monetary allocations at 0, which in this case is agent 1 for good 1, and agent 2 starts with allocations of both goods 1 and 2 but with little money (since it cannot be 0, since the marginal utility function of money tends to infinity when money is 0). This is the first time that a problem of this type has been faced, complicating the improvement of agents' utilities since agent 2 does not have such basic allocations as money, which leaves him with the only option left: to sell to improve his utilities, this being his incentive. On the other hand, agent 1 lacks good 1, therefore, what remains for this agent is to buy the good that agent 2 lacks in order to jointly improve utilities. Therefore, in this example, the equations of the bilateral exchanges from section 2 of the paper, specifically 2.14 and 2.20, can be better seen. After generating this matrix, the times it took for the trades between the agents to reach equilibrium are generated. In this case, in trade 0, it took 0.54 seconds to find equilibrium, also placing the maximum tolerance for the standard deviation of prices, giving a lower and achieving a first equilibrium. Unlike the other exercises, this one has the possibility of seeing whether it is a Walrasian equilibrium or not, since it is not as before and utilities and prices have other ways of being calculated, no longer through a Cobb-Douglas utility function. This new function with which utility is calculated is strongly concave, allowing to reach an equilibrium, having a second derivative less than 0, ensuring that in at least 1 iteration BTE will reach equilibrium.

[Imagen]

After this, we move on to trial 1, which would actually be the second iteration. Here, the initial goods matrix is ​​the same, except that the quantities of good 2 are exchanged between agents. The code, as explained in the documentation, allows a maximum of 1,000 iterations before reaching equilibrium. In this case, it fails to reach equilibrium before 1,000 iterations, forcing the BTE algorithm to cut off and declare that equilibrium could not be reached. It then enters the prices at which the algorithm remained before reaching a possible equilibrium. It then generates another matrix, which is the delta of the prices where the algorithm remained, which did not reach the maximum tolerance required by the exercise, which was 10e-18. The exercise ended up at approximately 9e-09, which is quite far off.

[Imagen]

Then, the median of the iterations where equilibrium was reached, the median of the times where equilibrium was reached, and the number of tests that reached equilibrium are generated, which in this case were 1 out of 2. This is generated because the algorithm reached the maximum number of iterations to reach equilibrium, having very demanding tolerances so that prices go down and find equilibrium.

[Imagen]

Now we will proceed to the graphic outputs that the code launches.

First, a graph called Pag is generated, which is the same as the two previous examples. A boxplot shows the price dispersion in each iteration. It can be seen that the prices are well balanced. The median is in the center, generating a low dispersion in the data for both goods.

[Imagen]

Now, a graph named Evbarp is generated that shows the prices at which equilibrium is reached in trial 0 for both agent 1 and agent 2, showing how the prices for each good evolved for each person in each iteration. The graph shows that equilibrium was reached at the approximate value of (19,18) for the price of good 1 and 2 respectively in iteration 0. This is the equilibrium prices for the holdings of agents 1 and 2. It can also be seen that iteration 2 (1 according to the exercise) does not converge as stated above. The lines are cut and a cross indicates where the equilibrium was truncated. The matrix mentioned in the written outputs indicates the prices where the exercise was truncated, without reaching equilibrium. The direction taken by the price lines is subject to the tolerance established in the exercise and how it goes down until reaching the maximum tolerance of deltas until reaching equilibrium. Where agent 1 is the buyer and agent 2 is the seller.

[Imagen]

A graph called Edgebox12 is then generated, which deals with the same idea as the previous exercise: the agents balance their holdings until they reach a maximum level of well-being for both. You can see here how agents 1 and 2 in iteration 0 balance their non-monetary positions until they reach the optimum, where they cannot be improved further, remembering that there is no Cobb-Douglas profit. Then, you can see iteration 1 below, which ends without converging, returning to 0, as confirmed by the exercise.

[Imagen]

Next, a graph called Pbar is created, which is distinct from the other Pbars generated by the other exercises. It represents the prices where equilibrium is not reached in iteration 2, specifically for agent 1, the equilibrium prices reached before the algorithm was truncated due to reaching the maximum allowed iterations. Agent 2 does not appear because it is not within the graph's price ranges. This could be changed to show where agent 2's price is truncated. It also already appears in the Evbarp graph, just to confirm.

[Imagen]

Finally, wealth evolution graphs are created for iterations 1 and 2. Wealth is calculated using the formula in the graph's title, where the first x in the sum represents the initial monetary endowment, followed by the utilities generated by the BTE equilibrium. In this way, it can be observed that both curves stabilize at a certain point, achieving a wealth equilibrium in iteration 0. Not mentioned previously, but the x-axis represents the iterations that have passed in the exercise, and the y-axis is a factor b that accompanies the quadratic variable in the utility function, confirming that the function is concave and reaches an equilibrium, which is doubly differentiable. As the value of b increases, the limit of the quadratic function becomes smaller, which is not good, so after a certain point, the value drops from 170 to 150, so as not to lose the opportunity to obtain maximum utility. Then, for the second graph of the second iteration, it can be seen that an equilibrium is not reached and that the value of b is becoming increasingly larger, affecting the marginality of the problem and reducing utility. The wealth of one increases, and that of the other remains constant over the iterations. One has excessive growth (agent 2) and the other remains constant (agent 1), where an equilibrium is not reached.

### Model extension

Finally, we will detail the code modifications for each example to measure new variables in detail that are not measured in the graphs generated by each example as output. These graphs will help clarify other outputs generated by the same exercise and validate that the BTE algorithm is better than the Walras algorithm and always approaches theoretical equilibrium, improving profitability.

#### Spider graph

Two fundamental spider plots were created. One to observe the evolution of agent assignments in each trial and how close they are to the theoretical equilibrium after their iterations in each trial. The other to observe the agents' utilities and how they differ from Walras utilities and how they have evolved relative to their initial utilities.

The spider graphs are subdivided, and each agent is placed in a specific trial of an example, since the logic doesn't change between examples due to changes in assignments or utilities. The code and its documentation are shown first (each generated code is placed at the end of each code).

#### 1. Utilities

**Documentation**
```python
import matplotlib.pyplot as plt
import numpy as np

# Check BTE convergence for each trial
for k in range(ntrials):
    if eq_status[k] == 0:
        print(f"Warning: The trial {k+1} did not reach equilibrium in BTE.")

# Generate spider graphs for each trial
for k in range(ntrials):
    # Calculate initial utilities using initial allocations
    U_initial = np.array([np.prod(EvAlloc[k][0][i,:]**Econ[k].alpha[i,:]) for i in range(Econ[k].I)])
    
    # Final BTE utilities (already calculated in Utilities_bte)
    U_final_BTE = Utilities_bte[k,:]
    
    # Walras utilities (already calculated in Utilities_wal)
    U_walras = Utilities_wal[k,:]
    
    # Create the spider chart
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360/5), labels=[f'Agent {i+1}' for i in range(5)])
    
    # Set fixed radial axis limit from 0 to 45
    ax.set_ylim(0, 45)
    

    # Set ticks on the radial axis every 5 units
    ax.set_yticks(np.arange(0, 46, 5))
    ax.set_yticklabels(np.arange(0, 46, 5))
    
    # Prepare the data for the chart (close the polygon)
    theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))
    
    U_initial_plot = np.concatenate((U_initial, [U_initial[0]]))
    U_final_BTE_plot = np.concatenate((U_final_BTE, [U_final_BTE[0]]))
    U_walras_plot = np.concatenate((U_walras, [U_walras[0]]))
    
    # Plot the series with specific colors and markers
    ax.plot(theta, U_initial_plot, label='Initial', color='b', marker='o')
    ax.plot(theta, U_final_BTE_plot, label='Final BTE', color='r', marker='s')
    ax.plot(theta, U_walras_plot, label='Walras', color='g', marker='^')
    
    # Adjust the legend to be at the top
    ax.legend(loc='upper right', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # Add title with trial number
    ax.set_title(f'Utilities for Trial{k+1}')
    
    # Save the chart as PDF
    plt.savefig(f'Utilities_trialwithoutwalras{k+1}.pdf', bbox_inches='tight')
    plt.close()
```

The #s are the lines of text corresponding to the documentation. In the first example, two graphs had to be drawn with the utilities because Walras's utilities were very uneven with respect to BTE's utilities. This is because Walras's methods for finding equilibrium are not optimal for all agents to improve their utilities in a controlled and sustained manner. In contrast, BTE shows how utilities improve in a sustained and equal manner for all agents. The axes represent each agent, with 5 for each. The numbered circles represent the utility value, which varies by agent. The rest is specified in the graph itself.

![Image](https://github.com/user-attachments/assets/ef80e9a0-1167-4a82-ab23-80c04a36b4dc)

Here is the code that contains the spider graph with the Walras utilities with its respective documentation.

```python
import matplotlib.pyplot as plt
import numpy as np

# Check BTE convergence for each trial
for k in range(ntrials):
    if eq_status[k] == 0:
        print(f"Warning: The trial {k+1} did not reach equilibrium in BTE.")

# Generate spider graphs for each trial
for k in range(ntrials):
    # Calculate initial utilities using initial allocations
    U_initial = np.array([np.prod(EvAlloc[k][0][i,:]**Econ[k].alpha[i,:]) for i in range(Econ[k].I)])
    
    # Final BTE utilities (already calculated in Utilities_bte)
    U_final_BTE = Utilities_bte[k,:]
    
    # Walras utilities (already calculated in Utilities_wal)
    U_walras = Utilities_wal[k,:]
    
   # Print values to check
    print(f"Trial {k+1}:")
    print("U_initial:", U_initial)
    print("U_final_BTE:", U_final_BTE)
    print("U_walras:", U_walras)
    
    # Check for NaN or infinity in U_walras
    if np.any(np.isnan(U_walras)) or np.any(np.isinf(U_walras)):
        print(f"Trial {k+1}: U_walras contains NaN or inf")
    elif np.all(U_walras == 0):
        print(f"Trial {k+1}: U_walras is all zeros")
    

    # Create the spider chart
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360/5), labels=[f'Agent {i+1}' for i in range(5)])
    
    # Calculate radial axis limit dynamically
    max_u = max(max(U_initial), max(U_final_BTE), max(U_walras))
    ax.set_ylim(0, max_u * 1.1)
    
    # Set ticks on the radial axis every 5 units
    ax.set_yticks(np.arange(0, int(max_u * 1.1) + 1, 5))
    ax.set_yticklabels(np.arange(0, int(max_u * 1.1) + 1, 5))
    
    # Prepare the data for the chart (close the polygon)
    theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))
    
    U_initial_plot = np.concatenate((U_initial, [U_initial[0]]))
    U_final_BTE_plot = np.concatenate((U_final_BTE, [U_final_BTE[0]]))
    U_walras_plot = np.concatenate((U_walras, [U_walras[0]]))
    

    # Plot the series with specific colors, markers, and line styles
    ax.plot(theta, U_initial_plot, label='Initial', color='b', marker='o', markersize=8, linestyle='-', linewidth=2)
    ax.plot(theta, U_final_BTE_plot, label='Final BTE', color='r', marker='s', markersize=8, linestyle='--', linewidth=2)
    ax.plot(theta, U_walras_plot, label='Walras', color='g', marker='^', markersize=8, linestyle=':', linewidth=2)
    
    # Adjust the legend to be at the top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
   # Add title with trial number
    ax.set_title(f'Utilities for Trial {k+1}')
    
    # Save the chart as a PDF with high quality
    plt.savefig(f'Utility_trialwithwalras{k+1}.pdf', bbox_inches='tight', dpi=300)
    plt.close()
```
This code generates the spider graph with Walras utilities, and you can see that they are much larger than the BTE. This clearly represents a total imbalance in the agents' holdings and indicates that the Walras algorithm is very inefficient in achieving an equilibrium of holdings and utilities for all agents, as explained repeatedly above.

![Image](https://github.com/user-attachments/assets/9e6a004d-dd36-408a-a7eb-b23f6ee2e508)

#### Assignments

**Documentation**
```python
import matplotlib.pyplot as plt
import numpy as np

# Define the categories (the 10 goods)
categories = list(range(10))

# Create a spider graph for each agent and each test
for k in range(ntrials):  # For each test (ntrials = 10)
    for i in range(5):  # For each agent (0 to 4)
        # Create figure with polar projection
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
       # Plot initial assignments
        initial_alloc = EvAlloc[k][0][i, :] # Initial assignment for test k and agent i
        ax.plot(np.linspace(0, 2 * np.pi, len(categories), endpoint=False), initial_alloc, label='Initial', marker='^', color='green')
        
        # Plot Walras allocations
        ax.plot(np.linspace(0, 2 * np.pi, len(categories), endpoint=False), x_we[i, :], label='Walras', marker='o', color='blue')
        
        # Plot BTE assignments (end of each test)
        bte_final = Econ[k].allocations[i, :] # Final BTE assignment for test k and agent i
        ax.plot(np.linspace(0, 2 * np.pi, len(categories), endpoint=False), bte_final, label=f'BTE Proof {k+1}', marker='s', color='red')
        
        # Configure axes
        ax.set_xticks(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
        ax.set_xticklabels([f'Good {j}' for j in categories])  # Label the assets as 'Good 0', 'Good 1', etc.
        
       # Adjust the y-axis range according to the data
        all_values = list(initial_alloc) + list(x_we[i, :]) + list(bte_final)
        max_val = max(all_values)
        ax.set_yticks(np.arange(0, max_val + 10, step=10))
        
        # Title and legend
        ax.set_title(f'Agent Assignments {i+1}, Proof {k+1}')

        ax.legend(loc='upper right')
        
        # Save the chart
        plt.savefig(f'Agent_Assignments_{i+1}_proof_{k+1}.pdf')
        plt.close()
```

It can be seen in all the generated graphs that there are 10 axes this time, representing the assets of each agent, each generated graph representing the allocations for each good for each agent in each of the corresponding tests (i.e., 50 spider graphs), generating their initial and final endowments and their comparison with the Walras endowments. This demonstrates how close the allocations of each agent's asset after making the trades are to those of Walras, thus demonstrating that the equilibrium of the holdings is similar to the theoretical holdings. It is a very good approximation of what is wanted, that is, to demonstrate that the BTE algorithm is better than the latter, even matching the theoretical holdings and being better in terms of utilities for each agent, improving in a controlled and progressive manner.

![Image](https://github.com/user-attachments/assets/529ca0c8-7175-459b-aac2-b6fc6bd8c53f)





















