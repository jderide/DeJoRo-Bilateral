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

![Image](https://github.com/user-attachments/assets/d95470eb-e028-4dc2-8863-dbf4d645ebcc)
![Image](https://github.com/user-attachments/assets/7c47fc4a-c48c-493d-be66-4f0c45a73b7c)

After this, a table is created where 4 tests of the algorithm (columns) are chosen for the 9 goods (rows) where the prices varied for each good are compared to make the comparison with the Walras prices, which are fixed without iterations. A price variation can be observed for good 1 between 0.9093905048137538 and 0.9473978286534999, coinciding with table 2 of the paper on page 21 of this, as well as the other goods. With respect to the Walras price, a similarity can be seen, indicating that the best way to reach equilibrium is with BTE, seeing that this value can be reached by the different ways of being able to trade between the different agents and goods and not just in one way, as is the restrictive Walrasian model. The BTE algorithm is highly efficient when searching for equilibrium prices, seeing how by establishing a maximum tolerance, prices remain stable in the different trades between agents with the different goods, obeying this slight restriction so as not to create discriminatory prices while maintaining utility between agents, seeking to improve their utility in each trade, quite the opposite of Walras, which is an established algorithm and does not allow free trading. 

![Image](https://github.com/user-attachments/assets/4c24ccfe-94c8-410e-9808-4e4602af4e1e)

The rows and columns are then inverted, and the change in the allocations of each good for each of the four iterations randomly chosen by the code is observed, comparing the BTE allocations with the Walras allocations. It can be observed that the BTE allocations are very close to those of Walras and vary between very close values. In the example, the BTE allocations for good 1 for the four chosen iterations vary by a maximum of approximately 7 units, between the values ​​58 and 51. This good has a Walras value of approximately 51. It can be seen how agents in the BTE algorithm can correct their allocations to improve their positions jointly rather than individually, reaching a value greater than the Walras allocations. This also speaks to the dynamism of the algorithm in seeking the best allocations for everyone, better than those of Walras, which is an algorithm that makes trades with restrictions and in a single test.

![Image](https://github.com/user-attachments/assets/2221f4bc-e612-4fe6-9d83-b97c2d843de5)

Now comes the section where you can see the graphics provided by this code.

First, the code outputs the graph of Walras utilities vs. BTE, placing each algorithm on the y and x axes respectively. Then, an x=y function is plotted to observe which of the two algorithms had the highest utility (Points above the function, the Walras utility is better. Points below the function, the BTE utility is better). It can be observed how all the points for each of the iterations show that the Walras utility is better, all being on the dotted function line. This is not necessarily optimal; this means that the Walras utilities for each of the agents are completely unequal. This algorithm, having restrictions, does not allow its agents to want to improve their utilities jointly such that all have a similar utility. It can be seen in the 5 points plotted in the graph that there is a large difference between the agents' utilities for Walras. BTE, on the other hand, has lower utilities but manages to compensate by making everyone's utilities similar, with no inequality between agents.

![Image](https://github.com/user-attachments/assets/30d71423-c23c-4bd1-873d-56ae1e0914da)

The code then generates a wealth graph, which follows the same logic as the first graph. Here, we can conclude the same thing as before. BTE's wealth is similar across agents and worse than in Walras, but Walras has very unequal wealth across its agents.

![Image](https://github.com/user-attachments/assets/14c287d8-6bfa-4113-b731-ef7f5cd01000)

A graph of the BTE vs. Walras Gini index is then drawn, indicating the inequality between each method, using the same logic as the other two graphs. As expected, the Walras Gini is greater than the BTE and remains at 0.4, as can be seen in the graph. The BTE Gini grows steadily with each iteration, with no more than 0.10 points on the Gini index, ensuring equality between agents.

![Image](https://github.com/user-attachments/assets/f2ca0ea8-24fc-46de-9a9f-31c5a242ad9d)

Ten graphs are then generated showing how utilities have evolved across the different trials of a test for each agent. The concave behavior of the curves with positive slope can be observed, as stated in the paper, reaching a limit where agents can no longer improve their utilities. The utility function is a Cobb-Douglas function that is convex, and the paper states on page 6 that when the preference equation is convex, it will always have a concave subset, as demonstrated by the graph. It can be seen that there is a relatively balanced utility among each of the agents in each of the iterations. The utilities are plotted on the y-axis, and the iterations of each test are plotted on the x-axis. This shows roughly at which iteration the utility equilibrium was reached.

![Image](https://github.com/user-attachments/assets/fdb6ee59-ff5e-4481-bf39-da0cbad1c392)

A graph called Pag is then generated, which shows how the prices of each good evolved, where each good is placed on the x-axis and the prices are placed on the y-axis. Each good is represented in a boxplot. Boxplots are used to measure dispersion in a data set, showing the median and quartiles, which are data positioning measures, to know exactly where each of them is located. These graphs also consider atypical data, which are those that are far from the others. In this graph, atypical data are excluded. It can be observed in each boxplot that the dispersion is low due to their size; the interquartile range (the difference between the third and first quartiles) is low, presenting a low dispersion of the central data with a median that is approximately in the center of the box. This shows the reliability of the model and that agents are reaching agreements to arrive at a specific price.

![Image](https://github.com/user-attachments/assets/e031bcf1-22c8-4ba0-9b05-1dcd86a3cb7b)

A graph called Pbar is then generated, which is the union of the medians of each boxplot from the previous graph for each iteration. The axes of the previous graph remain unchanged, but the boxplots are removed, demonstrating that in each iteration, prices remain virtually unchanged or the change is negligible.

![Image](https://github.com/user-attachments/assets/be2f35de-fcb8-4a9a-baeb-e018ac598173)

Finally, a graph called Pbardiffs is created. It randomly selects four iterations from the previous graph to clearly show the evolution without the other iterations interfering with the measurement. The logic is the same as Pbar.

![Image](https://github.com/user-attachments/assets/5c9c6eda-c2f1-4c68-b033-dcabd17aa363)


### Example 2
#### Explanation

Example 2 in the paper "Reaching an Equilibrium of Prices and Holdings of Goods Through Direct Buying and Selling" demonstrates the bilateral trading algorithm in an economy with $n+1 = 3$ goods (including money as good $j = 0$) and $m = 3$ agents. The initial holdings are highly imbalanced, with each good predominantly held by one agent, testing the algorithm's ability to reach equilibrium. The utility functions are of Cobb-Douglas type:

$u_i(x_i) = \prod_{j=0}^{2} x_{ij}^{\beta_{ij}}$, with $0 < \beta_{ij} < 1$, $\sum_{j=0}^{2} \beta_{ij} < 1$,

where $x_{ij}$ is the holding of good $j$ by agent $i$, and $\beta_{ij}$ are utility parameters. Initial holdings $x_{ij}^0$ and parameters $\beta_{ij}$ are given in Table 3, e.g., agent 2 holds most of good 2 ($x_{2,2}^0 = 80$), and agent 3 holds most of good 1 ($x_{3,1}^0 = 80$).

The algorithm (Algorithm 1) facilitates bilateral trades of a single good for money, based on price thresholds:

$p_{ij}(x_i) = \frac{\partial u_i / \partial x_{ij}}{\partial u_i / \partial x_{i0}}$,

A trade occurs when the seller's price satisfies:

$p_{i_1j}^+(x_{i_1}) = p_{i_1j}(x_{i_1}) + \delta_{i_1j}$,

and the condition:

$p_{i_1j}^+(x_{i_1}) \leq p_{i_2j}^-(x_{i_2}) = p_{i_2j}(x_{i_2}) - \delta_{i_2j}$,

where $\delta_{ij}$ is the premium. The traded quantity is:

$\xi_j = \min \{ \xi_j^+(x_{i_1}, \pi_j), \xi_j^-(x_{i_2}, \pi_j) \}$,

where $\xi_j^+(x_{i_1}, \pi_j)$ and $\xi_j^-(x_{i_2}, \pi_j)$ maximize the seller's and buyer's utilities, respectively.

Equilibrium prices $\bar{p}^\nu$ for runs $\nu = 1, 2$ (Table 4) and final holdings $\bar{x}_{ij}^\nu$ (Table 5) are consistent across runs despite the initial imbalance. Compared to Walrasian prices, computed via:

$x_{ij} - \beta_{ij} ( \sum_{j=1}^{n} p_j x_{ij}^0 ) = \beta_{ij} x_{i0}^0$,

$p_j \sum_{i=1}^{m} x_{ij}^0 - \sum_{k=1}^{n} p_k ( \sum_{i=1}^{m} \beta_{ij} x_{ij}^0 ) = \sum_{i=1}^{m} \beta_{ij} x_{i0}^0$,

with:

$\beta_{ij} = \frac{\beta_{ij}}{\sum_{k=0}^{n} \beta_{ik}}$,

$\bar{p}_j^W = \frac{p_j}{p_0}$,

the results highlight the Walrasian model's disconnect from true market dynamics.

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
The provided Python code defines a simulation setup for an economic model by importing the necessary libraries and configuring the settings for visualisation and computation. The BTE module is imported to provide the Economy class and auxiliary functions such as bilateral trading. The copy module's deepcopy function is used to create independent copies of objects. The Pandas library, aliased as 'pd', facilitates exporting data to Excel, while 'gc' optimises memory through garbage collection. The time module measures execution time and handles date formatting, while the os module supports system operations such as folder creation. The matplotlib library, along with mpl_toolkits.mplot3d, enables graph generation; however, 3D plotting is not utilised here. Matplotlib's rc settings configure graph fonts to Helvetica or a serif font such as Times New Roman, and enable LaTeX rendering to produce professional-looking mathematical typography. Redundant plt.rc calls ensure compatibility.

The Ex1 function models an economy with three agents and three goods (one of which is money). It defines a 3×3 'beta' matrix representing the agents' preferences in Cobb–Douglas utility functions, as well as a 3×3 'e' matrix for the initial endowments. An 'Economy' object is instantiated with these parameters, where 'Econ.alpha' sets preferences, 'Econ.e' assigns endowments and 'Econ.allocations' initialises allocations to endowments. The Econ.pag attribute computes agent-specific prices via the evalp method. A 'delta' array, initialised as 10% of 'Econ.pag', is assigned to 'Econ.delta' to adjust trading increments, reducing by 90% when trades cease until they become negligible. The function returns the configured 'Economy' object. Finally, the line of code 'np.random.seed(10)' ensures that agents are paired at random for trading, avoiding sequential ordering.

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
The code continues with the setup of the economic simulation, defining the variables and computations required to analyse an economy with multiple trials. The variable exfig is set to 'pdf' to format the output figures. An economy instance (Ec_aux) is created using the Ex1 function to compute theoretical Walrasian equilibrium prices (p_(we)) and allocations (x_(we)) via the Walraseq method. Retrievable Walrasian prices (p_(wertr)) and allocations (x_(wertr)) are also computed via the Walraseqrtr method. A dictionary called 'Econ' is initialised to store the economy instances for each trial. The number of trials is set to 20 using the ntrials function. Dictionaries EvPag, EvAlloc and EvDelta are created to store prices, allocations and delta values for each simulation. The TimesChange dictionary tracks the number of trades per iteration, while K records the total number of iterations as an integer. The eq_status array indicates whether equilibrium has been reached (1 for convergence, 0 otherwise) for each trial. Execution times are stored in ExTimes and EvSD tracks the maximum standard deviation of prices. The Evpbar array stores the simulated equilibrium prices and the Walras_prices and Walras_alloc arrays store the Walrasian prices and allocations, respectively. The BPT array stores equilibrium price data for non-monetary goods across trials, forming a matrix that can be used to create boxplots to visualise price dispersion. Wealth_(bte) and Utilities_(bte) compute agent wealth and Cobb–Douglas utilities for bilateral trading and compare them with Wealth_(wal) and Utilities_(wal) for the Walrasian equilibrium. The 'Prices_final' array stores the final equilibrium prices for non-monetary goods. An empty dictionary, insp, is initialised for inspection orders. l_order defines six permutations of agent inspection sequences (e.g. [0, 1, 2] to [2, 1, 0]), and g_order specifies normal and reverse goods orders ([0, 1] and [1, 0]). Initial wealth (w0) and utility (u0) arrays are created containing the number one for three agents. A loop iterates over the three agents in Ec_aux, computing the initial prices (p0) by adding 1 (the normalised money price) to each agent's prices (Ec_aux.pag[i,:]). The p0 vector is printed and the initial wealth (w0[i]) is calculated as the dot product of p0 and the agent's endowments (Ec_aux.e[i, :]). Initial utility (u0[i]) is computed as the product of allocations (Ec_aux.allocations[i, :]) raised to preferences (Ec_aux.alpha[i, :]) using the Cobb–Douglas utility function.

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
The Python code carries out an economic simulation by iterating over 20 trials in order to analyse bilateral trading against the Walrasian equilibrium. A loop with index `ku` processes `ntrials` (20). For trials 0–11, the inspection orders are assigned by fixed loops that combine six agent permutations (`l_order`), two goods orders (`g_order`), and are stored in `insp[2*io+go]` as['fixed', agent_order, goods_order]. For trials 12–19, `insp[ku]` is set to['ran', {}, {}]. This is for random selection. The inspection configuration is printed.

A loop over `k` trials prints the trial number, creates an independent economy (`Econ[k]`), via `Ex1()` and computes Walrasian equilibrium prices (`Walras_prices[:, k]`), and allocations (`Walras_alloc[k]`), using `Walraseqrtr()`. Execution time is tracked with t. The bilateral trading algorithm runs with the following parameters: `eps_prices = 1e-6` (price convergence threshold), `MAXIT = 250000` (maximum number of iterations), `lbda = 0.975` (price adjustment factor), `delta_tol = 1e-18` (minimum delta) and `inspection = insp[k]`. The algorithm returns the following: price history (EvPag[k]), allocation history (EvAlloc[k]), delta history (EvDelta[k]), trade counts (TimesChange[k]), iterations (K[k]), and equilibrium status (eq_status[k], 1 for convergence). The execution time (`ExTimes[k]`) is calculated and printed. Equilibrium prices (Evpbar[1:k]) are set as the maximum of Econ[k].pag, and the maximum price standard deviation (EvSD[k]) is computed and printed.

For each trial and agent (`i`), a matrix called `Z` (agents x iterations) keeps track of wealth. For each iteration of `kk` and agent `ii`, the sum of money holdings and the value of non-monetary goods (quantity x price) is calculated by `Z[ii,kk]`. The maximum equilibrium prices are stored in `BPT[k,:]`. Agent wealth (`Wealth_bte[k,i]` ) is the dot product of equilibrium prices and allocations. Utility (`Utilities_bte[k,i]` ) uses the Cobb–Douglas function with preferences (`Econ[k].alpha[i,:]`). Walrasian wealth (`Wealth_wal[k,i]`) and utility (`Utilities_wal[k,i]`) are similar.

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
The Python code visualises and analyses bilateral trading equilibrium (BTE) price thresholds in comparison to Walrasian equilibrium, performing this analysis over 20 trials. It generates four plots and performs a price difference analysis.

A scatter plot created using plt.figure() and plt.scatter() shows the BTE price thresholds for goods 1 and 2 (Evpbar[1,:]) and (Evpbar[2,:]) across trials, with the Walrasian prices (p_wertr[1] and p_wertr[2]) marked in red. Titled 'Final BTE equilibrium price thresholds' in LaTeX, the axes are labelled 'Goods $j$' and '$p$'. The plot is saved as 'Pbar.pdf' and closed.

A boxplot, created using plt.boxplot(BPT, showfliers=False), displays the price threshold distributions for non-monetary goods and demonstrates the robustness of the BTE. Titled 'Boxplot for equilibrium price thresholds' in LaTeX, the axes are labelled 'Goods $j$' and '$p$', and the plot is saved as 'Pag.pdf' and closed.

Edgeworth box plots are generated per trial. A loop defines line styles and colours, computes total supply (Ec_aux.e sum) and plots allocations (EvAlloc[k][kk][0, 1] and EvAlloc[k][kk][0, 2]) for agent 0 across iterations. Annotations mark the initial and final allocations. Titled 'Edgeworth box', the axes are 'Good 1' and 'Good 2', limited to supply. It is saved as 'Edgebox12.pdf' and closed.

A final plot compares trials with significant price differences. Mean prices (pmean) are computed from Evpbar. A lower triangular matrix, diff, stores the maximum absolute price differences between trials. The indices i0 and i1 (for the largest difference) and i2 and i3 (for the next largest difference) are found using the functions np.argmax() and np.unravel_index(). The prices for these trials are plotted, labelled and saved as 'Pbardiffs.pdf' alongside a legend.

The code visualises price dispersion, compares equilibria and analyses convergence using LaTeX-formatted PDF outputs, closing figures to free memory.

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
Remove quotation marks.
The Python code segment finalises the economic simulation by outputting the results, generating an Excel file and creating a visualisation for comparing bilateral trading equilibrium (BTE) and Walrasian equilibrium utilities. For brevity, it is constrained to under 2000 characters.

The code prints indices `i0`, `i1`, `i2`, `i3`, which identify trials with significant price differences. This is followed by a LaTeX-formatted table header (`j&i0&i1&i2&i3&Wal`), which provides information about goods and equilibrium prices. The LaTeX format is used to print rows of goods (`Ec_aux.n`), showing good index n, BTE prices (`Evpbar[n,i0]`, `Evpbar[n,i1]`, `Evpbar[n,i2]`, `Evpbar[n,i3]`), and Walrasian price (`p_we[n]`).

An Excel file, 'Output.xlsx', is created using 'pd.ExcelWriter'. For each trial `k` in `ntrials`, a matrix `XX` (iterations `K[k]` × 11) stores iteration data: iteration number, prices (`EvPag[k][j][0,0]`, etc.), and allocations (`EvAlloc[k][j][0,0]`, etc.). This is converted to a DataFrame (`data_df`), which is then written to a sheet named 'page_k' with 8 decimal places. The Walrasian allocations (`Walras_alloc[0]`), and equilibrium prices (`Evpbar`), are written to separate sheets (`Walras_alloc`, `Evpbar`), with similar precision. The file is saved using the function `writer.save()`.

A scatter plot compares utilities. A range x is created using np.linspace from the minimum to the maximum of Utilities_wal and Utilities_bte (excluding the Walrasian row). For each trial `k`, `plt.scatter` plots `Utilities_wal[k,:]` (x-axis) against `Utilities_bte[k,:]` (y-axis), with the labels showing the trial. A dashed line (`x=x`) is plotted using the plt.plot function. The axes are labelled 'Walras Utility' and 'BTE Utility'. A legend in the bottom right corner tracks the trials. The plot, saved as 'Utilities.pdf', shows BTE efficiency relative to the Walrasian equilibrium, with points above the line indicating surplus utility and points below indicating loss.

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

![Image](https://github.com/user-attachments/assets/7eea5091-10e2-4e83-91b7-526cb54e3d45)

Then the inspection method changes with respect to example 1. Now for each good and agent (3 goods and 3 agents) there are different ways to inspect them. The first 12 iterations are fixed, that is, they follow a pre-established order to find the equilibrium and then the next 8 iterations are random where all agents can trade with anyone freely, without an established order. In this case, in the fixed ways of inspecting they will always go in pairs since inspecting in the order [0,1,2] is the same as [0,1,2] only the order in which non-monetary goods are inspected changes and this does not generate differences in the final result of the equilibrium. 0 is good 2 and 1 is good 1 in the case of the inspection order of non-monetary goods. Trade opportunities are generated by inspecting each of the agents in the established order and seeing with whom they can have a trade opportunity. In the random mode (set by ran in the output), agents are free to trade with the first person they have the opportunity to do so and jointly earn utility. The code after the first price matrix generates a dictionary with everything mentioned above to maintain order.

![Image](https://github.com/user-attachments/assets/359e71eb-7e56-4148-b4dd-be369ebb8869)

The same thing happens next as in the previous code, except that it specifies the method for inspecting each trade and how this will be carried out. It specifies the standard deviation of prices that are below the tolerance established in the code, which is correct. Also, since it includes the pyutilib library, it specifies the time it took the algorithm to reach equilibrium, which allows us to draw the same conclusions as in Example 1.

As before, it also generates the number of balances that were successfully reached, in this case all of them (20/20). It also generates the median of the number of iterations that the inspections took to reach the balance, which does not exceed 1030 iterations, and also the median of the times that the inspections took to reach the same balance, which does not exceed 5 seconds.

![Image](https://github.com/user-attachments/assets/4238c493-cfc8-4f57-bde1-63695430ec68)

Two matrices of 20 rows and 20 columns are then generated, each representing the number of inspections performed and carried out. The BTE algorithm is run on the first matrix, noting that the first two rows are zeros, and then the third and fourth rows begin to show values. This means that no way was found to improve prices between agents for the first two ways of inspecting (i.e. "fixed, [0,1,2], [0,1] or [1,0]), then with the third way of inspecting the combination of agents with this pre-established order was found to improve the prices of the goods, and as the third way of inspecting is equal to the fourth, it has the same improvement value which in this case is percentage as in example 1. This is so until row 13 where inspections begin randomly. Here you can see that inspections are no longer done in pairs but 1 by 1 (exceptionally row 13 does it in pairs to then complete the matrix) finding more ways to make trades and not with a pre-established order, but randomly, making things easier for the bilateral algorithm, making an improvement in prices in a more progressive and sustained way. Then again another matrix is ​​created of 20 x 20 but this time with the Walras algorithm and you can see that with this inspection method things become much more difficult for the reasons explained above and it is more difficult to find a balance having some row values ​​at 0.

![Image](https://github.com/user-attachments/assets/59f2730a-19a2-4bd0-8218-46e692ecfe46)
![Image](https://github.com/user-attachments/assets/aab3720c-fa96-43c6-a319-2c36fd13be4d)
![Image](https://github.com/user-attachments/assets/b82d0f52-622d-47eb-bbf5-6d875642cbba)
![Image](https://github.com/user-attachments/assets/6584634e-59f6-4f6a-bc84-4b5768d4441c)
![Image](https://github.com/user-attachments/assets/da8324e5-65ef-48bc-abec-7844469d632c)

Then, as before, a price comparison matrix is ​​generated between Walras and BTE. This time, the inspection method is chosen equally at random, that is, two fixed iterations and two random iterations. This allows for a comparison between these two methods, not only comparing the algorithms but also the inspections. A significant difference can be seen between the inspection methods and their prices with Walras. First, the fixed inspection method yields significantly higher prices than the random inspection method, with differences not only for good 1 but also for good 2. This may be due to the restrictive nature of the fixed inspection method imposed on the BTE algorithm. This way, it cannot find the best way to combine agents to optimally improve their utilities, but rather in a forced manner, thus raising prices more indiscriminately. In contrast, random arrangements show that prices rise in a more controlled manner for each agent. Both inspections are far from resembling Walrasian prices.

![Image](https://github.com/user-attachments/assets/c8e6137f-2093-4d8a-8edc-932275aeaac5)

At the end, place a price vector that represents the final vector of the value of the prices of each of the non-monetary goods in Walras.

![Image](https://github.com/user-attachments/assets/0766ff8a-efb1-4ce1-b2d9-b9820e8e2449)

We now begin by looking at the graph outputs for example 2.

One of the graphs generated, called Utility, is a graph that shows the evolution of agents' utilities in the different methods, both in "fixed" and in "ran". Comparing not only these inspection methods, but also comparing them with the Walras utilities for each inspection, the X axis being the Walras utilities and the Y axis the BTE utilities. Then, an x ​​= y function is plotted to measure the efficiency of the utility, that is, which gives a higher value in utility, if BTE or Walras, as mentioned for example 1. The points in the lower left side represent agent 1 (those above the random inspection and those below the fixed inspection). It can be observed how the utility of agent 1 is better for random inspections, having a higher utility than fixed inspections. Then the points furthest to the right are the utilities of agent 2 (the other way around, just like the next agent, with random inspection at the bottom and fixed inspection at the top). You can see here that random inspections are worse in performance than fixed inspections, where the utilities of agent 2 and 3 are very similar (agent 3, the last points on the right). This is because in fixed inspections it is easier for those with larger positions at the beginning to want to trade among themselves to improve their utilities in a more controlled manner, since agent 1 has much lower utilities with respect to their initial endowments, making it inconvenient for agents 2 and 3 to trade with the latter because their utilities would have to go down and this is not the idea for the BTE algorithm or it would have to be a very slow growth with respect to agents 2 and 3 who have the majority of the endowments of goods 1 and 2. Also, the way of inspecting can greatly affect this value, since the true optimal growths for one agent and another for utility are not found. For this reason, random inspections are better than fixed ones, seeing the example of the graph where the points of agent 1 equal the utilities of agents 2 and 3, having a balance in the utilities of each agent.

![Image](https://github.com/user-attachments/assets/d002e05f-697a-43a2-a04b-9d23a1c4336e)

A graph called Utilities_bte is then generated, which is made up of stacked bars, each part of which is color-coded to distinguish the three agents. This graph compares the utilities of each agent across different inspections. The y-axis represents the utility, and the x-axis represents the 20 inspections (12 fixed and 8 random). The last bar in the graph represents the Walras utilities, where the utility equality between agents 2 and 3, who have the largest initial allocations, can be observed. The huge difference with the utilities of agent 1 can also be seen (for clarity, agent 1 = blue, agent 2 = orange, and agent 3 = green). Here we can see the effectiveness of the "Ran" method, which helps the BTE algorithm find better balances and synchronously improve the utilities of each of its agents, unlike the "fixed" method, which causes the utilities of agents 2 and 3 to skyrocket and the utilities of agent 1 to plummet. The Ran inspection method is the best for equalizing the utilities of each of the agents, confirming the previous graph.

![Image](https://github.com/user-attachments/assets/f1013f7d-35f6-496d-8904-424c997bf399)

Next, there's a graph similar to the one before, except it shows the agents' wealth for each type of inspection, plotting only the utilities for each agent's wealth using the same colors as before on the y-axis. At the end, a stacked bar representing each agent's Walras wealth is placed for comparison. It's clear that the wealth of agent 1 is greater than that of agents 2 and 3 in both fixed and random inspections. This is because, in trying to equalize their positions with agents 2 and 3, agent 1 tries to improve their positions as quickly as possible to equalize the positions of agents 2 and 3. Since fixed inspections are more complex, they have to improve their positions quite a bit in a small pair of iterations, thus improving their wealth more than agents 2 and 3, who are already balanced in terms of holdings of good 1 and good 2. This improves their wealth among themselves in a more controlled and less abrupt manner. It can also be observed that the wealth in random inspections for agent 1 is much greater than in those of 2 and 3, thus improving their situation and balancing their holdings with agents 2 and 3. Here we can see how random inspections with these three graphs cause the BTE algorithm to have a more balanced growth among all agents in both utilities and wealth. The fixed inspection method has many restrictions, which does not allow the BTE algorithm to act freely so that all agents improve their utility jointly. Finally, we can see the great inequality between the wealth of agents 2 and 3 and that of agent 1.

![Image](https://github.com/user-attachments/assets/0ceddcb9-8600-4d6e-bdcb-bbc8629c536f)

Then a social utility graph is generated, which is the sum of the total utilities of all agents (from the Utility_bte graph the full bar. For example, for inspection 0 the sum of all the utilities of all agents is 80). In this graph it can be observed that the Walras social utility is always constant at 100, while the BTE utility is changing and at the beginning has a large value, which would be for the values ​​of the first inspections of the algorithm (fixed), where the utilities of agent 2 and 3 shoot up with respect to the utilities of 1, the utilities being uneven, where the holdings are concentrated in only 2 agents instead of all. Then, when we switch to the random inspection method, we can see that utilities drop significantly, indicating that agent 1, through the bilateral exercise, managed to equalize his holdings with agents 2 and 3, equalizing everyone's utilities and making the distribution of goods better for everyone. This is costly for utility, but equalizing agents 2 and 3. This can be seen in two ways. The increase in social utility is good since there is a society that is satisfied with the goods it has and they satisfy its preferences, but when those goods are concentrated in the hands of a few and there are a large number of agents or people who do not have the same goods, that value is just a facade, a lie hidden under numbers. Social utility worsens, not because those with greater positions produce less, but because agents with smaller positions have fewer restrictions on making their trades and producing in such a way as to equalize the positions of those who have more, having a lower Gini coefficient, and resulting in a more egalitarian society.

![Image](https://github.com/user-attachments/assets/f47c6d7c-1764-448a-9445-4dedd06dbe2d)

After this, the same graph generated for exercise 1, called Pag, is generated. This graph is the same as the previous one, only for two goods. The difference in prices can be seen in the boxplots, and how they vary much more violently than in the previous exercise. The difference in inspection methods greatly favors this, with a certain number of prices for fixed inspections and other prices for random inspections. These two methods, as mentioned above, significantly change the way the BTE algorithm operates. Thus, you can see how the box is quite elongated, indicating a large dispersion of the data between the prices at the bottom and those at the top of the boxplot. You can see how the median is closer to the third quartile, indicating a large dispersion of the data in the first section of the box (for both good 1 and good 2). In this way, the largest values ​​are close to the median, where the data are most concentrated, and the smallest values ​​are farther away and more dispersed. This indicates how different the inspection models are from one model to another, with smaller and more dispersed prices for fixed inspections and, conversely, larger values ​​for random inspections, especially for Agent 1.

![Image](https://github.com/user-attachments/assets/a68da7ea-bafe-4172-9500-bf6a9eee9530)

Then another graph is created called Pbar, which is different from the Pbar graph in example 1. Here you can see that the X axis represents the goods and the Y axis the prices. You can see different points distributed around the graph. You can see that the first points that are in blue in the lower left side of the graph are the representation of the random inspection prices. Then there is another set of points that are between 0.1 and 0.2 of each axis that are the prices of fixed inspections, demonstrating that they are better than those of random inspections. This is not necessarily better since it shows that there is a greater inequality between agents, thus not reaching equilibrium prices. Finally, there is a last red point, which is the Walrasian prices. The points that are closer to the theoretical equilibrium have a greater inequality between their utilities, as seen previously. Therefore, fixed inspections are worse than random inspections, which demonstrate that they have a more controlled growth in prices.

![Image](https://github.com/user-attachments/assets/6d5cabee-7847-4f2c-b65d-8fe2f47ab05e)

Then another graph is created called Pbardiffs which is similar to the one in example 1. This graph shows that in random inspections there is greater equality of prices between agents and with fixed inspections, they are greater but have a greater dispersion, with unequal growth between agents for the reasons explained above.

![Image](https://github.com/user-attachments/assets/1c382f33-3627-4556-86d2-08d5894d25ac)

Finally, a new type of graph is created that was not present in the previous example: the edgebox. These edgeboxes plot how each good evolved in relation to the other, starting with the agent with the lowest number of positions (in this case, 1). It can be observed how both positions progressively and controlledly improve their assets 1 and 2 to reach equilibrium with the other agents, in this case, agents 2 and 3, who have the largest number of positions of assets 1 and 2. It can also be seen that with the fixed inspection methods, agent 1 did not reach equilibrium, but with the random method, they perfectly reached equilibrium with the assets of agents 2 and 3. This edgebox confirms the effectiveness of the random inspection method over the fixed one. The same applies to the edgebox that compares monetary positions with assets 1 and 2; they do not reach equilibrium with the fixed inspection method, but they do with the random method.

![Image](https://github.com/user-attachments/assets/191657d9-382d-4d1e-9b60-49c76074b5d1)

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
The Python code implements the core algorithm from section 5 of the paper, supporting sections 3 and 4, and defines the 'Bilateral' method. It simulates bilateral trading to reach equilibrium using the following parameters: `eps_prices` (price convergence tolerance), `MAXIT` (max iterations), `lbda` (delta reduction factor), `delta_tol` (delta tolerance) and `inspection` (agent/goods selection strategy). Dictionaries `EvPag`, `EvAlloc`, and `EvDelta` store price, allocation, and delta histories, with `TimesChange` tracking trades, `KK` counting iterations, and `eq_status` indicating equilibrium (1 if reached, 0 otherwise). Initial prices, allocations and deltas are copied.

The algorithm iterates up to `MAXIT`, terminating if price standard deviation falls below `eps_prices` (equilibrium reached) or maximum delta is below `delta_tol` (negligible trades). Random (`ran`) or sequential (`det`) inspection is used to select agents and goods. For each agent pair (`i1`, `i2`), and good (`j`), it computes a trade price (`pit`), which is an intermediate between the seller's (`pip`) and buyer's (`pin`) prices, and is adjusted by deltas. The trade quantities (`xij`), representing the minimum of the buyer's (`xin`) and seller's (`xip`) offers, are important here. Trades occur if the prices align, there is sufficient stock, and the quantities are viable. This updates the allocations and recalculates the prices via the `evalp` function.

If no trades occur, the deltas will reduce by `lbda`. Trade counts are recorded in 'TimesChange'. A warning is printed if `MAXIT-1` is reached without equilibrium. Histories are updated each iteration, and the method returns `EvPag`, `EvAlloc`, `EvDelta`, `TimesChange`, `KK`, and `eq_status` for analysis.

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
The Python code defines the `Walrascheck` method, which verifies if input prices `p` form a Walrasian equilibrium. It initialises an allocation matrix, `x`, matching the endowments (`self.e`), using `np.zeros_like`. A loop then iterates over the agents (`self.I`), creating a Pyomo optimisation model (`ConcreteModel`). This model defines a set of goods (`model.n`), which is set via the range function (`range(self.n)`). The preferences of the agent (`self.alpha[ii]`), which are set via `alpha_init` and stored in `model.alpha`, are of particular interest. The linear coefficients (`model.a`), and the quadratic coefficients (`model.b`), for the utility function are initialised by setting money (good 0) to 0, and the others to `self.a[ii,j-1]` and `self.b[ii,j-1]`. Initial endowments (`model.e`), and prices (`model.p`), are set, with money's price normalized to 1 and others from `p[j-1]`. The decision variables (`model.x`), which represent allocations, have the following bounds: money is strictly positive (`1e-20, None`); the others are non-negative (`0.0, None`). They are initialised with the current allocations (`self.allocations[ii, j]`).

The objective function (`obj_rule`), which maximises utility, combines a Cobb–Douglas term for money (`(model.x[0])**model.alpha`), and a quadratic term for other goods (`sum(model.a[j]*model.x[j] - 0.5*model.b[j]*model.x[j]**2`). The budget constraint (`bc_rule` ) ensures that expenditure (`sum(model.p[j]*(model.x[j]-model.e[j])`) is non-positive. The IPOPT solver (`SolverFactory('ipopt')`) is used to optimize the model, and the results are stored in `x[ii,j]`. The aggregate excess supply (`ES`), meanwhile, is computed as follows: `np.sum(self.e-x, axis=0)`. If the maximum absolute excess (`np.max(np.abs(ES))`) is less than 1% of the total supply (`0.01*np.max(np.sum(self.e, axis=0))`), then prices are said to form a Walrasian equilibrium. Otherwise, this is not the case. The method returns `ES` (excess supply) and `x` (optimal allocations).

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
The Python code defines two methods, 'Walraseq' and 'Walrasdyn', which compute Walrasian equilibrium prices and allocations algebraically and dynamically, respectively, as described in Section 5 of the paper.
The `Walraseq` method initialises an allocation matrix `x` that matches the `self.allocations` variable, sets the initial prices to 1, and computes the matrix `B` (the sum of the preferences of the agents, `self.alpha`), and the matrix `E` (the aggregate endowments of the agents, `self.e`), per good. A matrix, `A`, and a vector, `b`, are constructed to solve the linear Walrasian system. For each good `j` and `l`, `A[j,l]` is set as follows: on the diagonal (`j==l`), it's `E[j] - sum(self.alpha[:,j]*self.e[:,l]/B[:])`; off-diagonal, it's `-sum(self.alpha[:,j]*self.e[:,l]/B[:])`. Vector `b[j]` is `sum(self.alpha[:,j]*self.e[:,0]/B[:])`. Prices are solved using `np.linalg.solve(A, b)` and normalised by `p[0]`. A loop over agents computes allocations `x[i,j]` as `ul*self.alpha[i,j]/p[j]`, where `ul` is the agent's income (`sum(self.allocations[i,:]*p)/B[i]`). The method returns equilibrium prices `p` and allocations `x`.
'Walrasdyn' performs dynamic price adjustment (tatonnement). Initial prices `p0` are copied to `p`, excess supply `ES` and allocations `x` are computed via `Walrascheck(p)`, and `ld=0.01` (price adjustment rate) and `MAXIT=1000` (maximum iterations) are set. A loop iterates up to `MAXIT`, with `k` being the variable in question. If the minimum excess supply (`np.min(ES)` ) is at least `-1e-3`, the program prints 'Walras equilibrium' and returns `p`. Otherwise, prices are adjusted (`p + ld*ES`), and both `ES` and `x` are recalculated. The printing of a warning and the returning of the final 'p' is the action taken by the system if 'MAXIT' is reached.

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
The Python code imports the necessary libraries and configures the settings for the economic simulation. It then defines a utility function for graph visualisation. It imports BTEQuad for economy-related classes and functions, Pandas (as 'pd') for exporting Excel data, and GC for optimising memory. The time module tracks execution time and date formatting, while os handles system operations such as folder creation. Matplotlib.pyplot and mpl_toolkits.mplot3d are used for plotting, although 3D graphics are not utilised. The graph is styled using rc('font', family='sans-serif', sans-serif=['Helvetica']) and rc('text', usetex=True), which use Helvetica and LaTeX for professional typography. This is reinforced by plt.rc('text', usetex=True) and plt.rc('font', family='serif'), which use serif fonts such as Times New Roman.

The makeArrow function draws arrows on plots. It takes the following parameters: ax (plot axis), pos (starting position), function (to compute y-values) and direction (arrow orientation). A delta variable is set to 0.0001 if direction ≥ 0 and to -0.0001 otherwise, determining the arrow’s direction. The ax.arrow method then draws an arrow from (pos, function(pos)) to (pos + delta, function(pos + delta)), setting the arrowhead size with head_width = 0.05 and head_length = 0.1.
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
The code establishes the primary execution block for an economic simulation, initialising the variables required to analyse bilateral trading equilibrium (BTE) in relation to the Walrasian equilibrium. The 'if __name__ == "__main__":' condition ensures that the script only runs when executed directly. The `np.random.seed(10)` sets a fixed seed for reproducible random agent pairings, which is important for reproducibility and repeatability in the system. The variable exfig='pdf' specifies PDF format for figures. The test economy, labelled `Ec_aux`, is created via the function `Ex2()`. Dictionaries `Econ`, `ES`, and `xopt` store economy instances, equilibrium states, and optimal allocations per trial. The number of trials is set to two (`ntrials=2`). Dictionaries `EvPag`, `EvAlloc`, and `EvDelta` store price, allocation, and delta histories. The arrays `K`, `eq_status`, `ExTimes`, and `EvSD` track iteration counts, equilibrium status (1 for convergence, 0 otherwise), execution times, and maximum price standard deviations, respectively. The functions `Evpbar` and `Walras_prices` are used to store equilibrium and Walrasian prices, while the function `Walras_alloc` is used to store Walrasian allocations. `Col_c` defines plot colours (`['b', 'r', 'g']`). 'BPT' stores the price thresholds for non-monetary goods in a 2x3 matrix for visualising price dispersion in a boxplot. The arrays `Wealth_bte` and `Utilities_bte` compute agents' wealth and Cobb-Douglas utilities for BTE, while the corresponding Walrasian equilibrium utilities are computed by the arrays `Wealth_wal` and `Utilities_wal`. Initial prices `p0` are set to ones, with `p0[1:]` updated to the maximum of `Ec_aux.pag` for non-monetary goods.

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
The code iterates over `ntrials` simulations, indexed by `k`, to analyse bilateral trading equilibrium (BTE) against the Walrasian equilibrium. Each trial prints its number. It also creates an economy `Econ[k]` via `Ex2()`. Endowments are adjusted, with `Econ[k].e[0,2]` decreasing by `6.0*k` for agent 1 and `Econ[k].e[1,2]` increasing by `6.0*k` for agent 2. Allocations are set to endowments (`Econ[k].allocations`), and price thresholds are computed with `evalp` (`Econ[k].pag`). Walrasian allocations are initialised as zeros (`Walras_alloc[k]`). Execution time is tracked with the command `t`. The Bilateral method runs with the following parameters: The parameters are as follows:
-	`eps_prices=1e-4` (price convergence threshold)
-	`MAXIT=1000` (maximum iterations)
-	`lbda=0.5` (delta reduction)
-	`delta_tol=1e-18` (delta tolerance)
-	`inspection='det'` (sequential pairing) The function returns price (`EvPag[k]`), allocation (`EvAlloc[k]`), delta (`EvDelta[k]`), trade counts (`TimesChange[k]`), iterations (`K[k]`), and equilibrium status (`eq_status[k]`). The execution time (`ExTimes[k]`), equilibrium prices (`Evpbar[1:, k]`), and maximum price standard deviation (`EvSD[k]`) are printed. Walrasian equilibrium is checked using Walrascheck, which stores excess supply (`ES[k]`) and optimal allocations (`xopt[k]`).

For each trial and agent (`i`), a matrix called `Z` (agents x iterations) keeps track of wealth. It does this by adding up money (`EvAlloc[k][kk][ii,0]`) and the value of non-monetary goods (`EvAlloc[k][kk][ii,nn+1] * EvPag[k][kk][ii,nn]`). A plot is produced for each trial showing wealth evolution per agent. This is titled in LaTeX and the axes are labelled 'Iteration' and '$b$' (wealth). The plot is saved as 'Wealth_trial<k>.pdf' and then closed. The maximum prices (`BPT[k,:]`) are stored. Wealth (`Wealth_bte[k,i]`) and utilities (`Utilities_bte[k,i]`) for BTE are computed using allocations, prices and a quadratic utility function. Walrasian wealth (`Wealth_wal[k,i]`) and utilities (`Utilities_wal[k,i]`) are calculated in a similar way. Memory is freed with the function call `gc.collect()`.
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
The code defines the visualisation settings and generates plots for an economic simulation that compares bilateral trading equilibrium (BTE) across trials. The line styles (solid, dotted, dashed) are set by the `lnst` variable, which is repeated for `ntrials`. The function of the `cols` variable is to define a normalized RGB color palette for the purpose of distinguishing plots. A step size of `delta=0.0025` is set for coordinate meshing. Total supply `s` is computed as the sum of endowments (`Ec_aux.e`). This is done per good. The ranges `x0` and `x1` go from 0 to supplies of goods 1 and 2, with a step size of `delta`. This forms a 2D mesh (`X0`, `X1`, etc.) via the function `np.meshgrid`. A figure and axis are initialised with `plt.subplots` for an Edgeworth box plot. The calculation of utilities for agents 1 (`Z1`) and 2 (`Z2`) is done using Cobb-Douglas for money (`Ec_aux.e[i,0] ** Ec_aux.alpha[i]`) and quadratic terms for goods 1 and 2 (`Ec_aux.a[i,j]*X - 0.5*Ec_aux.b[i,j]*X**2`). For each trial `k`, arrays `XX` and `YY` store allocations of goods 1 and 2 for agent 1 across iterations `kk`. These are plotted with `plt.plot`, labelled by trial, with linewidth 1.5, varying styles (`lnst[k]`), and normalized colours. Annotations mark the initial (`$x_0^k$`) and final (`$x_k$`) allocations with dynamic offsets. Arrows are drawn at the interval of `K[k]//15` using the function `plt.arrow`, with the size and colour of the heads configured. Labels for agents and trials are added with the function `plt.annotate`. The axes are labelled 'Good 1' and 'Good 2', and the plot is titled 'Edgeworth box, goods 1 and 2' in LaTeX. The limits are fixed to xlim: 0, 3 and ylim: 1, 9. The plot is saved as "Edgebox12.pdf" with `plt.tight_layout` and closed after a light gray background and dashed grid are set. The user creates a scatter plot with `plt.figure`, plotting equilibrium prices (`Evpbar[1,k]`, `Evpbar[2,k]` ) per trial, which they label. The axes are labelled "p_1" and "p_2", and a frameless legend is added to the top left. The plot is saved as 'Pbar.pdf' and closed.

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
The code creates a figure (`fig`, `ax`) for plotting price thresholds. The arrays `XX` and `YY` are initialised to store the prices of goods 1 and 2 (`EvPag[k][kk][:0]`, `EvPag[k][kk][:1]`), respectively, across iterations `kk`, by a loop over `ntrials` (index `k`). For each agent `i`, prices are plotted (`XX[i,:], YY[i,:]`), with a label, linewidth 1, style (`lnst[k]`), and a normalized colour. Arrows are drawn at intervals of `5*(1+5*k)` from `kd=1` to `K[k]-150*(1+2*k)`, with a head size of 0.1 and a matching colour. Initial prices are annotated in LaTeX (`$p^{0,k}_{i+1}$`) at offset coordinates and marked with circles. If equilibrium is reached (`eq_status[k] = 1`), the final price for agent 0 is marked with a plus and annotated (`${\bar p}^k$`). Otherwise, the final prices are marked with an 'x' and annotated as `${\bar p}^k$`. Labels for trials 0 and 1 (agents 1 and 2) are added at specific coordinates. The plot is titled in LaTeX and the axes are labelled 'Price threshold for good l=1 ($p_1$)', 'p₂', and are limited to `xlim(16, 24)` and `ylim(12, 24)`. A light grey background, dashed grid and tight layout are applied. The plot is saved as 'Evbarp.pdf' and closed.

An Excel file, 'Evs.xlsx', is created using `pd.ExcelWriter`. For each trial `k`, a matrix `XX` (iterations × 11) stores the iteration number, prices and allocations for agents 1 and 2. Data is written to sheets (`page_k`) with eight decimal precision and saved. A LaTeX table matrix `Tab7` (4 rows and 4 columns) stores the initial and final prices of goods 1 and 2 for each agent and trial. Labels and indices are set for the LaTeX output and the file is saved as 'Tab7_results.tex' with four-decimal formatting.

#### Installing libraries and configurations to run the code

For this example, you don't need to install any additional libraries; they're all already installed, and the Python 3.11 environment is ready to use. Go directly to the outputs.

#### Outputs 

First, as in the previous two examples, it generates written outputs within the same console. First, the code prints the iteration number it is in. It first prints iteration 0 with the agents' main allocations before making the trades. It can now be seen how one of the agents starts with non-monetary allocations at 0, which in this case is agent 1 for good 1, and agent 2 starts with allocations of both goods 1 and 2 but with little money (since it cannot be 0, since the marginal utility function of money tends to infinity when money is 0). This is the first time that a problem of this type has been faced, complicating the improvement of agents' utilities since agent 2 does not have such basic allocations as money, which leaves him with the only option left: to sell to improve his utilities, this being his incentive. On the other hand, agent 1 lacks good 1, therefore, what remains for this agent is to buy the good that agent 2 lacks in order to jointly improve utilities. Therefore, in this example, the equations of the bilateral exchanges from section 2 of the paper, specifically 2.14 and 2.20, can be better seen. After generating this matrix, the times it took for the trades between the agents to reach equilibrium are generated. In this case, in trade 0, it took 0.54 seconds to find equilibrium, also placing the maximum tolerance for the standard deviation of prices, giving a lower and achieving a first equilibrium. Unlike the other exercises, this one has the possibility of seeing whether it is a Walrasian equilibrium or not, since it is not as before and utilities and prices have other ways of being calculated, no longer through a Cobb-Douglas utility function. This new function with which utility is calculated is strongly concave, allowing to reach an equilibrium, having a second derivative less than 0, ensuring that in at least 1 iteration BTE will reach equilibrium.

![Image](https://github.com/user-attachments/assets/e936bb2b-aaac-4f9a-ad40-489baae7df9a)

After this, we move on to trial 1, which would actually be the second iteration. Here, the initial goods matrix is ​​the same, except that the quantities of good 2 are exchanged between agents. The code, as explained in the documentation, allows a maximum of 1,000 iterations before reaching equilibrium. In this case, it fails to reach equilibrium before 1,000 iterations, forcing the BTE algorithm to cut off and declare that equilibrium could not be reached. It then enters the prices at which the algorithm remained before reaching a possible equilibrium. It then generates another matrix, which is the delta of the prices where the algorithm remained, which did not reach the maximum tolerance required by the exercise, which was 10e-18. The exercise ended up at approximately 9e-09, which is quite far off.

![Image](https://github.com/user-attachments/assets/40e5f809-f6cc-4c1a-967f-2278f2562321)

Then, the median of the iterations where equilibrium was reached, the median of the times where equilibrium was reached, and the number of tests that reached equilibrium are generated, which in this case were 1 out of 2. This is generated because the algorithm reached the maximum number of iterations to reach equilibrium, having very demanding tolerances so that prices go down and find equilibrium.

![Image](https://github.com/user-attachments/assets/56d990ed-58a6-4147-9cee-2d8da12223ba)

Now we will proceed to the graphic outputs that the code launches.

First, a graph called Pag is generated, which is the same as the two previous examples. A boxplot shows the price dispersion in each iteration. It can be seen that the prices are well balanced. The median is in the center, generating a low dispersion in the data for both goods.

![Image](https://github.com/user-attachments/assets/fde5e836-0c08-4f0a-824c-92e0821fabbc)

Now, a graph named Evbarp is generated that shows the prices at which equilibrium is reached in trial 0 for both agent 1 and agent 2, showing how the prices for each good evolved for each person in each iteration. The graph shows that equilibrium was reached at the approximate value of (19,18) for the price of good 1 and 2 respectively in iteration 0. This is the equilibrium prices for the holdings of agents 1 and 2. It can also be seen that iteration 2 (1 according to the exercise) does not converge as stated above. The lines are cut and a cross indicates where the equilibrium was truncated. The matrix mentioned in the written outputs indicates the prices where the exercise was truncated, without reaching equilibrium. The direction taken by the price lines is subject to the tolerance established in the exercise and how it goes down until reaching the maximum tolerance of deltas until reaching equilibrium. Where agent 1 is the buyer and agent 2 is the seller.

![Image](https://github.com/user-attachments/assets/3eb172d9-4723-486e-bee4-eeb4c4b017f5)

A graph called Edgebox12 is then generated, which deals with the same idea as the previous exercise: the agents balance their holdings until they reach a maximum level of well-being for both. You can see here how agents 1 and 2 in iteration 0 balance their non-monetary positions until they reach the optimum, where they cannot be improved further, remembering that there is no Cobb-Douglas profit. Then, you can see iteration 1 below, which ends without converging, returning to 0, as confirmed by the exercise.

![Image](https://github.com/user-attachments/assets/2883f7d0-8884-405a-922f-71ed8b626582)

Next, a graph called Pbar is created, which is distinct from the other Pbars generated by the other exercises. It represents the prices where equilibrium is not reached in iteration 2, specifically for agent 1, the equilibrium prices reached before the algorithm was truncated due to reaching the maximum allowed iterations. Agent 2 does not appear because it is not within the graph's price ranges. This could be changed to show where agent 2's price is truncated. It also already appears in the Evbarp graph, just to confirm.

![Image](https://github.com/user-attachments/assets/5eaae891-8f79-4bff-a432-57be03e89f7a)

Finally, wealth evolution graphs are created for iterations 1 and 2. Wealth is calculated using the formula in the graph's title, where the first x in the sum represents the initial monetary endowment, followed by the utilities generated by the BTE equilibrium. In this way, it can be observed that both curves stabilize at a certain point, achieving a wealth equilibrium in iteration 0. Not mentioned previously, but the x-axis represents the iterations that have passed in the exercise, and the y-axis is a factor b that accompanies the quadratic variable in the utility function, confirming that the function is concave and reaches an equilibrium, which is doubly differentiable. As the value of b increases, the limit of the quadratic function becomes smaller, which is not good, so after a certain point, the value drops from 170 to 150, so as not to lose the opportunity to obtain maximum utility. Then, for the second graph of the second iteration, it can be seen that an equilibrium is not reached and that the value of b is becoming increasingly larger, affecting the marginality of the problem and reducing utility. The wealth of one increases, and that of the other remains constant over the iterations. One has excessive growth (agent 2) and the other remains constant (agent 1), where an equilibrium is not reached.

![Image](https://github.com/user-attachments/assets/6fe11806-91a6-4c60-a8d3-5d0f0c8d4500)

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





















