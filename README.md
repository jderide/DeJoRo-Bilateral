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
The main algorithm of the paper in section 5 is implemented, which also explains sections 3 and 4 of the paper, where the first four objects are defined:
- eps_prices: equilibrium tolerance (epsilon p in section 5)
- MAXIT: maximum number of iterations
- lbda: delta reduction factor (lambda in section 5 of the paper)
- delta_tol: tolerance for delta
- inspection: inspection strategy
After this, structures are generated to store the price history, assignments, deltas, and time. Theorem 1 of the paper is then applied, where all agents must reach an agreement on prices, setting a maximum tolerance of the epsilon standard deviation that prices must fluctuate to close trades and ending the iterations by reaching a price equilibrium with a maximum tolerance given by eps_prices. After the prices are evaluated, the premiums must become increasingly smaller so that each trade tends toward a convergence between prices. This is what delta_tol is for, which is a maximum tolerance for deltas once they are very small, meaning that equilibrium is being reached. If equilibrium is not reached using eps_prices and delta_tol, the following lines of code, if and elif, are activated. The first three lines of code with the if and random are to make combinations between agents randomly, so that all agents interact with each other and none are left out, so that it is not a predictable trade and no agents or goods are left out (explained on page 20 of the paper). Then there is the deterministic mode that makes combinations in sequential order, instead of random, removing trade opportunities between agents and goods, not obeying a random exchange. Finally there is the else method, which is to support the two previous methods, more similar to the sequential method than the random one. Likewise, the method that will be used the most is the random one, since it does not leave out any trade opportunities, since the sequential method leaves trade opportunities out by placing pairs of agents in order in each iteration, therefore evaluating each good in order agent by agent until an optimal exchange opportunity is found. Furthermore, its execution takes much longer when done in order, with random inspection being much more optimal and faster, except when a solution is being reached; in that case, the sequential method is better, since it performs an exhaustive search for the solution, case by case, so as not to exceed the given tolerances. After all the possible inspection methods, a negotiation parameter is placed, equivalent to pi, which appears in section 3.2 of the paper, which is an intermediate price between sellers and buyers, to reach an agreement between them. i1 is placed as selling agents, i2 as buying agents, and j as the goods traded in the market. This is where the possible combinations are evaluated with the necessary inspection method (random or sequential). It imposes the restriction that an agent cannot transact with themselves. Then, with pip and pin, or the selling and buying prices respectively, they reach an equilibrium with the delta premiums once transactions can no longer be made, activating the restrictions of delta_tol and eps_price. There is also pit, which is the equilibrium price at which agents decide to trade their goods between themselves, where l_aux, or the intermediate price pit, is activated, between two prices to reach equilibrium with one. In this way, there are three ways to reach an agreement on a price: the buyer proposes it, the seller proposes it, or both prices are averaged using the intermediate price and exchanged at that price. The pip is the minimum price at which agent i1 sells their products, while the pin is the maximum price at which agent i2 buys. Then there are the maximum quantities that both parties are willing to exchange. The xin, in the case of xshort, is the maximum quantities that the buyer is willing to buy, redundantly, while xip in xlong is the maximum quantity that the seller is willing to sell. Both, according to their preferences, and then, within an object called xij, store the minimum quantity to be exchanged by both parties. After this, it is verified that the buyer's prices are higher than the seller's prices, in addition to always ensuring that the seller has enough stock to sell and also ensuring that the trades are economically viable and that the quantities sold are not small (which is explained in part 3.4 of the paper). Then, in the allocations, trades are carried out using the restrictions imposed previously. Seller i1 receives the money using xij*pij and transfers their positions using -xij, the reverse for buyers. Finally, the threshold prices are saved in self.pag, evaluating new broker holdings and updating this value iteration after iteration.
Continuing with the last block of code, if no possible trades are found using the trade_aux==0 function, the delta premium is decreased using lambda to find new trade possibilities, increasing the number of acceptable prices for the next iteration. The else function then stores the number of successfully executed trades using TimesChange, which, with KK, stores the iteration number up to the point where the transaction was completed. The total number of iterations is then saved in trade_aux, which can be used for convergence analysis. Next comes the if statement, which is a warning if equilibrium was not reached with the maximum number of iterations using the MAXIT-1 function. That is, if the last possible iteration was reached and no possible solution was found, an equilibrium was not found with the eps_prices precision. Finally, with the Ev statements, the allocations, prices, and deltas are saved for each iteration of the algorithm, thus preserving the algorithm's behavior, i.e., its history.

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
To begin, a matrix x is defined, which searches for the optimal demand of the agents given a price vector p. This part is key, since it starts the algorithm with the best combinations to reach the maximum possible utilities, starting with the initial endowments matrix self.e. After this, a for loop is made, where the optimization problem is solved for each of the agents individually to maximize their utility. Then, an optimization model is created for each agent, using the ConcreteModel function of pyomo, to solve the previous for loops. Then, for the model, all the goods, including money, are defined, and then the Cobb-Douglas utility function is used for each of the agents (exercise 1, section 5 of the paper) and the optimal utility of each agent is calculated using def until model.alpha. After this, the initial endowments are defined, to start from an iterative basis using model.e. Then, the prices are set, since they are in equilibrium. Next, the decision variables related to the optimal allocations for each agent are found—that is, the optimal variable to be found—starting from the initial endowments and imposing a restriction that endowments must be positive or 0. Following this is the objective function, which is the Cobb-Douglas maximization of each agent. A budget restriction is immediately placed so that agents' expenses do not exceed their income and there is no contradiction, that is, agents always have more positions than they spend, subtracting their current holdings from their initial holdings and ensuring that they are less than 0, to say that agents have more income than expenses. This is done through a def bc_rule(model) up to model.bc. Lines of code are then placed to solve the nonlinear optimization problem with opt and opt.solve. The results and the agents' optimal positions are then stored with a new for loop. Finally, it is verified that there is Walrasian equilibrium, setting as a condition that if the excess demand is less than 1%, Walrasian prices p are reached (described in section 3, theorem 1)

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
Here the Walraseq function is implemented, which is used to calculate the optimal holdings of agents when there is a surplus in demand, greater than that described above. It is an algebraic method. It begins by entering the final allocations to work, which are stored in x, p for the initial prices, B for the sum of the alphas for each agent, and E for the aggregate endowments of each good. B is to see the agents' returns to scale, that is, how much they produce based on the change in inputs, and E the aggregate supply of each good, that is, the quantity of goods they are willing to exchange for a certain price. Then, the linear Walras problem is solved by calculating the agent's total income divided by the sum of their preferences by the share of the different goods in the agent's utility divided by the price, thus algebraically calculating agent i's demand for good j (this process is algebraic, not iterative). After this, aggregate demand must equal aggregate supply to reach equilibrium with the excess supply. The sum of the xij being the demand and the capital E the sum of the supplies, by summing the initial endowments eij, equating the latter to reach equilibrium. In this part of the code is the alphaij which is the share of good j in the utility of i, eil which is the initial endowment of agent i in good l and Bi, which as mentioned before is the sum of the preferences of agent i. All these terms are entered into a matrix that is divided into two parts, the main diagonal, where the aggregate supply of good j is subtracted and part of the aggregate demand that depends on the same good j, which is the division that was exposed before, between the total income of the agent and the sum of the agent's preferences. And the other part that is outside the diagonal which are all the captures of how the endowments of good l affect the demand for good j. This vector A is multiplied by the price vector and results in vector b, which results in goods that depend solely on the provision of money, since the price is already set by being in cash, which would be the silver left over from excess supply and demand. After this, p is implemented, which is used to solve the previous matrix system and sets money as in cash. Finally, a for loop is created to calculate the equilibrium allocations, obeying what section 5 of the paper says, where agents must spend a fraction alpha of their income on goods, where ul is the total income of agent i adjusted by B[i] and xij as the optimal allocation for good j.
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
We start by creating exfig for the figure format. Then, a trial economy is created using Ec_aux. With this trial economy, p_we and x_we are entered to calculate the theoretical Walrasian equilibrium. It is also necessary to save each economy in each of the trials, therefore the variable Econ is created. Next, it is necessary to know the number of simulations or trials, entering ntrials (which in this case will be 10). It is also necessary to save the allocations, prices, and deltas for each simulation, therefore, EvPag, EvAlloc, and EvDelta are created. Then, the number of trades for each of the iterations is saved in the variable TimesChange. Afterwards, a variable K is created that accounts for the total iteration amount for each simulation. An equilibrium condition is then created with q_status, where 1 indicates whether the good converges to equilibrium and 0 indicates whether it does not. As discussed above, each good, including money, is equivalent to 1. The execution times must then be calculated, measured using the variable ExTimes. The maximum standard deviation for prices is then set using EvS. Equilibrium prices must also be evaluated through simulation, which is done using Evpbar. Finally, the agents' Walrasian prices and current Walrasian price allocations are evaluated using Walras_prices and Walras_alloc. The equilibrium price information for all goods except money is then stored using the BPT function, generating a 9x10 matrix where the rows represent the simulations and the columns the quantity of non-monetary goods per agent. These boxplots show the dispersion of threshold prices. Wealth_bte calculates agent wealth for each agent i in simulation k to generate a comparison using the Walras method. Utilities_bte calculates the utility of each agent's Cobb-Douglas functions and compares them with Walras equilibrium utilities. Wealth_wal measures the wealth generated by the Walras equilibrium using Walraseqrtr() prices. Finally, Utilities_wal calculates Walras equilibrium utility with its respective allocations.

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
This part of the code is key because it compares, through a number of iterations, the Walrasian method for reaching equilibrium and the bilateral method proposed in the paper. First, a for loop iterates over all the simulations carried out, allowing us to study the consistency of the comparison results. A print statement is then placed to identify the simulation evaluation steps in order to track the process. The variable Eco then creates an independent copy of the initial economy for each test, which is crucial because it allows each agent to start from the same initial state based on their holdings, and because negotiation processes alter the states of the economy. Walras_prices and Walras_alloc are also created, where the first stores the equilibrium prices at iteration k and the second stores the equilibrium allocations for all agents, respectively. This will serve as a comparison with the other method. A variable t is then created, which measures the current time taken to execute the bilateral algorithm, which will be measured later. Then the following lines of code execute the bilateral algorithm with the following variables:
- eps_prices=1e-4: Threshold for considering that prices have converged (when the standard deviation of prices between agents is less than this value)
- MAXIT=250000: Maximum number of iterations allowed
- lbda=0.998: Price premium reduction factor deltaij
- Delta_tol=1e-18: Minimum value allowed for deltaij
- inspection=ran: Strategy for selecting random pairs of agents and goods to trade
The results returned are:
- EvPag[k]: Price history during the iterative process
- EvAlloc[k]: Allocation history during the iterative process
- EvDelta[k]: History of deltaij premiums during the iterative process
- TimesChange[k]: Number of times transactions were made
- K[k]: Total number of iterations performed
- eq_status[k]: Indicator of whether equilibrium was reached (1) or no(0)
Then, the ExTimes[k] function calculates the time it took to execute the bilateral algorithm for the current test, and then prints the execution time. After this, the final equilibrium prices for the current test are stored as follows:
- Econ[k].pag: Stores the threshold prices for all agents (i.e., the limit when revenues equal costs).
- np.max: Takes the maximum for each good.
- Evpbar: Stores all goods starting from position 1.
Then, the maximum standard deviation of prices between agents is calculated for any good:
- np.std: Calculates the standard deviation per good.
- np.max: Takes the maximum of these deviations.
- Measures how close agents are to agreeing on prices (convergence).

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
In this part of the code, graphs are generated to visualize the evolution of the agent's wealth by how the iterations pass. It starts with a for loop that iterates over all the agents in the economy. Then, a variable Z is created that creates a matrix of zeros with dimensions (number of agents x number of iterations). This matrix will store the wealth of each agent in each iteration. Then another for loop is created that iterates over all the iterations of the bilateral algorithm. Then another for loop is created again that iterates for each agent ii for each iteration kk. Then, a variable Z is created that will store time series of wealth for visualization, in addition to how it evolves for each agent as they make trades. This contains the function EvAlloc[k] that contains the complete history of allocations in simulation k. Then, a [kk] is added, which is a matrix of allocations that the agents have in the kk-th iteration. Finally, [ii, 0] is added, which specifically accesses the agent's wealth and how it evolves. Z then stores agent ii's wealth at iteration kk, reflecting each agent's subjective valuation of each good, as described in section 2.8 of the paper. A plt.figure is then used to create a graph. A for loop is then created to plot each agent's wealth across all iterations. A title for the graph is then created with the plt.title function, specifying the evolution of wealth. The x and y axes are then specified with the plt.xlabel and plt.ylabel functions, respectively. The x axis represents the number of iterations, and the y axis represents the evolution of the agent's wealth. A legend is then displayed with plt.legend with the labels for each agent. Finally, the figure is saved to a file with plt.savefig , and closed with plt.close to optimize memory space.
This is the most important part of the code, as it compares the results of bilateral equilibrium and Walrasian equilibrium. First, a variable BPT is created, where the maximum of the personal prices of all agents for each good is stored, relating to part 3.5 of the paper that explains that all agents converge to compatible prices, taking the maximum as the market price. Then, with Wealth_bte, wealth is calculated as a dot product between equilibrium prices and final allocations, where Evpbar[:,k] is a vector of equilibrium prices for all goods in simulation k and Econ[k].allocations[i,:] which is the vector of final allocations of agent i. This is related to part 3.6 of the paper that implements the same as explained above. Then, the function Utilities_bte is created, which calculates the utility of each agent with a Cobb-Douglas function of the allocations, where Econ[k].alpha[i,:] are the preference parameters of agent i (exponent of the Cobb-Douglas function). ** is used to raise powers and finally np.prod for the product of all the elements. This section implements the Cobb-Douglas function described in Section 5, Equation 5.2. The Walrasian equilibrium is then calculated. The variable Wealth_wal is created, which is similar to the bilateral calculation but with Walrasian prices and allocations. The same utility function is then created, but applied to Walrasian allocations, with Utilities_wal.

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
This last part of the code evaluates the key statistical results and visualizations that validate the paper's findings. First, a printout is generated reporting how many tests reached equilibrium (eq_status=1) vs. those that did not, showing a percentage of those that converged. Then, the number of iterations per simulation called K is printed, thus also calculating the median, which is the central value. Then, there is another printout to print the amount of execution time each iteration took per test. Then, a new figure is created from the matplotlib library to graph the equilibrium prices with plt.figure(). Afterwards, Evpbar stores all the equilibrium prices pj for each good j in all the simulations calculated previously. Plt.plot in this case plots the prices pj on the x-axis and the goods j on the x-axis. Plt.title is to put a title to the graph, dealing with the equilibrium threshold prices between the agents' goods and all this with Latex notation as stated in the paper (threshold prices refers to the point where the consumer is satisfied when consuming and there is no surplus, but has enough resources to have what is necessary that he needs). Plt.xlabel is to put a title to the x-axis, which in this case puts as a title the goods with non-monetary index j. Plt.ylabel meanwhile is the title of the y-axis which is the threshold price p normalized to 1 in each of the goods j, explained in section 2 of the paper. Plt.savefig is to save the graph of the evolution of the prices pj of the goods j without including money, saving this graph in PDF with the exfig function, where Pbar is the data previously presented in this section. Plt.close closes the figure to free memory. Plt.figure to prepare the comparison between BTE vs Walras utilities. Then, using a variable x, 100 equidistant points are created using the np.linespace function, and then using np.amin, the minimum of the utilities of all agents in both BTE and Walras is found. The same is done with np.amax, but the maximum of the utilities of all agents. This is done to measure the technical efficiency of BTE and Walras, with BTE being the X axis, which represents bilateral exchanges, and Walras being the Y axis, which is optimal, generating a line x=y. It is used to see how far away or close to theory bilateral exchanges are from Walras, which are optimal and theoretical respectively. Then, through a for loop, a scatter plot is iterated to see how the utilities of bilateral exchange (BTE) vary vs. the Walras utilities in a certain iteration K using plt.scatter where Utilities_BTE[k, :] is the utility vector of all agents in the simulation k of bilateral exchanges and utilities_wal[k, :] are the utilities corresponding to the Walrasian equilibrium, where each point in this regression represents an agent in a specific simulation. Plt.plot(x, x, …) graphs the line x=y with a discontinuous style (==) by setting np.linestyle to make this happen. What we are looking for with this is to see how efficient in terms of utility the bilateral market equilibrium is vs. the Walrasian equilibrium, which is the ideal. When the points on the line of bilateral equilibrium move far away from that of Walrasian equilibrium in terms of utility, it depends. If the utility line of the bilateral market equilibrium is above the Walrasian, there is a surplus, therefore the agent has greater well-being. On the contrary, if it is below the Walrasian utility line, this surplus does not exist, therefore there is a loss of well-being. The x-axis is placed with plt.xlabel to show the utilities of the decentralized mechanism, that is, bilateral exchanges, and on the other side, on the y-axis, plt.ylabel shows the utility of the Walrasian equilibrium, that is, the artificial centralized equilibrium introduced in the paper. Then, in plt.legend, the legend is placed so that the k simulations can be tracked throughout the iterations due to the randomness of the negotiations, where each legend groups a group of points by the number of simulations there were, which in this case are 10, condensed in a single graph. Then, with plt.savefig, save the figure and show the evolution according to the simulations of each of the agents of the relationship between the Walrasian equilibrium and the bilateral equilibrium to see the efficiency of the latter.

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
Now we proceed to do the same as before, only comparing the riches of the bilateral equilibrium with the Walrasian equilibrium. Plt.figure to prepare the comparison between BTE vs Walras riches. Then, using a variable x, 100 equidistant points are created with the np.linespace function, and then with np.amin we search for the minimum of the riches of all the agents of both BTE and Walras, the same with np.amax, but the maximum of the riches of all the agents. This is done to measure the technical efficiency of BTE and Walras, with BTE being the X axis, which is the bilateral exchanges, and Walras the Y axis, which is the optimal one, generating a line x = y. It is used to see how much the riches are far from or closer to the theory of bilateral exchanges vs the Walras, which are the optimal and theoretical ones respectively. Then, through a for loop, a scatter plot is iterated to see how the wealth of bilateral exchange (BTE) varies vs. the Walras wealth in a certain iteration K using plt.scatter where Wealth_BTE[k, :] is the wealth vector of all agents in the bilateral exchange simulation k and Wealth_wal[k, :] are the wealth corresponding to the Walrasian equilibrium, where each point in this regression represents an agent in a specific simulation. Plt.plot(x, x, …) graphs the line x=y with a discontinuous style (==) by setting np.linestyle to make this happen. What we are looking for with this is to see how efficient the bilateral market equilibrium is vs. the Walrasian equilibrium, which is the ideal. When the points on the line of bilateral equilibrium with respect to the Walrasian equilibrium are on it, the bilateral equilibrium generates more wealth, and when they are not, the BTE generates a loss of wealth for the agents. The x-axis is placed with plt.xlabel to show the wealth of the decentralized mechanism, that is, bilateral exchanges, and on the other side, on the y-axis, with plt.ylabel, the wealth of the Walrasian equilibrium is placed, that is, the artificial centralized equilibrium introduced in the paper. Then, in plt.legend, the legend is placed so that the k simulations can be tracked throughout the iterations due to the random nature of the negotiations. Each legend groups a group of points by the number of simulations there were, which in this case are 10, condensed into a single graph. Then, with plt.savefig, the figure is saved and shows the evolution of the wealth according to the simulations of each of the agents in the relationship between the Walrasian equilibrium and the bilateral equilibrium to see the efficiency of the latter.

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
The Gini coefficient is defined for each simulation of bilateral market equilibria using gini_bte with a matrix of zeros, and np.zeros with the number of simulations ntrials. It is used to quantify how the BTE mechanism affects Walrasian inequality. Then the variable Gini_wal is created, which stores the gini coefficient calculated for the wealth distribution in the Walrasian equilibrium for each simulation, which will have 10 rows, 1 x each iteration, or 10. This is done to visualize the inequality between the Walras or theoretical equilibrium and the bilateral market equilibrium. Then a for loop is created that will evaluate all the simulations and how strong they are when it comes to having random order variations in the BTE equilibrium where each simulation represents a different section of the bilateral negotiation process, allowing statistical analysis. Then the variable gini_bte[k] is created, which depends on the iteration in which the gini coefficient is being evaluated in the BTE equilibrium. The gini() is used to calculate the gini coefficient for a wealth vector, in this case Wealth[k, :] which is the wealth vector of the 5 agents in simulation k in bilateral equilibrium. The gini coefficient measures inequality, where 0 is perfect equality and 1 is maximum inequality. Then the same is done for the Walras equilibrium with a variable Gini_wal[k], where the gini() is calculated for Wealth_wal[k, :], where the inequality between each simulation is measured, where the wealth distribution depends on the initial and final endowments of the agents. This is done to contrast the additional inequality caused by bilateral equilibria over the Walras or theoretical equilibrium, which in most cases reduces the inequality between the wealth of the agents. Then we proceed in the same way as with wealth and utility by graphing the line x=y to compare the performance of the Gini coefficient of wealth and that of Walras. Plt.figure to prepare the comparison between BTE vs Walras Gini coefficients. Then, using a variable x, 100 equidistant points are created using the np.linespace function, and then using np.amin, the minimum of the Gini coefficients of all agents in both BTE and Walras is found. The same is done with np.amax, but the maximum of the Gini coefficients of all agents. This is done to measure the inequality between the BTE and Walras models, with BTE being the X axis, which represents bilateral exchanges, and Walras the Y axis, which is optimal, generating a line x = y. It is used to see how much the inequality decreases or increases between the theory of bilateral exchanges vs. Walras, which are the optimal and theoretical ones respectively. Then, through a for loop, a scatter plot is iterated through to see how the Gini coefficients of bilateral exchange (BTE) vary vs. the Walras Gini coefficients in a certain iteration K using plt.scatter where Gini_BTE[k, :] is the vector of Gini coefficients of all agents in the simulation k of bilateral exchanges and Gini_wal[k, :] is the vector of Gini coefficients corresponding to the Walrasian equilibrium, where each point in this regression represents an agent in a specific simulation. Plt.plot(x, x, …) plots the line x=y with a dashed line style (==) by setting np.linestyle to make this happen. What we are looking for with this is to see how unequal the bilateral market equilibrium is vs. the Walrasian equilibrium, which is the ideal. When the points on the bilateral equilibrium line with respect to the Walrasian equilibrium line are on the latter, bilateral equilibrium generates more inequality, and when they are not, they generate a decrease in inequality for agents, the BTE. The x-axis is placed with plt.xlabel to show the inequality, that is, the Gini of the decentralized mechanism, that is, that of bilateral exchanges, and on the other hand, on the y-axis, plt.ylabel shows the inequality, that is, the Gini of the Walrasian equilibrium, that is, the artificial centralized equilibrium introduced in the paper. Then, in plt.legend, the legend is placed so that the k simulations can be tracked throughout the iterations due to the randomness of the negotiations. Each legend groups a set of points by the number of simulations there were, which in this case are 10, condensed into a single graph. Then, with plt.savefig, the figure is saved and shows the evolution of inequality according to the simulations of each of the agents of the relationship between the Walrasian equilibrium and the bilateral equilibrium to see the efficiency of the latter, as shown in the last figure of the paper on page 26. After finishing this, a new matplotlib figure is created for a box plot with plt.figure(). Then, the boxplot is created with plt.boxplot, entering BPT, which are the threshold prices for each good and simulation, that is, the equilibrium prices where all the necessary goods can be purchased. In other words, the agent's income is enough to cover all his needs, generating neither profits nor losses. This includes 10 simulations and 9 non-monetary goods. Showfliers=False allows to exclude outliers. The boxplot covers the interquartile range from Q1 to Q3, showing the median Q2, with an extension of up to 1.5*IQR. This boxplot shows the dispersion of threshold prices pj between simulations. A compact box shows an agreement in the price of good j between agents, a large one shows a certain sensitivity between the equilibrium prices of good j between agents. Then a title is given to the boxplot with plt.title notating the threshold prices with $/bar{p}_{\cdot j}$ and cdot makes it clear that it is analyzed by good and not by agent. The title is “equilibrium price thresholds” as it appears in the text always associated with the equilibrium prices of goods between agents. Then the title of the x-axis is given with plt.xlabel using r which is a raw string, which is used to ensure that everything is in latex format. Also indexing non-monetary assets in the model with $j$, excluding money, since by normalization it has a value of 1. The y-axis title is set with plt.ylabel with $p$ which is the price normalization (p=1) and the use of r to maintain the LaTeX format. The boxplot represents the pj values ​​for each good j. Then, with plt.savefig, the figure of the graph generated with plt.boxplot is saved. With +exfig, it is saved in PDF format and saved as Pag.pdf, which means that it contains the threshold prices for each of the agents in the different iterations. Then, the image is closed with plt.close to free memory.

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
Another figure is created with plt.figure to compare simulations with many differences in their equilibria. Then, a variable pmean is created to make a vector of 10 elements that are all the goods and take an average between the prices of each of these goods in each simulation, where good j are the rows and simulation k are the columns, all of this contained in Evpbar, leaving a 10x10 matrix by the quantity of goods (including money) and the number of simulations, which is 10. This is done to normalize the prices of the different simulations and see what they are converging towards. Then, a variable diff is created, which is a matrix to compare the difference between the equilibria of each of the models (BTE and Walras). The square matrix is ​​created with np.zeros, where the rows and columns will be the number of simulations that have been carried out using the algorithms, being lower triangular to avoid redundant calculations. After this, two for loops are created, one that goes through all the components of the matrix, that is, through all the simulations, and then another for loop that compares only with the previous simulations so as not to be redundant, which are i and j respectively. After these two loops, the variable diff[i,j] is placed to see the price difference in the simulations of each of the algorithms and see any large discrepancies that may exist. All of this is done with np.abs, which calculates the absolute difference for each simulation of the different algorithms between Evpbar[:, i] and Evpbar[:, j]. Then, with np.max, it takes the maximum difference between these values ​​and places them in the lower triangular matrix, thus capturing the worst result between the differences between the algorithms. Then, the variables i0 and i1 are created, which are the pairs where there was a greater price difference in simulations i and j of the goods, where diff is the aforementioned square matrix of the simulations of the different methods. Then np.argmax(diff) finds the maximum value of the price differences of the agents' goods in the different simulations as a flat index. To prevent this from happening, np.unravel_index() is used to convert the flat index into the exact coordinate where the element in question is located. Diff.shape provides the matrix indices and then performs the conversion. It then results in i0,i1 being the coordinates (row,column) with the greatest difference. Then, a graph of a specific simulation is created. In this case, the equilibrium prices of the first simulation for each good are placed, forming a vector of size n, corresponding to the total quantity of goods mentioned above (all this is done with Evpbar[:, i0]). Then, with plt.plot, a line is created with the prices per good, where the X axis corresponds to the index of the good and the Y axis to the equilibrium price for that specific good. Then, using label, a legend is created with a label for the simulation in which it is located, in this case, i0. The next line of code proceeds in the same way, only with the second simulation and its equilibrium prices, to select the other simulation with the greatest difference, graphing the same axis. Then, a print(diff) is issued, which displays the lower triangular matrix that shows the greatest difference between simulations of the different algorithms, useful for visualizing the greatest differences. Then, in the following four lines of code, previously processed data is deleted to find new possible combinations.

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
Then, in the following 3 lines of code, the second most divergent pair is graphed, creating the variables i2, i3, which are the pairs where there was a greater price difference in simulations i and j of the goods, where diff is the aforementioned square matrix of the simulations of the different methods. Then, np.argmax(diff) finds the maximum value of the price differences of the agents' goods in the different simulations as a flat index. To prevent this from happening, np.unravel_index() is used, which converts the flat index into the exact coordinate where the element in question is located. Diff.shape provides the matrix indices to then perform the conversion. Then, it results in i2, i3 being the coordinates (row, column) with the greatest difference. Then the graph of a specific simulation is created, in this case the equilibrium prices of the first simulation of each good are placed forming a vector of size n, corresponding to the total quantity of goods mentioned above (all this is done with Evpbar [:, i2]). Then with plt.plot a line is created with the prices per good where the X axis corresponds to the index of the good, and the Y axis to the equilibrium price for that specific good. Then with label a legend is created with a label of the simulation in which it is located, in this case i2. Then in the next line of code, the same procedure is carried out only with the second simulation and its equilibrium prices, to select the other simulation with the greatest difference, graphing the same axis. After this, add a legend with plt.legend placing it in the lower right corner with loc = "lower right". Then save the graph using plt.savefig with four different price curves to see how far they are from equilibrium with a legend identifying each simulation, generating a PDF file called Pbardiffs.pdf with the exfig function. Then close the graph with plt.close to save memory. Then with a print, the indices i0, i1, i2 and i3 are printed showing the number of trials selected for the analysis. Then a print is placed to begin the preparation of the table for the comparisons of the deviations of each of the simulations, where the first line is printed j&{}&{}&{}&{}&Wal(& in latex format separates the columns) which is the format of the table of the columns j which in this case are the goods and the Walrasian column and to print the corresponding numbers of the chosen simulations you put format(i0, i1, i2, i3). Next, a for loop is created that iterates over all goods from 0 to n-1, which are all contained in Ec_aux.n, which is why it is placed within the in range, to iterate over all goods. Then, we proceed again with a latex-type table like the one mentioned above, only with all the goods that are in each iteration chosen for the analysis with Evpbar, which is the matrix of equilibrium prices and the number of simulations. For each good n, a row is printed in the latex format described above. The field for each row is n, that is, each good for each simulation that was chosen that is within the columns, and then the Walrasian price is entered at the end to see how far it is from the theoretical equilibrium. Then a variable Pshow is created that prepares a combined matrix with the most divergent equilibrium prices and the Walrasian prices, where evpbar selects columns from the 4 most divergent simulations, p_we adds Walrasian prices as the last column and with np.column_stack() the simulations are combined horizontally. Then a variable called Idags is created to place professional indices for each row (goods). A latex format is created with $..$, j=0 is placed for money and j1 for the first good, placing np.asarray() for consistency with pandas. Then the variable labls is generated, which generates the format of the table in latex where $\overline p^{5}$ is the latex notation with p average bar and simulation superscript and the same for the Walras superscript with $\overline p^{W}$, thus generating 5 labels, the 4 simulations and the Walras price. Then a variable df is created, where with pd.dataframe the table number 2 of the paper is created that compares the Walrasian equilibrium prices and bilateral exchanges further from equilibrium. Where Pshow shows a matrix of numerical data where columns i, are 4 selected simulations plus the Walrasian prices p_we, and the rows are the prices of each good j (10 goods including money). Columns = labls is to place a latex format on the columns of the table where it places the prices of each of the goods in each iteration and index = idags for the latex format of the rows of the same, where it places the goods j.

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
[Falta agregar los nuevos graficos de araña creados y poner entre que linea se esta documentando]

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

[Agregar foto del output del algoritmo]

  The times are measured in Final (which are measured with the ExTimes command in Final1) to evaluate the computational efficiency of the BTE algorithm compared to the Walrasian approach. This is important because it demonstrates that it not only converges theoretically, but also does so in a practical time, even for economies with many agents and goods. The time reflects the algorithm's scalability. Scalability refers to the algorithm's ability to maintain its efficiency (in terms of execution time and resource usage) as the problem grows in size and complexity. The paper shows that BTE reaches equilibrium for 5 agents and 9 goods in 2.12 seconds (median), and the problem for 10 agents and 100 goods converges in 711 seconds, which is approximately 12 minutes. While the time increases with the size of the problem, it does so in a manageable and predictable way. The growth in time is also not exponential, making it impractical for large economies. Time grows sustainably, allowing for the study of the effect of more agents and goods participating in the market, with realistic interactions between them. In contrast, the Walrasian system requires solving systems of linear equations (such as Walraseq() in Final1), which can become computationally expensive for many goods and agents. The BTE, on the other hand, avoids this and, being an iterative and decentralized process, avoids the need to solve global systems with linear equations, making it more scalable in practice without expending significant resources. The code was run 10 times and values ​​between 2 seconds and 5 seconds of delay were obtained, which is what is expected or within a range of values ​​indicated in the paper, which makes sense for small economies. After generating the execution times and the standard deviation that the example takes to reach equilibrium, the code launches an output of the number of times the equilibrium was reached and the median of the number of iterations that the algorithm took to reach equilibrium in each of the tests. The paper says that the code has a median of 3093 iterations to reach equilibrium in each of the tests. Measurements from running the code 10 times gave a range between 3050 and 3100 iterations as the median for each of the tests, which supports what the paper says, as well as the efficiency of the BTE algorithm and how quickly each of these iterations are done, taking between 2 to 5 seconds of execution times. It also displays the average time it took the algorithm to reach equilibrium in each of the tests, which in most of the tests performed gave a value between 2 and 5, as previously stated. 

[Colocar imagen]

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
    - np.std: Calculates the standard deviation per good
    - np.max: Takes the maximum of these deviations
    - Measures how close agents are to agreeing on prices (convergence)
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

One of the graphs that is generated is called Utility, it represents the utility values ​​given for the different agents in the different trials, both for BTE and Walras where each one respectively is the y and x axis. According to what the documentation says, each point represents an agent, of which there are 60 points, since for each agent there were 20 inspections and the number of agents is 3. The graph at first glance is not very clear, but it is saying that BTE with the difference in assignments between its agents their utilities vary a lot even within the same agent the difference between the utility is quite a lot between different trials. Walras is even now less unequal than BTE seeing the dispersion of the points in the graph. This may be due to the difference in inspection methods, since having fixed inspection methods for 12 iterations makes the agent want to seek equilibrium more quickly or with less equality between agents, having more volatile values ​​between its agents and therefore increasing its utility more abruptly (represented by the points on the far right of the graph). On the other hand, in random inspections, the BTE algorithm has greater freedom to make trades, thus achieving slower but controlled growth, so that there is equality between agents. It also depends on the agent we are talking about; for example, agent 3 has almost all of the initial positions of good 2, which means its utility should be maintained to have equality of utilities with the other agents, maintaining itself over the iterations. From this we conclude that the sets of points that are lower down are the agents that have more positions (agent 2 has all the positions of good 2 and agent 3 has all the positions of good 1) and the points that are higher up are the agent that has the least amount of positions and has to increase his utility more quickly to equal agent 2 and 3 in their holdings so that they equal utility.

[Imagen]





  





























