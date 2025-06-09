# DeJoRo-Bilateral
Official repository for the paper (link)
## Study Objective:
This repository seeks to implement and thoroughly analyze the bilateral negotiation-based market equilibrium model proposed in the article *"Reaching an Equilibrium of Prices and Holdings of Goods through Direct Buying and Selling"* (Deride et al., 2023), with three main axes: (1) **theoretical validation** of the convergence toward price and holdings equilibria through decentralized transactions, contrasting the results with traditional Walrasian models; (2) **detailed empirical analysis** of Examples 1, 2, and 3 of the article, exploring how variables such as randomness in the trading order, the structure of utility functions (Cobb-Douglas vs. non-Cobb-Douglas), and the initial distribution of goods (including cases with zero holdings) affect the dynamics and stability of the equilibrium; and (3) **model extension** to overcome identified limitations (such as the assumption of positive holdings across all assets) by proposing algorithmic adjustments to handle realistic scenarios involving non-core assets. The repository will include replicable simulations, interactive visualizations of price and holdings developments, and technical documentation linking the results to decentralized equilibrium economic theories, thus providing a practical tool for studying markets based on direct interactions between agents.
### 1. Theoretical validation
According to the supporting paper "Reaching an equilibrium of prices and holdings of goods through direct buying and selling", a theoretical comparison is presented that demonstrates the superiority of decentralized methods over the classical Walrasian model in terms of economic realism, practical implementation, and convergence to stable equilibria.
#### Walrasian Model
There are three criticisms of the Walrasian model, which has fundamental problems:
- **Unrealistic centralization:** Requires a "Walrasian auctioneer" that sets prices abstractly, without real market mechanisms (section 1 of the paper)
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
```python
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)
```
Implement the random inspection of agents and goods mentioned in part 5 of the paper (random inspection strategy)
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
```python
def evalp(self,x):
        A = np.zeros((self.I,self.n-1))
        for i in range(self.I):
            for j in range(self.n-1):
                A[i,j] = (self.alpha[i,j+1]/self.alpha[i,0])*(x[i,0]/x[i,j+1])
        return A
```
The evalp function is defined, where the agent's threshold prices p_ij(x_i) are calculated. Threshold price is understood as the agent's purchases and sales through an initial vector of goods or initial allocations. It is measured in units of the agent's utility per unit of price. It implements $p_{ij} = (\alpha_{ij}/\alpha_{i0}) \times (x_{i0}/x_{ij})$ for j>0 (equation 2.8 of the paper)
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











