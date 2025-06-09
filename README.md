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



