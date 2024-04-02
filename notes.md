# Notes

What have I been reading up until now? 2024-04-01
1. [Outrageously Large Neural Networks](#outrageously-large-neural-networks)
2. [Switch Transformers](#switch-transformers)
3. [ST-MoE](#st-moe)
4. [Empirical understanding of MoE](#towards-an-empirical-understanding-of-moe)
5. [Hash Layers](#hash-layers-for-large-sparse-models)
6. [Mamba](#mamba-linear-time-sequence-modeling)
7. [Linear State-Space Layers](#linear-state-space-layers)
8. [HiPPo Recurrent Memory](#hippo-recurrent-memory)

## MoE
### Outrageously Large Neural Networks
The Sparsely-Gated Mixture-of-Experts Layer
[Link](https://arxiv.org/abs/1701.06538)

#### Summary

Introduction of a new component: the sparsely-gated MoE

**Key take away:**
Rather than applying the same parameters to all inputs, sparse expert networks dynamically select which parameters to use for each input

This gatting function $G(x)$ selects two experts to perform computations on $x$. And $E_{i}(x)$ is the $i$-th experts output

We note that:

$$ y = \sum_{i=1}^{n} G(x)_{i} E_{i}(x) $$

##### The Gating Network

Different types of gating:

- **Softwax**: $G_{\sigma}(x) = Softmax(x \cdot W_{g})$ ($W_{g}$ is a trainable weight matrix)

- **Noisy Top-K**: Adding a tunabe Gaussian noise matrix to only keep the top k values and set the rest to $- \infty$ st:
$$ G(x) = Softmax(KeepTopK(H(x), k)) \\
H(x)_{i}=(x\cdot W_{g})_{i} + StdNormal() \cdot Softplus((x \cdot W_{noise})_{i}) \\
KeepTopK(v, k)_{i} = \begin{cases}
  v_{i}, & \text{if $v_{i}$ is in the top k elements of v}, \\
  - \infty, & \text{otherwise}.
\end{cases}
$$

##### Balancing Expert Utilization
Using an importance matrix tied to the gated matrix weights to not oversaturate expert

##### Dataset:
One Billion Word Benchmark

### Switch Transformers
Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
[Link](https://arxiv.org/abs/2101.03961)

Simplify routing. Switch transformer encoder block replaces FFN layer

Instead of routing to 2 experts as MoE usually do, they route to only one expert.

##### Load Balancing

Auxiliary load balacing loss helps distribute the tokens better across different experts. 
Before: Done with separate load-balacing and importance weighting losses

Using Mesh-TensorFlow

Dataset:
C4 (~7TB): https://www.tensorflow.org/datasets/catalog/c4

### ST-MoE
DESIGNING STABLE AND TRANSFERABLE SPARSE  EXPERT MODELS
[Link](https://arxiv.org/abs/2202.08906)

#### Summary

Setting the number of experts:
Depends on hardware specifically, memory transfer vs efficiency trade off.
In sparse models increasing number of experts decreases the compute-to-memory ratio.
For TPU's they recommend one expert (or less) per core.

Choosing routing algorithm:
Test on top-1 2 and n routing algorithms

##### Conclusions:
Increasing CF improves quality.
Small gains of top-(n+1) over top-n given **fixed capacity factor**

Router-z-loss that resolves instability issues

Dense FFN immediately before and after each sparse layer improves quality.

Batch Prioritzied Routing (BPR) worked on. Gives a global view of all tokens to determine
which tokens should be dropped instead of left-to-right ordering.
Works by looking at all N tokens getting sent to Expert $i$ and then only routing the $M$
ones with the highest probabilities from the router.

### Towards an empirical understanding of MoE
### Hash Layers For Large Sparse Models

[Link](https://arxiv.org/abs/2106.04426)

Investigating if non-parametric models can approach learned models

Outperforms Switch Transformers and BASE layers.

Note: This is a good sign as it tokenization might prove to not be the most optimal
solution if non-parametric alternatives work they could prove to also outperform
at a byte level too.

Background
Let us first introduce the Mixture-of-Experts setting where we apply our hash-based routing strategy.
We use the same setting as where a feedforward network (FFN) in a Transformer is
replaced by its MoE version. Given a tokenized input sequence ${x_1 , x_2 , . . . , x_T }$ of $T$ tokens, a
representation for each token is computed in parallel by a standard Transformer
$$
h_1 , h_2 , . . . , h_T = TRANSFORMER (x_1 , x_2 , . . . , x_T ).
$$
The Transformer consists of L layers that computes final hidden states for each token, and each layer
is composed of self-attention and FFN sublayers, where FFNs are two-layer fully connected networks
$$
h̄_{l}^{t} = SelfAttn(h_{t}^{l-1}) \\
h_{l}^{t} = FFN(h̄_{l}^{t})
$$
(2)
Here we omit skip-connections and normalization for brevity. We can then replace one or more of the
FFN sublayers with expert modules. Replacing the FNN at layer l with K expert FFNs, their output
is then mixed with some gating function $g(·)$:
$$
h_{l}^{t} = FFN(h̄_{l}^{t})
→
h_{l}^{t} = \sum_{i=1}^{K} g_i (h_{l}^{t}) FFN_i (h_{l}^{t}),
t = 1, . . . , T,
$$

where importantly each token is routed to a different mixture of experts, as the gating function
depends on the token’s specific hidden state h̄lt .
Sparse MoE methods assume gating values gi are often zero, so only a few experts need to be
computed for better efficiency. As expert FFNs do not share parameters, the number of parameters
increases with K while the amount of computations per input token stays the same if the MoE FFN
only routes to a single expert, and computation of gi is cheap. While this allows training of large
capacity models with small compute budget, optimizing gi in the sparse setting can be tricky.

#### Hash functions used

Random Hash,
Balanced hash:
Bigram Hash: Uses previous and next token (This could be interesing in the relation between different bytes)
Previos Token Hash similar to Bigram but only previous token
Position hash: Based on position of token in sequence
Oracle Future Hash: based on output token
Clustered Hashes: Similar tokens may want to route to the same expert. (I don't think this one will be necessarily that helpful)
Dispersed Hashes: Opposite to clustered hashes. (Could be interesting too)

#### MultiHash Layers

Best deploying multiple hashes give better allocations in many contexts.

We consider such schemes in the context of sparse routing. Let us
assume we are given N different hashing functions, and for a given input token x we compute these
hashes, denoted as $k_m = hash_m (x), m=1,...,N$ Assuming the usual expert FFN is a function

Use hashing to select the parameters we are going to use for each segment, and then
concatenate them together. The advantage is that we are now no longer reliant on the quality of a
single hash function, but have multiple chances to produce good quality partitions. This perhaps can
also be seen as analogous to the multi-head attention process already used in Transformers.


Datasets:
Pushshift.io Reddit
RoBERTa+cc100en Data
Wikitext-103
Downstream BST tasks

## Mamba

### Mamba: Linear Time Sequence Modeling
[Link](https://arxiv.org/abs/2312.00752)
Improvements:
- Letting SSM paramenters be functions of the input addresses their weakness
with discrete modalities, this permits the model to selectively propagate or forget info
along the sequence length dim depending on the current token. 
- The change prevents the use of efficient convolutions, we design a hardware-aware
parallel algorithm in recurrent mode.

### Linear State-Space Layers:
[Link](https://arxiv.org/abs/2110.13985)

![LSSLOverview](assets/LSSL-Overview.png)
**Linear State-Space Layer (LSSL)**:

Is a simple sequence model that maps a 1-dim function or sequence $u(t) \to y(t)$
through an implicit state $x(t)$ by simulating a linear continuous-time state-space
representation in discrete-time
$$\dot x(t) = Ax(t) + B(t) \qquad (1) \\ y(t) = Cx(t) + Du(t) \qquad (2)$$
where $A$ controls the evolutions of the system and $B,C,D$ are projection parameters.

#### Properties:

- LSSLs can be a **linear recurrence** if we specify a step-size $Delta t$ and by applying
a [discretization](#discretization).
This has many perks such as being able to be simulated during inference as a recurrent
model with constant memory and computation per time step.
- LSSLs can be represented by a continous convolution as stated by the [discretization](#discretization). Which in it's 
discrete-time version can **parallelized**.
- LSSLs are differential equations.Thus have a **continuous-time** interpretation.

#### Tradeoffs and solution:

1. They inherit the limitations of both RNNs and CNNS on long sequences
2. The choice of A and $Delta t$ is crucial for the model to work well. 

These issues are addressed by chosing A from a class of structured matrices that generalize
prior work on continuous-time memory and mathematically capture long dependencies
with respect to a learnable family of measures. More info on [HiPPo Recurrent Memory](#hippo-recurrent-memory)
and [Continuous-Time Memory](#continuous-time-memory)

#### Discretization

Using the **generalized bilinear transform (GBT)** specialized in linear ODEs of shape (1)
$$ x(t + \Delta t) = (I - \alpha \Delta t \cdot A)^{-1} (I + (1 − \alpha) \Delta t \cdot A)x(t) + \Delta t(I − \alpha \Delta t \cdot A)^{−1} B \cdot u(t)) \qquad (3)$$
Cases: 
1. $\alpha = 0$  GBT becomes a classic *Euler method* 
2. $\alpha = 1$  GBT becomes a *backward Euler method*
3. $\alpha = \frac{1}{2}$ GBT becomes a *bilinear method* which preserves stability

For $\alpha = \frac{1}{2}$ we can define $\bar A$ and $\bar B$ to be the matrices on (3)

Such that the **discrete-time** state-space model becomes
$$x_t = \bar{A}x_t + \bar{B}u_t \qquad (4) \\ y_t = Cx_t + Du_t \qquad (5)$$

#### Continuous-Time Memory
For more information on HiPPo go to my summary of the paper [HiPPo](#hippo-recurrent-memory)

For an input function $u(t)$, a fixed probability measure $w(t)$, and a sequence of $N$ basis
functions such as polynomials. At every time t, we can project the history of $u$ onto this basis
yielding a vector of coefficients $x(t) \in {\R}^N$. This mapping can be done using the
**High-Order Polynomial Projection Operator (HiPPO)**.
In special cases such as the uniform measure $w = I\{[0, 1]\}$ and the
exponentially-decaying measure $w(t) = exp(−t)$ [HiPPo](#hippo-recurrent-memory) showed that $x(t)$ satisfies a differential equation
(1) and derived closed forms for the matrix A.

#### Views of LSSLs
Given a fixed state space representation A,B,C, and D
1. Continuous-Time:
$$\dot x(t) = Ax(t) + B(t) \qquad (1) \\ y(t) = Cx(t) + Du(t) \qquad (2)$$
2. Recurrence:
The recurrent state $x_{t-1} \in {\R}^{H X N}$ carries the context of all inputs before $t$. <br>
Thus the output $y_t$ and current state $x_t$ are computed using: 
$$x_t = \bar{A}x_t + \bar{B}u_t \qquad (4) \\ y_t = Cx_t + Du_t \qquad (5)$$

3. Convolution: Let initial state $x_{-1} = 0$ then (4) + (5) yields
$$y_k = C{(\bar A)}^k \bar B u_0 + C{(\bar A)}_{k-1}\bar B u_1 + \dots + C{(\bar AB)}u_{k-1} + \bar B u_k + D u_k \qquad (6)$$
Then $y$ is simply the (non-circular) convolution $y = K_L(A,B,C) * u + Du$, where
$$K_L(A,B,C) = (C A^i B)_{i \in [L]} \in \R^L = (CB, CAB,\dots, CA^{L-1}B) \qquad (7)$$
The entire output $y \in \R^{HxL} can be computed at once by a convolution, which can be efficiently implemented with three FFTs.

### HiPPO Recurrent Memory
[Link](https://arxiv.org/abs/2008.07669)

RNN have a hard time capturing long-term dependencies resulting in vanishing gradients.
This has been addressed in Legendre Memory Units (LMU) and Fourier Recurrent Units. But 
these solutions still still lack theoretical guarantees on gradient bounds.

HiPPO tries to able to address dependencies of arbitrary length **without** priors on the timescale.

Using **orthogonal polynomials** a natural basis emerges which you can update with an optimal
polynomial approximation as the input sequence is being revealed through time.


### Orthogonal Polynomials
[Link](https://arxiv.org/pdf/1303.2825.pdf)




