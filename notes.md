# Notes

## MoE

### Outrageously Large Neural Networks
The Sparsely-Gated Mixture-of-Experts Layer
Link: https://arxiv.org/abs/1701.06538

#### Summary

Introduction of a new component: the sparsely-gated MoE

**Key take away:**
Rather than applying the same parameters to all inputs, sparse expert networks dynamically select which parameters to use for each input

This gatting function $G(x)$ selects two experts to perform computations on $x$. And $E_{i}(x)$ is the $i$-th experts output

We note that:

$$ y = \sum_{i=1}^{n} G(x)_{i} E_{i}(x) $$

##### The Gating Network

Different types of gating:

- Softwax: $G_{\sigma}(x) = Softmax(x \cdot W_{g})$ ($W_{g}$ is a trainable weight matrix)
- Noisy Top-K: Adding a tunabe Gaussian noise matrix to only keep the top k values and set the rest to $- \infty$ st:
$$
G(x) = Softmax(KeepTopK(H(x), k)) \\
H(x)_{i}=(x\cdot W_{g})_{i} + StdNormal() \cdot Softplus((x \cdot W_{noise})_{i}) \\
KeepTopK(v, k)_{i} = \begin{cases}
  v_{i}, & \text{if $v_{i}$ is in the top k elements of v}, \\
  - \infty, & \text{otherwise}.
\end{cases}
$$

##### Balancing Expert Utilization

Using an importance matrix tied to the gated matrix weights to not oversaturate expert

Dataset:

One Billion Word Benchmark

#### Switch Transformers
Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
link: https://arxiv.org/abs/2101.03961

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
link: https://arxiv.org/abs/2202.08906

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

### Hash Layers For Large Sparse Models

link: https://arxiv.org/abs/2106.04426

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
