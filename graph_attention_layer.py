"""
==============================================================================
graph_attention_layer.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file defines a "Graph Attention Network" (GAT) layer, a type of Graph
Neural Network (GNN) building block, implemented in PyTorch. This is the
kind of layer that would consume the k-nearest-neighbor spot graph built by
`calcADJ()` in the spatial-transcriptomics dataset file from earlier in this
project (each tissue spot = one "node", and neighboring spots are connected
by an "edge").

Background concept #1 -- What is a graph, and why do we need a special
kind of neural network for it?
    A "graph" here just means a collection of "nodes" (individual items --
    e.g. one node per tissue spot, or one node per person in a social
    network) plus a set of "edges" connecting certain pairs of nodes
    together (e.g. "these two spots are close together on the tissue" or
    "these two people are friends"). Each node also usually has its own
    list of numbers describing it -- its "features" (e.g. a spot's gene
    expression values, or a person's profile information).
    Ordinary neural network layers (like a plain `nn.Linear`) process every
    node completely independently and never look at its neighbors. A Graph
    Neural Network layer instead lets every node gather and combine
    information from its CONNECTED neighbors, so the final representation
    of each node reflects both "what this node itself looks like" and
    "what its neighborhood looks like".

Background concept #2 -- What makes a Graph ATTENTION Network special?
    The simplest way to combine a node's neighbors is to just average all
    of them together, treating every neighbor as equally important. A GAT
    layer instead LEARNS how much attention (importance) to give to each
    neighbor -- some neighbors might matter a lot for a given node, others
    barely at all -- and computes a different attention weight for every
    single node-neighbor pair. This is conceptually the same idea as the
    "attention" mechanism used in Transformer models (like the ones behind
    modern language models), just applied to the fixed connections of a
    graph instead of to every position in a sequence.

This file defines two classes:
    1. GraphAttentionLayer -- ONE single graph-attention "head" (one
                              complete attention computation, producing
                              one updated feature vector per node).
    2. MultiHeadGAT        -- stacks SEVERAL independent
                              GraphAttentionLayer "heads" side-by-side
                              (an idea called "multi-head attention" --
                              see explanation #4 below), then combines
                              their outputs.

------------------------------------------------------------------------------
KEY ALGORITHMS/CONCEPTS USED IN THIS FILE (explained in plain language)
------------------------------------------------------------------------------
This closely follows the original Graph Attention Network paper
(Velickovic et al., 2018, "Graph Attention Networks", arXiv:1710.10903),
referenced directly in this file's original docstring.

1) Linear projection: `Wh = torch.mm(h, self.W)`
   ------------------------------------------------
   `h` holds every node's raw input feature vector (one row per node).
   Before computing any attention, every node's features are first passed
   through a shared, LEARNED linear transformation `W` (a matrix of
   weights -- exactly like a normal `nn.Linear` layer, but written out by
   hand here as a matrix multiplication). This projects every node's
   features into a new space of size `out_features`, which is more useful
   for computing attention scores and is also how the layer changes the
   size of the feature vectors as data flows through the network (e.g.
   going from `in_features` numbers per node down/up to `out_features`
   numbers per node).

2) Computing raw attention scores `e` (the "attentional mechanism")
   --------------------------------------------------------------------
   For every ordered PAIR of nodes (i, j), the layer needs one number
   `e[i, j]` describing "how much should node i pay attention to node j".
   The original GAT paper computes this efficiently, without ever
   building a full (num_nodes x num_nodes x 2*out_features) tensor, using
   a trick implemented in `_prepare_attentional_mechanism_input()`:
     - A learnable vector `a` (of length `2 * out_features`) is split into
       two halves: the first half is used to score how much a node
       "offers" as a MESSAGE SENDER (`Wh1`), and the second half scores how
       much a node "wants" as a MESSAGE RECEIVER (`Wh2`).
     - `e[i, j] = Wh1[i] + Wh2[j]`, computed for every pair at once via
       `Wh1 + Wh2.T` (broadcasting one column vector against one row
       vector, which PyTorch automatically expands into a full
       num_nodes x num_nodes score matrix).
     - The result is passed through LeakyReLU (a variant of ReLU that lets
       a small negative slope through instead of a hard 0), which is the
       specific nonlinearity choice used by the GAT paper for this step.
   This clever "split a shared vector in half" trick is mathematically
   equivalent to concatenating each node-pair's two projected feature
   vectors and running them through a small single-layer network, but is
   much faster to compute on a GPU because it avoids ever materializing
   pairwise-concatenated tensors.

3) Masking with the adjacency matrix + softmax normalization
   -----------------------------------------------------------
   We only want a node to pay attention to its ACTUAL graph neighbors, not
   to every other node in the whole graph. The `adj` matrix (built earlier
   by something like `calcADJ()`) tells us, for every pair (i, j), whether
   an edge exists (adj[i, j] > 0) or not.
     - `zero_vec = -9e15 * ones_like(e)`: an extremely large NEGATIVE
       number used as a "impossible" placeholder score.
     - `attention = where(adj > 0, e, zero_vec)`: keep the real computed
       score `e[i,j]` wherever an edge exists, but REPLACE it with the
       giant negative placeholder wherever there is no edge.
     - `attention = softmax(attention, dim=1)`: softmax converts raw
       scores into proper probabilities that sum to 1 across each node's
       row. Because non-neighbor entries were set to an enormous negative
       number, softmax squashes them down to essentially exactly 0 --
       meaning: after this step, each node's attention weights sum to 1,
       spread ONLY across its true graph neighbors, and non-neighbors
       contribute (essentially) nothing.
   This masking trick is a standard, elegant way to make softmax "ignore"
   certain entries entirely.

4) Combining neighbor information: `h_prime = torch.matmul(attention, Wh)`
   ----------------------------------------------------------------------
   Once every node has a proper probability distribution over its
   neighbors (the `attention` matrix), the new feature vector for each
   node is simply a WEIGHTED AVERAGE of its neighbors' projected feature
   vectors `Wh`, using those attention weights. Nodes with a high
   attention weight toward a particular neighbor contribute more of that
   neighbor's information to the result.

5) Multi-head attention (the `MultiHeadGAT` class)
   --------------------------------------------------
   Instead of learning just ONE way to compute attention, it often works
   better to learn SEVERAL independent attention "heads" at once, each
   with its own separate `W` and `a` parameters, run entirely in parallel,
   and then combine their results. Each head might end up specializing in
   picking up on a different kind of relationship between neighbors. Two
   different combination strategies are used here, matching the original
   GAT paper's recommendation:
     - For HIDDEN layers (`concat=True`): the outputs of all heads are
       CONCATENATED together (stuck end-to-end into one longer vector) --
       this is why `self.out_att` below expects an input size of
       `nhid * heads` (the hidden size, multiplied by however many heads
       there were).
     - For the FINAL output layer (`concat=False`): instead of
       concatenating, the heads' raw outputs are simply averaged/summed
       together internally by returning `h_prime` directly (unconcatenated,
       unactivated) -- and here, `MultiHeadGAT.forward()` applies one final
       ELU activation to that single combined output afterward.

6) Dropout
   -----------
   Dropout randomly "turns off" (zeroes out) a fraction of values during
   TRAINING ONLY (never during evaluation -- controlled automatically by
   `self.training`, a flag PyTorch sets for you when you call
   `model.train()` vs `model.eval()`). This is a regularization technique:
   by forcing the network to work even when some information randomly
   disappears, it becomes less reliant on any single feature/connection
   and tends to generalize better to new data. It is applied here in two
   places: to the raw input node features, and to the computed attention
   weights themselves (randomly ignoring some neighbor connections during
   training).

7) Activation functions used: LeakyReLU and ELU
   -------------------------------------------------
   - LeakyReLU(x) = x if x > 0, else `alpha * x` (a small negative slope,
     instead of flattening negative values to exactly 0 like plain ReLU
     does). Used here specifically while computing the raw attention
     scores `e`, matching the original GAT paper.
   - ELU(x) = x if x > 0, else `exp(x) - 1` (a smooth curve for negative
     inputs, instead of a straight line). Used on the FINAL node
     representations produced by each attention head.

8) Xavier ("Glorot") weight initialization
   ---------------------------------------------
   Before training starts, `self.W` and `self.a` need starting random
   values. Xavier initialization is a specific mathematically-derived
   recipe for picking those starting numbers so that signal strength
   neither explodes nor vanishes as it passes through the network -- it is
   the standard recommended initialization for layers that don't use ReLU
   as their primary activation (unlike Kaiming initialization, which is
   tuned specifically for ReLU networks and was used in the CapsNet model
   from earlier in this project). The `gain=1.414` (approximately
   sqrt(2)) is a scaling adjustment recommended when the layer's output
   will be followed by a LeakyReLU/tanh-like activation.

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: torch (PyTorch). No other dependencies.

2. This file only defines the LAYER/MODEL architecture -- it does not load
   data or run a training loop itself. To reproduce a specific trained
   model you additionally need the exact training script, dataset, and
   random seed used.

3. Because weight initialization (Xavier init) and dropout both use random
   numbers, fix the random seed BEFORE creating the model, and be aware
   dropout will behave differently every time you call the model in
   TRAINING mode (this is expected/intentional):
       import torch
       torch.manual_seed(0)
       model = MultiHeadGAT(in_features=..., nhid=..., out_features=...,
                             dropout=0.2, alpha=0.01, heads=4)
       model.eval()   # turn dropout OFF for a deterministic, repeatable
                      # forward pass (e.g. for evaluation/inference)

4. Expected inputs to `forward(x, adj)`:
     - `x`   : a (num_nodes, in_features) tensor -- one feature row per
               graph node (e.g. one row per tissue spot).
     - `adj` : a (num_nodes, num_nodes) adjacency matrix/tensor, where a
               value greater than 0 at position [i, j] means "node j is a
               neighbor of node i" (this is exactly the kind of object
               produced by `calcADJ()` in the spatial-transcriptomics
               dataset file used earlier in this project).

5. For fully bit-for-bit reproducible GPU training (optional, and usually
   somewhat slower), also add:
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGAT(nn.Module):
    """
    A complete "multi-head" Graph Attention Network block: several
    independent GraphAttentionLayer "heads" running in parallel, whose
    outputs are combined, followed by one more GraphAttentionLayer that
    produces the final output. See explanation #5 near the top of this
    file for why multiple heads are useful.

    Parameters
    ----------
    in_features : int
        Number of input features per graph node (per row of `x`).
    nhid : int
        Number of output features produced by EACH individual attention
        head in the hidden layer.
    out_features : int
        Number of output features per node in the FINAL output (after all
        heads have been combined).
    dropout : float
        Probability (between 0 and 1) of randomly zeroing out a value
        during training -- see explanation #6 above.
    alpha : float
        Negative-slope parameter for the LeakyReLU activation used inside
        every attention computation -- see explanation #7 above.
    heads : int
        How many independent attention heads to run in the hidden layer
        (default: 4).
    """

    def __init__(self, in_features, nhid, out_features, dropout, alpha, heads=4):
        super(MultiHeadGAT, self).__init__()
        self.dropout = dropout

        # Create `heads` independent GraphAttentionLayer objects, each with
        # its OWN separate learnable weights. `concat=True` tells each one
        # to apply an ELU activation and be ready to be concatenated
        # together with the other heads' outputs (see explanation #5).
        self.attentions = [
            GraphAttentionLayer(in_features, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(heads)
        ]

        # `add_module` registers each attention head as a proper
        # sub-module of this network (giving it a name like
        # "attention_0", "attention_1", ...). This step is REQUIRED for
        # PyTorch to correctly track these layers' parameters (so they get
        # included when you call `.parameters()`, get moved to a GPU with
        # `.to(device)`, get saved with `.state_dict()`, etc.) -- simply
        # putting them in a plain Python list (as `self.attentions` above)
        # is NOT enough on its own for PyTorch to notice them.
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # The final output attention layer. Its input size must be
        # `nhid * heads` because we are about to CONCATENATE all the
        # hidden heads' outputs together before feeding them in here.
        # `concat=False` tells it NOT to apply its own ELU activation
        # internally (that happens once, afterward, in forward() below).
        self.out_att = GraphAttentionLayer(nhid * heads, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        """
        Runs the full multi-head graph attention computation.

        Parameters
        ----------
        x : torch.Tensor
            Shape (num_nodes, in_features) -- input node features.
        adj : torch.Tensor
            Shape (num_nodes, num_nodes) -- adjacency matrix describing
            which nodes are connected (see explanation #3 above).

        Returns
        -------
        torch.Tensor
            Shape (num_nodes, out_features) -- the final, graph-aware
            feature vector for every node.
        """
        # Randomly zero out some input features (training only).
        x = F.dropout(x, self.dropout, training=self.training)

        # Run EVERY attention head on the SAME input `x` and adjacency
        # `adj`, then concatenate (`torch.cat(..., dim=1)`) all their
        # outputs side-by-side into one longer feature vector per node.
        # `.squeeze(0)` removes a redundant size-1 batch dimension that
        # some GraphAttentionLayer computations may produce.
        x = torch.cat([att(x, adj).squeeze(0) for att in self.attentions], dim=1)

        # Randomly zero out some of the concatenated hidden features
        # (training only) before the final layer.
        x = F.dropout(x, self.dropout, training=self.training)

        # Run the final attention layer, then apply one ELU activation to
        # its output (since that final layer itself was told
        # `concat=False`, meaning it skips its own internal activation).
        x = F.elu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
    ONE single Graph Attention layer/"head" -- the core building block of
    this file. Implements the attention mechanism from the Graph Attention
    Network paper (Velickovic et al., 2018, arXiv:1710.10903). See
    explanations #1-#4, #6, #7, and #8 near the top of this file for a
    full step-by-step plain-language walkthrough of everything this class
    computes.

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features per node produced by this layer.
    dropout : float
        Probability of randomly zeroing out an attention weight during
        training.
    alpha : float
        Negative-slope parameter for the LeakyReLU used when computing raw
        attention scores.
    concat : bool
        If True, this layer applies an ELU activation to its output and is
        expected to have its output concatenated with other heads (used
        for hidden layers in a multi-head setup). If False, the raw,
        un-activated output is returned instead (used for a final output
        layer, where activation/combination happens elsewhere).
    """

    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # `self.W`: the shared learnable linear-projection matrix applied
        # to every node's raw input features (see explanation #1 above).
        # `torch.empty` just allocates memory without setting any values
        # yet -- `nn.init.xavier_uniform_` (below) then fills it in with
        # properly-scaled random starting numbers (see explanation #8).
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # `self.a`: the shared learnable "attention scoring" vector, later
        # split into two halves inside `_prepare_attentional_mechanism_input`
        # (see explanation #2 above).
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU activation used specifically for computing raw
        # attention scores (see explanation #7 above).
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        Runs one full attention computation for every node in the graph at
        once.

        Parameters
        ----------
        h : torch.Tensor
            Shape (num_nodes, in_features) -- raw input node features.
        adj : torch.Tensor
            Shape (num_nodes, num_nodes) -- adjacency matrix (nonzero
            entries mark real graph edges).

        Returns
        -------
        torch.Tensor
            Shape (num_nodes, out_features) -- this layer's updated
            feature vector for every node, after gathering
            attention-weighted information from each node's neighbors.
        """
        # Step 1: project every node's raw features through the shared
        # learnable matrix W (see explanation #1 above).
        Wh = torch.mm(h, self.W)

        # Step 2: compute the raw (pre-mask, pre-softmax) attention score
        # for every possible pair of nodes (see explanation #2 above).
        e = self._prepare_attentional_mechanism_input(Wh)

        # Step 3: build a matrix full of an enormous negative number,
        # matching `e`'s shape -- this will stand in for "no edge exists
        # here, so this pair should get essentially zero attention".
        zero_vec = -9e15 * torch.ones_like(e)

        # Step 4: wherever `adj` says an edge really exists (`adj > 0`),
        # keep the real computed score `e`; everywhere else, swap in the
        # giant negative placeholder instead (see explanation #3 above).
        attention = torch.where(adj > 0, e, zero_vec)

        # Step 5: softmax turns each node's row of scores into a proper
        # probability distribution over its neighbors (summing to 1);
        # thanks to step 4, non-neighbors end up with ~0 probability.
        attention = F.softmax(attention, dim=1)

        # Step 6: randomly zero out some attention weights during training
        # only (regularization -- see explanation #6 above).
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Step 7: combine every node's neighbors' projected features
        # (`Wh`), weighted by the attention probabilities just computed
        # (see explanation #4 above) -- this produces the new,
        # graph-aware feature vector for every node.
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            # Hidden-layer mode: apply the ELU activation before returning
            # (see explanation #7 above), since this output is meant to be
            # concatenated with other attention heads' outputs.
            return F.elu(h_prime)
        else:
            # Final-layer mode: return the raw combined result unchanged;
            # any activation/combination happens outside this class
            # (e.g. in MultiHeadGAT.forward()).
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Efficiently computes the raw attention score `e[i, j]` for EVERY
        pair of nodes (i, j) at once, without ever building a full
        pairwise-concatenated tensor. See explanation #2 near the top of
        this file for the full plain-language walkthrough of exactly what
        this computes and why it is mathematically equivalent to (but much
        faster than) the "concatenate-then-score" description used in the
        original GAT paper.

        Parameters
        ----------
        Wh : torch.Tensor
            Shape (num_nodes, out_features) -- every node's PROJECTED
            feature vector (the output of `torch.mm(h, self.W)` in
            forward() above).

        Returns
        -------
        torch.Tensor
            Shape (num_nodes, num_nodes) -- the raw (pre-mask,
            pre-softmax) attention score for every ordered pair of nodes.
        """
        # First half of `self.a` scores each node as a "message sender".
        # Result shape: (num_nodes, 1) -- one number per node.
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])

        # Second half of `self.a` scores each node as a "message
        # receiver". Result shape: (num_nodes, 1) -- one number per node.
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        # Broadcasting a (num_nodes, 1) column vector against its own
        # (1, num_nodes) transpose produces a full (num_nodes, num_nodes)
        # matrix where entry [i, j] = Wh1[i] + Wh2[j] -- exactly the raw
        # attention score for the (i, j) node pair, computed for every
        # pair simultaneously in one operation.
        e = Wh1 + Wh2.T

        # Apply the LeakyReLU nonlinearity (see explanation #7 above)
        # before these scores get masked and softmax-normalized back in
        # forward().
        return self.leakyrelu(e)
