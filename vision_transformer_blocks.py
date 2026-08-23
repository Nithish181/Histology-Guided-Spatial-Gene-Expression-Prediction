"""
==============================================================================
vision_transformer_blocks.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file implements the core building blocks of a "Transformer" neural
network (the same family of architecture behind modern large language
models), specifically set up in the style commonly used for Vision
Transformers (ViT) -- i.e. Transformers applied to image-derived data rather
than text. It is written in PyTorch, using the `einops` library for some
tensor-reshaping operations.

Background concept #1 -- What problem does a Transformer solve, and what is
"self-attention"?
    A Transformer processes a SEQUENCE of items (in text, these are words;
    in a Vision Transformer, these are usually small image patches, or --
    as is likely the case in this project, given the capsule/graph-based
    feature extraction built in earlier files -- some other kind of
    per-location feature vector). The key idea, called "self-attention",
    is that every item in the sequence gets to look at EVERY OTHER item in
    the same sequence and decide how much attention/importance to give
    each one, before updating its own representation. This lets the model
    capture long-range relationships between items regardless of how far
    apart they are in the sequence -- unlike, say, a convolution, which
    can typically only look at a small local neighborhood at once (as
    seen in the convolutional layers of `capsule_network_model.py` and
    `omni_dimensional_dynamic_conv.py` earlier in this project).

Background concept #2 -- Why does this project need BOTH graph attention
(`graph_attention_layer.py`) AND this separate Transformer/self-attention
mechanism?
    They serve related but different purposes. The Graph Attention layer
    from earlier in this project only lets a node (e.g. one tissue spot)
    attend to its LOCAL graph neighbors (the handful of physically nearest
    spots, as determined by `spot_knn_graph_builder.py`). The
    self-attention mechanism in THIS file instead lets every item attend
    to EVERY other item, with no distance restriction at all -- useful for
    capturing longer-range patterns across an entire tissue sample that a
    purely local neighbor-based graph might miss. It is common in modern
    architectures to combine both kinds of attention (local/graph-based
    and global/full self-attention) for complementary strengths.

This file defines, from the ground up:
    1. pair()          -- a tiny utility function (explained below).
    2. PreNorm          -- wraps another layer so its input gets normalized
                            first (see "Pre-Norm architecture" below).
    3. FeedForward       -- a small two-layer neural network applied
                            independently to every item in the sequence
                            (sometimes called an "MLP block").
    4. Attention         -- the actual multi-head self-attention
                            computation (the heart of a Transformer).
    5. Transformer       -- stacks several (Attention + FeedForward) pairs
                            on top of each other, `depth` times, with
                            residual ("skip") connections around each one.
    6. ViT               -- a thin wrapper around `Transformer`, adding an
                            embedding-level Dropout before it.

------------------------------------------------------------------------------
KEY ALGORITHMS/CONCEPTS USED IN THIS FILE (explained in plain language)
------------------------------------------------------------------------------

1) Multi-head self-attention, step by step (the `Attention` class)
   ------------------------------------------------------------------
   For every item in the input sequence, the layer computes THREE
   different vectors from it:
     - a "Query" (Q): "what am I looking for?"
     - a "Key"   (K): "what do I have to offer, as a label/tag?"
     - a "Value" (V): "what do I have to offer, as actual content?"
   These are all computed at once here via a single combined linear layer
   `self.to_qkv`, then split into three separate Q, K, V tensors using
   `.chunk(3, dim=-1)`.
   For every PAIR of items (i, j) in the sequence, the layer computes a
   raw attention score by taking the dot product of item i's Query with
   item j's Key: a high score means "item j's key strongly matches what
   item i's query is looking for". These raw scores are divided by
   `self.scale` (explained in point #2 below), then passed through a
   Softmax so that, for every item i, its attention scores across all
   possible j's add up to exactly 1 (turning raw scores into a proper
   probability-like weighting). Finally, item i's NEW representation is
   computed as a WEIGHTED AVERAGE of every item's Value vector, using
   those attention weights -- items that scored a high match get to
   contribute more of their Value content to item i's updated
   representation.
   "Multi-head" means this entire Q/K/V/attention process is run several
   times IN PARALLEL (`heads` separate times), each with its own smaller
   slice of the projected dimensions, so the model can simultaneously
   track several different KINDS of relationships between items (e.g. one
   head might learn to track one kind of pattern, another head a
   different kind), before all the heads' results get concatenated back
   together and passed through one final linear layer (`self.to_out`).

2) Scaled dot-product attention -- why divide by `self.scale`?
   -------------------------------------------------------------------
   `self.scale = dim_head ** -0.5` is `1 / sqrt(dim_head)`. Without this
   scaling, as the dimensionality of the Query/Key vectors (`dim_head`)
   grows, the raw dot-product scores tend to grow larger in magnitude
   too, which can push the following Softmax into a very "extreme"/
   saturated regime (nearly all attention weight dumped onto just one
   item, with the rest of the possible gradient signal effectively
   vanishing during training). Dividing by `sqrt(dim_head)` keeps the
   scores in a more reasonable, consistent numeric range regardless of
   `dim_head`, which is exactly the "Scaled Dot-Product Attention" trick
   from the original Transformer paper ("Attention Is All You Need",
   Vaswani et al., 2017) -- and the very same underlying idea used inside
   the efficient routing computation of `RoutingLayer` back in
   `capsule_network_model.py`.

3) `einops.rearrange` -- reshaping tensors with readable, named axes
   -----------------------------------------------------------------------
   Ordinary PyTorch reshaping (`.view()`, `.permute()`, etc.) requires you
   to keep track of exactly which numeric axis position means what, which
   gets error-prone and hard to read once tensors have many dimensions.
   `einops`'s `rearrange` function instead lets you describe a reshape
   using short, human-readable pattern strings. For example:
       rearrange(t, 'b n (h d) -> b h n d', h=h)
   reads as: "this tensor `t` has a batch dimension `b`, a sequence-length
   dimension `n`, and a combined dimension `(h d)` that is actually
   `heads * dim_head` numbers squashed together -- split that combined
   dimension back apart into two separate dimensions, `h` (heads) and `d`
   (dim_head), and move `h` to sit right after the batch dimension." The
   opposite operation, `rearrange(out, 'b h n d -> b n (h d)')`, does the
   reverse: it takes the separate per-head results and squashes them back
   together into one long per-item vector (this is exactly the
   "concatenate all heads back together" step described in point #1
   above).

4) `einsum` -- a compact way to describe batched matrix multiplications
   -----------------------------------------------------------------------
   "Einstein summation notation" (`einsum`) is a flexible way to describe
   many different kinds of multiply-and-sum tensor operations using a
   short pattern string, similar in spirit to `einops.rearrange` above.
       einsum('b h i d, b h j d -> b h i j', q, k)
   reads as: "for every batch `b`, every head `h`, every query position
   `i`, and every key position `j`: multiply together `q`'s and `k`'s
   values along their shared `d` dimension and sum the result" -- which is
   exactly the dot-product-between-every-pair-of-positions computation
   described in point #1 above, computed for the WHOLE batch and ALL
   heads simultaneously in one call, with no explicit Python loop needed.
   The second `einsum` call in this file, computing `out`, works the same
   way: it multiplies the attention weights by the Value vectors and sums
   over the key-position dimension `j`, producing one final output vector
   per query position `i`.

5) Residual ("skip") connections: `x = attn(x) + x`
   -------------------------------------------------------
   Rather than replacing `x` entirely with the attention layer's output,
   the ORIGINAL input `x` is ADDED back on afterward. This is called a
   "residual" or "skip" connection. It means each layer only needs to
   learn what CHANGE/adjustment to make to its input, rather than having
   to learn how to reconstruct the entire input from scratch every single
   time -- which makes very deep networks (many stacked layers) much
   easier to train successfully, since gradients during backpropagation
   have a direct, unobstructed path back through every `+ x` addition,
   rather than only being able to flow back through a long chain of
   transformations.

6) Pre-Norm architecture and `LayerNorm`
   ------------------------------------------
   The `PreNorm` class normalizes `x` FIRST, before passing it into
   whatever sub-layer (`fn`) it wraps (either `Attention` or
   `FeedForward`), rather than normalizing AFTER the sub-layer runs (an
   alternative arrangement called "Post-Norm", used in the original
   Transformer paper). This "Pre-Norm" arrangement is now the more common
   choice in modern Transformer implementations, since it tends to make
   training deep stacks of Transformer blocks noticeably more stable.
   `nn.LayerNorm` itself normalizes each individual sequence item's own
   feature vector to have mean 0 / standard deviation 1, entirely on its
   own -- this is different from the `BatchNorm2d` layers seen earlier in
   this project (`capsule_network_model.py`,
   `omni_dimensional_dynamic_conv.py`), which instead normalize using
   statistics computed ACROSS the whole batch of examples. LayerNorm's
   per-example, no-batch-statistics-needed approach is generally
   preferred for Transformers, partly because sequence lengths and batch
   compositions can vary, and per-example normalization behaves
   identically whether you process one example or many at once.

7) The `FeedForward` block ("MLP" / position-wise feed-forward network)
   -------------------------------------------------------------------------
   After the attention sub-layer lets items exchange information with
   each other, the `FeedForward` block then processes EVERY item's
   (now-updated) feature vector completely INDEPENDENTLY (it never lets
   different sequence positions interact with each other -- that's only
   the Attention block's job). It's a simple two-layer network: expand
   the feature dimension up to a larger `hidden_dim`, apply a GELU
   activation, apply Dropout, then project back down to the original
   `dim`, followed by one more Dropout. GELU ("Gaussian Error Linear
   Unit") is a smooth, curved activation function (unlike the sharp-cornered
   ReLU/LeakyReLU functions seen in earlier files of this project) that has
   become a very common default choice specifically inside Transformer
   FeedForward blocks.

8) The `project_out` special case in `Attention`
   ----------------------------------------------------
   `project_out = not (heads == 1 and dim_head == dim)`. If there is only
   ONE attention head, AND that head's dimension already exactly matches
   the model's overall working dimension `dim`, then concatenating the
   (single) head's output back together already produces a tensor of
   exactly the right shape -- so the extra final linear projection layer
   (`self.to_out`) would be redundant, and is replaced with a plain
   `nn.Identity()` (a "do nothing, just pass the input through unchanged"
   placeholder layer) instead, saving a small number of unnecessary
   parameters and computation in that specific edge case.

9) The `pair()` helper function
   -----------------------------------
   `pair(t)` simply returns `t` unchanged if it is already a Python tuple,
   or wraps it into a `(t, t)` tuple otherwise. This is a common
   convenience utility in Vision Transformer implementations, normally
   used so that a size-related argument (like an image height/width, or a
   patch height/width) can be given EITHER as one single number (meaning
   "use this same value for both height and width") OR as an explicit
   `(height, width)` tuple (for non-square configurations). NOTE: this
   particular function is not actually called anywhere else in THIS file
   -- it is defined here for use by whatever code elsewhere in this
   project handles turning raw images into the patch-based input sequence
   that eventually gets fed into the `ViT`/`Transformer` classes defined
   below.

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: torch, einops.

2. This file only defines the MODEL architecture -- it does not load data
   or run a training loop itself.

3. Every layer here (`nn.Linear`, `nn.LayerNorm`) uses PyTorch's own
   default weight initialization (no custom initialization code is
   written in this file, unlike the explicit Kaiming/Xavier
   initialization seen in `capsule_network_model.py`,
   `graph_attention_layer.py`, and `omni_dimensional_dynamic_conv.py`
   earlier in this project). To get reproducible starting weights, fix
   the random seed BEFORE creating the model:
       import torch
       torch.manual_seed(0)
       model = ViT(dim=..., depth=..., heads=..., mlp_dim=...)

4. Dropout is used throughout this file (`emb_dropout`, and the
   `dropout` argument passed into both `Attention` and `FeedForward`).
   Remember to call `model.eval()` before running inference/prediction if
   you want deterministic, repeatable output (see the Dropout explanation
   in `graph_attention_layer.py` earlier in this project for full
   background on why `model.train()` vs `model.eval()` matters here).

5. Input shape expected by `ViT.forward(x)` / `Transformer.forward(x)`:
   a tensor of shape `(batch_size, sequence_length, dim)` -- i.e. this
   implementation expects an ALREADY-EMBEDDED sequence of feature
   vectors (there is no patch-embedding, positional-encoding, or
   class-token logic inside this particular file, unlike a typical
   complete ViT implementation) -- that embedding step must happen
   elsewhere in this project's full pipeline (very likely using the
   capsule/graph/dynamic-convolution building blocks from the earlier
   files in this project) before this Transformer is applied.

6. A commented-out line, `# @get_local('attn')`, appears just above
   `Attention.forward()`. This is a hint that the original code was, at
   some point, instrumented with a debugging/visualization decorator
   (commonly from a small third-party library used to record and later
   plot attention maps) that has since been disabled by commenting it
   out. It has no effect on the model's actual behavior or numerical
   output as currently written, and can be safely ignored unless you
   specifically want to re-enable attention-map visualization (in which
   case you would need to import and use whatever `get_local` utility the
   original project relied on).
==============================================================================
"""

from einops import rearrange
from torch import nn, einsum


def pair(t):
    """
    Utility function: if `t` is already a tuple, return it unchanged;
    otherwise, wrap it into a two-item tuple `(t, t)`. See explanation #9
    near the top of this file for why this pattern is useful (allowing a
    caller to pass either one number or an explicit (height, width)-style
    pair for a size-related argument).
    """
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """
    Wraps another layer (`fn`) so that its input is normalized (via
    LayerNorm) FIRST, before being passed into `fn`. See explanation #6
    near the top of this file for why this "Pre-Norm" arrangement is used.

    Parameters
    ----------
    dim : int
        The size of the feature vector for each item in the sequence
        (needed so LayerNorm knows what size to normalize over).
    fn : nn.Module
        The sub-layer to wrap -- in this file, either an `Attention` or a
        `FeedForward` instance (see the `Transformer` class below).
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Normalizes `x`, then runs the wrapped sub-layer `fn` on the
        normalized result. `**kwargs` simply forwards along any extra
        keyword arguments the wrapped layer's own `forward()` method might
        need (none of the sub-layers used in this file actually require
        any, but this keeps `PreNorm` generically reusable for any future
        sub-layer that might).
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    The position-wise "MLP" block applied independently to every sequence
    item. See explanation #7 near the top of this file for the full
    plain-language walkthrough.

    Parameters
    ----------
    dim : int
        Input/output feature size per sequence item.
    hidden_dim : int
        Size of the internal expanded ("hidden") representation -- for
        Transformers this is usually LARGER than `dim` (a common choice
        is 4x `dim`), giving the network more room to compute
        intermediate representations before compressing back down.
    dropout : float
        Dropout probability applied after the activation and again after
        the final linear layer (see the Dropout explanation in
        `graph_attention_layer.py` earlier in this project).
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # `nn.Sequential` simply chains these layers together, running
        # each one's output straight into the next one's input, in order.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),   # expand: dim -> hidden_dim
            nn.GELU(),                    # smooth nonlinearity (see explanation #7)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),   # contract back: hidden_dim -> dim
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Runs the whole expand -> activate -> contract pipeline on `x`."""
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention -- the central computation of a Transformer.
    See explanations #1-#4 and #8 near the top of this file for the full
    plain-language, step-by-step walkthrough of exactly what this class
    computes and why.

    Parameters
    ----------
    dim : int
        Input/output feature size per sequence item.
    heads : int
        How many parallel attention "heads" to compute (see explanation
        #1 above for why multiple heads are useful).
    dim_head : int
        The size of the Query/Key/Value vectors used WITHIN each
        individual head.
    dropout : float
        Dropout probability applied to this layer's final output
        projection (only relevant when `project_out` is True -- see
        explanation #8 above).
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        # Total combined size across all heads put together
        # (heads * dim_head) -- this is the size Query/Key/Value vectors
        # get projected into internally, before being split apart into
        # individual heads.
        inner_dim = dim_head * heads

        # See explanation #8 above: skip the final output-projection
        # layer entirely in the specific edge case where there's only one
        # head and its dimension already matches `dim` exactly.
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        # The "scaled" part of "scaled dot-product attention" -- see
        # explanation #2 above.
        self.scale = dim_head ** -0.5

        # Softmax turns raw attention scores into a proper probability
        # distribution (summing to 1) over "which other item should I pay
        # attention to". `dim=-1` normalizes across the LAST dimension of
        # whatever tensor gets passed in, which (as used in forward()
        # below) corresponds to the "key position" (`j`) dimension --
        # i.e., for every query position, its attention weights across
        # all possible key positions sum to 1.
        self.attend = nn.Softmax(dim=-1)

        # One single combined linear layer computes Query, Key, AND Value
        # projections all at once (three times `inner_dim` worth of
        # output), which then get split apart in forward() below. This is
        # a common efficiency trick -- mathematically equivalent to three
        # separate linear layers, but only requires one matrix
        # multiplication instead of three.
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Final output projection, back down from the combined
        # multi-head size to the model's regular working dimension `dim`
        # -- skipped (replaced with a no-op `nn.Identity()`) in the
        # special edge case described in explanation #8 above.
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    # @get_local('attn')
    def forward(self, x):
        """
        Runs the full multi-head self-attention computation over the
        input sequence `x`. See explanations #1, #3, and #4 near the top
        of this file for the complete plain-language walkthrough of every
        step below.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, sequence_length, dim) -- a batch of
            sequences of feature vectors.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, sequence_length, dim) -- the
            attention-updated feature vector for every item in every
            sequence.
        """
        # Unpack the input's shape: `b` = batch size, `n` = sequence
        # length, and (the underscore `_`) the feature dimension `dim`
        # (not needed by name here). `h` = number of attention heads.
        b, n, _, h = *x.shape, self.heads

        # Project `x` into combined Query+Key+Value space in one go, then
        # split that single big tensor into three equal pieces along the
        # last dimension: `q`, `k`, and `v`.
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # Reshape each of q, k, v from "one long combined-heads vector per
        # item" into "one separate vector per head, per item" -- see
        # explanation #3 above for exactly how this `rearrange` pattern
        # works.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Compute the raw attention score between EVERY pair of sequence
        # positions (query position `i`, key position `j`), for every
        # batch and every head at once, then apply the scaling factor
        # from explanation #2 above.
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Turn raw scores into a proper attention-weight distribution
        # (summing to 1 across key positions `j`, for every query
        # position `i`).
        attn = self.attend(dots)
        # print(attn.shape)
        # quit()

        # Combine every position's Value vectors, weighted by the
        # attention weights just computed, producing one updated output
        # vector per query position `i`.
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Reshape back from "separate vector per head, per item" into "one
        # long combined-heads vector per item" -- the exact reverse of
        # the earlier rearrange step (see explanation #3 above).
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Apply the final output projection (or the no-op Identity, in
        # the special single-head edge case from explanation #8 above).
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Stacks `depth` repetitions of (Pre-Norm Attention + Pre-Norm
    FeedForward), each wrapped in its own residual/skip connection. This
    is the standard "Transformer encoder" arrangement. See explanations
    #5 and #6 near the top of this file for why residual connections and
    Pre-Norm are used.

    Parameters
    ----------
    dim : int
        Feature size per sequence item, kept CONSTANT throughout every
        layer of the stack (residual connections require the input and
        output of each block to be the same shape, so they can simply be
        added together).
    depth : int
        How many (Attention + FeedForward) blocks to stack.
    heads : int
        Number of attention heads used in every Attention block.
    dim_head : int
        Size of each individual attention head's Query/Key/Value vectors.
    mlp_dim : int
        Hidden-layer size used inside every FeedForward block (see
        `FeedForward`'s `hidden_dim` parameter above).
    dropout : float
        Dropout probability, used inside both the Attention and
        FeedForward sub-layers.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        # `nn.ModuleList` is just like a plain Python list, except
        # PyTorch properly tracks every module placed inside it (their
        # parameters get included in `.parameters()`, moved correctly by
        # `.to(device)`, saved correctly by `.state_dict()`, etc.) -- see
        # the same concern raised about `add_module()` in
        # `graph_attention_layer.py` earlier in this project; a plain
        # Python list would NOT provide this automatic tracking.
        self.layers = nn.ModuleList([])

        # Build `depth` identical (but independently-weighted) pairs of
        # (Attention, FeedForward) sub-layers, each already wrapped in
        # `PreNorm`.
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Runs `x` through every stacked (Attention, FeedForward) pair in
        order, adding a residual/skip connection around EACH one (see
        explanation #5 near the top of this file).
        """
        for attn, ff in self.layers:
            # Residual connection around the attention sub-layer: the
            # ORIGINAL `x` is added back on top of the attention output.
            x = attn(x) + x
            # Residual connection around the feed-forward sub-layer,
            # applied to the (already attention-updated) `x` from the
            # line above.
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """
    A thin wrapper that adds embedding-level Dropout in front of a
    `Transformer` stack. NOTE: unlike a textbook-complete Vision
    Transformer, this particular class does NOT itself include any
    patch-embedding, positional-encoding, or classification-token logic --
    see reproducibility note #5 near the top of this file. It expects to
    receive an already-prepared sequence of feature vectors as its input.

    Parameters
    ----------
    dim : int
        Feature size per sequence item.
    depth : int
        How many Transformer blocks to stack (see the `Transformer` class
        above).
    heads : int
        Number of attention heads per Transformer block.
    mlp_dim : int
        Hidden-layer size inside every FeedForward block.
    dim_head : int
        Size of each individual attention head's Query/Key/Value vectors
        (default: 64, a very common choice in Transformer literature).
    dropout : float
        Dropout probability used inside the Transformer's Attention and
        FeedForward sub-layers.
    emb_dropout : float
        Dropout probability applied directly to the raw input embeddings,
        BEFORE they enter the Transformer stack at all -- a small amount
        of extra regularization right at the input.
    """

    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # `nn.Identity()` here is a placeholder "do nothing" layer -- kept
        # for interface compatibility with a more complete ViT
        # implementation, where `to_latent` might otherwise perform some
        # additional final transformation before the model's output is
        # used elsewhere (e.g. by a classification head). As written
        # here, it simply passes its input straight through unchanged.
        self.to_latent = nn.Identity()

    def forward(self, x):
        """
        Applies embedding dropout, runs the Transformer stack, then
        passes the result through the (currently no-op) `to_latent`
        layer.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, sequence_length, dim) -- see
            reproducibility note #5 near the top of this file for what
            kind of "already-embedded" sequence this is expected to be in
            this project's full pipeline.
        """
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        return x
