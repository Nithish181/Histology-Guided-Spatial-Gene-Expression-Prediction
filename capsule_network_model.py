"""
==============================================================================
capsule_network_model.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file defines a neural network architecture called a "Capsule Network"
(often shortened to "CapsNet"), specifically a lightweight version of it
known in the research literature as "Efficient-CapsNet". It is written using
PyTorch, a popular Python deep learning library.

Background concept #1 -- What problem is a Capsule Network trying to solve?
    A normal Convolutional Neural Network (CNN) is very good at detecting
    WHETHER a feature (like an eye, a wheel, an edge) is present somewhere
    in an image, using single numbers ("scalars") to represent each
    detected feature's strength. The problem: a plain CNN throws away a lot
    of information about HOW that feature looks -- its exact rotation,
    size, thickness, skew, etc. -- because it only keeps one number per
    feature. It can also get confused if features are present but arranged
    in the wrong spatial relationship (a famous example: a CNN might still
    recognize a "face" even if the eyes, nose, and mouth are jumbled into
    the wrong positions, because it doesn't track *how the parts relate to
    each other*).

    A Capsule Network fixes this by representing each detected feature not
    as a single number, but as a small VECTOR (a short list of numbers,
    e.g. 8 numbers instead of 1). This is called a "capsule". The vector's
    LENGTH (its overall magnitude) represents "how confident are we that
    this feature exists", while the vector's DIRECTION (the relative sizes
    of its individual numbers) represents "what pose/style/orientation does
    this feature have". Capsules at one layer then "vote" for capsules at
    the next layer, and votes that agree with each other (point in similar
    directions) get combined more strongly -- this voting process is why
    the technique is sometimes called "routing by agreement".

Background concept #2 -- What is "Efficient-CapsNet"?
    The original Capsule Network idea (Sabour et al., 2017) used an
    iterative routing algorithm that is somewhat slow and heavy on
    parameters. "Efficient-CapsNet" (the architecture this file
    implements) is a lighter-weight redesign that:
      (a) uses a "depthwise separable convolution" (explained below) to
          build the first layer of capsules cheaply, and
      (b) replaces the iterative routing procedure with a single-pass,
          self-attention-style routing computation (see RoutingLayer
          below), which is much faster while keeping the same core idea
          of "let capsules vote and combine their votes by agreement".

This file defines, from the ground up:
    1. squash()            -- the special activation function used to turn
                               any vector into a "capsule" (see below).
    2. length()             -- computes how "confident"/"present" each
                               capsule is (used to decide which capsule
                               corresponds to the predicted class).
    3. mask()               -- zeroes out every capsule except the
                               "winning" one (used when a follow-up
                               reconstruction network needs to be told
                               "reconstruct the image assuming THIS was
                               the detected class").
    4. PrimaryCapsLayer     -- turns ordinary CNN feature maps into the
                               very first layer of capsules.
    5. RoutingLayer         -- takes primary capsules and "routes" them
                               (via the efficient self-attention-like
                               mechanism) into a smaller number of
                               higher-level capsules (e.g. one capsule per
                               possible class, for a classification task).
    6. EfficientCapsNet     -- the full model: a small stack of ordinary
                               convolutional layers (a normal CNN "feature
                               extractor" backbone) feeding into the
                               PrimaryCapsLayer and then the RoutingLayer.

------------------------------------------------------------------------------
KEY ALGORITHMS/CONCEPTS USED IN THIS FILE (explained in plain language)
------------------------------------------------------------------------------

1) The "squash" activation function
   -------------------------------------
   Ordinary neural network layers often use activation functions like ReLU
   or sigmoid on individual numbers. Capsules are VECTORS, so they need a
   vector-aware activation function instead. "Squash" takes a vector and:
     - keeps its DIRECTION unchanged (so the "pose" information the vector
       encodes is preserved), and
     - rescales its LENGTH to always be between 0 and 1 (so the length can
       be interpreted as a probability: "how sure are we this feature/
       class is present").
   Mathematically, for a vector `v` with length `n = ||v||`:
       squash(v) = (1 - 1/(e^n + eps)) * (v / (n + eps))
   As n grows large, the multiplier (1 - 1/(e^n + eps)) approaches 1, so
   long vectors stay almost the same length (near-certain detections).
   As n shrinks toward 0, the multiplier shrinks toward 0 too, so short
   vectors get squashed even smaller (uncertain/absent detections get
   suppressed). `eps` is a tiny constant added purely to avoid dividing by
   zero.

2) Depthwise separable convolution (used inside PrimaryCapsLayer)
   ---------------------------------------------------------------
   A normal `nn.Conv2d` layer mixes information ACROSS all input channels
   to produce each output channel -- this is powerful but has many
   parameters (weights to learn). A "depthwise" convolution (created here
   by passing `groups=in_channels` to `nn.Conv2d`) instead applies a
   SEPARATE, independent small filter to each input channel on its own,
   never mixing channels together. This is much cheaper (fewer parameters,
   faster to compute) -- which is exactly why the "Efficient" in
   "Efficient-CapsNet" uses it here, instead of the more expensive capsule
   construction used in the original CapsNet paper.

3) Primary capsules (PrimaryCapsLayer)
   ------------------------------------
   After the depthwise convolution produces a stack of feature maps, this
   layer simply RESHAPES that stack of numbers into a set of small
   vectors -- each vector becomes one "capsule" -- and then applies
   `squash()` to each one. No new learnable weights are introduced by the
   reshape itself; the only trainable part here is the depthwise
   convolution weights.

4) "Routing by agreement" -- efficient/self-attention style (RoutingLayer)
   --------------------------------------------------------------------------
   This is the most important algorithm in the file, so we go step by
   step through `RoutingLayer.forward()`:
     a) `u = einsum("...ji,kjiz->...kjz", input, self.W)`
        Every incoming ("primary") capsule `j` casts a "vote" for every
        outgoing ("higher-level") capsule `k`, by being multiplied through
        a learned weight matrix `W`. Think of this as: "if this really is
        feature j, what would the pose of higher-level concept k look
        like?" -- one predicted vote vector per (incoming capsule,
        outgoing capsule) pair.
     b) `c = einsum("...ij,...kj->...i", u, u)[..., None]`
        For every outgoing capsule, this measures how much all of that
        capsule's incoming votes AGREE with each other (their dot-product
        similarity, summed up). This is the "routing coefficient" --
        votes that closely resemble each other (i.e. many primary
        capsules "agree" on this higher-level capsule's pose) get a
        bigger weight.
     c) `c = c / sqrt(dim_capsules)` then `c = softmax(c, dim=1)`
        The dividing step keeps the numbers in a reasonable numeric range
        (this "scaled dot product" trick is the same idea used in
        Transformer self-attention). Softmax then turns the raw agreement
        scores into proper weights that add up to 1 across the routing
        dimension -- i.e. "how much attention should each outgoing
        capsule pay to its votes, relative to the alternatives".
     d) `c = c + self.b`
        Adds a learned per-capsule bias term (a small adjustable offset,
        similar to the bias term in an ordinary Dense/Linear layer).
     e) `s = sum(u * c, dim=-2)`
        Combines (weighted-sums) all the votes for each outgoing capsule
        using the agreement weights just computed -- capsules whose votes
        agreed strongly contribute more to the final result.
     f) `return squash(s)`
        Finally squashes the combined vector so its length again
        represents a clean 0-1 "how confident are we this higher-level
        capsule/class is present" probability.
   Compared to the original CapsNet's routing (which repeats a similar
   "vote -> agree -> combine" loop several times, iteratively refining the
   routing weights), this Efficient-CapsNet version computes everything in
   a SINGLE pass (no iteration/loop), which is what makes it fast.

5) Convolutional "feature extractor" backbone (inside EfficientCapsNet)
   -----------------------------------------------------------------------
   Before any capsules are built, the raw input first passes through 4
   ordinary convolution layers (conv1..conv4), each followed by Batch
   Normalization and a ReLU activation:
     - Convolution layer: slides small learnable filters across the input
       to detect local visual patterns (edges, textures, simple shapes).
     - Batch Normalization: rescales the outputs of each layer so their
       average is near 0 and their spread is consistent, which generally
       makes training faster and more stable.
     - ReLU ("Rectified Linear Unit"): a simple activation function that
       replaces every negative number with 0 and leaves positive numbers
       unchanged -- this is what lets the network learn non-linear
       (i.e. more complex than a straight line) patterns.
   This stack is a completely standard CNN feature extractor; the capsule
   machinery only kicks in AFTER these 4 layers.

6) Kaiming ("He") weight initialization
   ------------------------------------------
   Before training starts, a neural network's weights must be filled with
   some SOME starting numbers. Kaiming initialization is a specific
   mathematically-derived way of picking those starting random numbers so
   that signals don't explode or vanish as they pass through many ReLU
   layers -- it is the standard recommended initialization for networks
   built from ReLU-activated convolution/linear layers, which is why it is
   used here for every conv layer and for the routing layer's weight `W`.

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: torch (PyTorch). No other dependencies.

2. This file only defines the MODEL architecture -- it does not load data
   or run a training loop itself. To reproduce a specific trained model,
   you additionally need:
     - the exact training script (loss function, optimizer, learning
       rate, number of epochs),
     - the exact dataset and preprocessing used,
     - and the same random seed used to initialize weights.

3. Because weight initialization (Kaiming init) uses random numbers, you
   must fix the random seed BEFORE creating the model if you want the
   exact same starting weights every run, e.g.:
       import torch
       torch.manual_seed(0)
       model = EfficientCapsNet(rout_capsules=10, route_dim=16)
   (Use whatever `rout_capsules` / `route_dim` values match your original
   experiment -- see the docstring on the `EfficientCapsNet` class below
   for what these two numbers mean.)

4. For fully bit-for-bit reproducible GPU training (optional, and usually
   somewhat slower), also add:
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

5. Input shape note: `conv1` expects an input with 16 channels
   (`in_channels=16`). This is unusual for a typical image (photos
   normally have 1 grayscale or 3 RGB channels), so this model is most
   likely meant to be plugged in AFTER some other feature-producing
   module (or on data that has already been transformed into a
   16-channel representation) rather than directly on a raw photo. Keep
   this in mind when preparing whatever input tensor you feed into
   `EfficientCapsNet.forward()`.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(input, eps=10e-21):
    """
    The capsule-network "squashing" nonlinearity (see explanation #1 near
    the top of this file).

    Parameters
    ----------
    input : torch.Tensor
        A tensor whose LAST dimension is treated as one capsule's vector
        (there can be many capsules at once -- one per "row" -- this
        function processes them all in parallel).
    eps : float
        A tiny number added to denominators purely to avoid a
        divide-by-zero error if a vector's length happens to be exactly 0.

    Returns
    -------
    torch.Tensor
        Same shape as `input`, but every capsule vector now has a length
        between 0 and 1, while pointing in the same direction as before.
    """
    # `n` = the length (Euclidean/L2 norm) of each capsule vector.
    # `keepdim=True` keeps the reduced dimension as size 1 instead of
    # removing it, so the later division broadcasts correctly.
    n = torch.norm(input, dim=-1, keepdim=True)

    # The actual squashing formula: shrinks length into the (0, 1) range
    # while keeping direction unchanged (dividing by `n` normalizes the
    # vector to length 1, then the left-hand multiplier rescales it back
    # down to the desired squashed length).
    return (1 - 1 / (torch.exp(n) + eps)) * (input / (n + eps))


def length(input):
    """
    Computes the LENGTH of each capsule vector, which (after squashing)
    can be read as "how confident is the network that this capsule's
    feature/class is present" -- a value between 0 and 1.

    Parameters
    ----------
    input : torch.Tensor
        A tensor whose last dimension is one capsule's vector.

    Returns
    -------
    torch.Tensor
        Same shape as `input` but with the last dimension removed --
        one plain number (the vector's length) per capsule.
    """
    # Standard Euclidean length: sqrt(sum of squares). `1e-8` avoids
    # a numerically unstable gradient exactly at length == 0.
    return torch.sqrt(torch.sum(input ** 2, dim=-1) + 1e-8)


def mask(input):
    """
    Zeroes out every capsule except one "winning" capsule per example,
    and flattens the result into a single vector. This is typically used
    when a Capsule Network has an additional RECONSTRUCTION sub-network
    attached after it (not included in this file) -- that sub-network
    needs to be told exactly which single capsule's pose vector to
    reconstruct an image from, with every other capsule's information
    hidden (zeroed out).

    Parameters
    ----------
    input : torch.Tensor or [torch.Tensor, torch.Tensor]
        Either:
          - just the capsule tensor (shape: batch x num_capsules x
            dim_capsules), in which case this function figures out the
            "winning" capsule itself (the one with the largest length,
            i.e. the model's own top predicted class), OR
          - a two-item list [capsule_tensor, mask_tensor] where you
            explicitly supply which capsule to keep for each example
            (useful during training, where you may want to force-reveal
            the TRUE class's capsule rather than the model's current best
            guess).

    Returns
    -------
    torch.Tensor
        Shape: (batch_size, num_capsules * dim_capsules) -- one flattened
        vector per example, containing zeros everywhere except the
        entries belonging to the single "winning" capsule.
    """
    if type(input) is list:
        # Caller explicitly provided which capsule to keep (a one-hot-style
        # mask), for example the ground-truth label during training.
        input, mask = input
    else:
        # No mask was given -> use the model's own prediction: find, for
        # every example, which capsule has the LARGEST vector length (the
        # class the model is currently most confident about), and build a
        # one-hot mask (a vector of all 0s except a single 1 at that
        # capsule's position) from it.
        x = torch.sqrt(torch.sum(input ** 2, dim=-1))
        mask = F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).float()

    # Broadcast-multiply: every capsule gets multiplied by either 0 (not
    # the winning capsule) or 1 (the winning capsule), zeroing out all the
    # non-winning capsules' numbers.
    masked = input * mask.unsqueeze(-1)

    # Flatten from (batch, num_capsules, dim_capsules) down to
    # (batch, num_capsules * dim_capsules) -- one long vector per example,
    # ready to be fed into a following fully-connected reconstruction
    # network.
    return masked.view(input.shape[0], -1)


class PrimaryCapsLayer(nn.Module):
    """
    Builds the very first layer of capsules directly from ordinary CNN
    feature maps (see explanation #3 near the top of this file).

    Parameters
    ----------
    in_channels : int
        Number of channels coming IN to this layer (from the previous
        convolution layer).
    kernel_size : int
        Size (height/width) of the sliding convolution filter.
    num_capsules : int
        How many separate capsules to produce.
    dim_capsules : int
        How many numbers make up each individual capsule's vector.
    stride : int
        How many pixels the convolution filter moves at each step
        (default 1 = move one pixel at a time).
    """

    def __init__(self, in_channels, kernel_size, num_capsules, dim_capsules, stride=1):
        super(PrimaryCapsLayer, self).__init__()

        # A DEPTHWISE convolution: setting groups=in_channels means each
        # input channel is convolved completely separately from every
        # other channel (no mixing across channels) -- see explanation #2
        # near the top of this file for why this is used here.
        # padding="valid" means NO padding is added around the input, so
        # the output will be slightly smaller than the input (standard
        # behavior when you don't want any border padding).
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding="valid",
        )
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def forward(self, input):
        """
        Runs the depthwise convolution, then reshapes ("reinterprets") its
        output numbers into a batch of capsule vectors, then squashes
        them.
        """
        output = self.depthwise_conv(input)

        # Reshape from a normal (batch, channels, height, width)
        # convolution output into (batch, num_capsules, dim_capsules) --
        # i.e. we simply relabel groups of numbers as "capsule vectors"
        # instead of thinking of them as a spatial feature map anymore.
        output = output.view(output.size(0), self.num_capsules, self.dim_capsules)

        # Apply the squashing nonlinearity described above, so each
        # capsule's vector length becomes a clean 0-1 "presence"
        # probability.
        return squash(output)


class RoutingLayer(nn.Module):
    """
    Implements the efficient, single-pass "routing by agreement" algorithm
    described in detail in explanation #4 near the top of this file. Takes
    in a set of lower-level ("primary") capsules and produces a smaller set
    of higher-level capsules (for example, one capsule per output class).

    Parameters
    ----------
    num_capsules : int
        How many OUTPUT (higher-level) capsules to produce -- e.g. set
        this to the number of classes for a classification task.
    dim_capsules : int
        How many numbers make up each output capsule's vector.
    """

    def __init__(self, num_capsules, dim_capsules):
        super(RoutingLayer, self).__init__()

        # `self.W` is the learned "voting" weight tensor. Its shape
        # (num_capsules, 16, 8, dim_capsules) means:
        #   - num_capsules : one weight block per OUTPUT capsule
        #   - 16           : expects exactly 16 INPUT (primary) capsules
        #                    (this must match PrimaryCapsLayer's
        #                    num_capsules, since votes are cast FROM each
        #                    of those primary capsules)
        #   - 8            : expects each input capsule to have exactly 8
        #                    numbers (must match PrimaryCapsLayer's
        #                    dim_capsules)
        #   - dim_capsules : size of the OUTPUT capsule vector this weight
        #                    block votes for
        self.W = nn.Parameter(torch.Tensor(num_capsules, 16, 8, dim_capsules))

        # `self.b` is a small learned bias added to the routing/agreement
        # scores for each output capsule (one bias value per input capsule
        # "slot", broadcast during the addition in forward()).
        self.b = nn.Parameter(torch.zeros(num_capsules, 16, 1))

        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the learnable weight `W` using Kaiming/He
        initialization (see explanation #6 near the top of the file), and
        the bias `b` to all zeros (a very common, safe default -- letting
        the routing weights themselves do all the initial "work" while the
        bias starts out neutral).
        """
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, input):
        """
        Runs the full efficient routing-by-agreement computation. See
        explanation #4 near the top of this file for a full plain-English
        walkthrough of each line below.
        """
        # Step (a): every input capsule casts one "vote" per output
        # capsule, using the learned weight tensor W.
        u = torch.einsum("...ji,kjiz->...kjz", input, self.W)

        # Step (b): measure how much all the votes FOR one output capsule
        # agree with each other (their pairwise similarity, summed).
        c = torch.einsum("...ij,...kj->...i", u, u)[..., None]

        # Step (c): scale down the raw agreement scores (this "scaled
        # dot-product" trick, borrowed from Transformer self-attention,
        # keeps the softmax numerically stable), then turn them into
        # proper routing WEIGHTS that add up to 1.
        c = c / torch.sqrt(torch.Tensor([self.dim_capsules]).type(torch.FloatTensor))
        c = torch.softmax(c, axis=1)

        # Step (d): add the learned bias.
        c = c + self.b

        # Step (e): combine every vote using its routing weight -- votes
        # that agreed strongly with the group contribute more.
        s = torch.sum(torch.mul(u, c), dim=-2)

        # Step (f): squash the combined vector back into a valid capsule
        # (length between 0 and 1).
        return squash(s)


class EfficientCapsNet(nn.Module):
    """
    The complete Efficient-CapsNet model: a small ordinary-CNN "feature
    extractor" backbone, followed by a PrimaryCapsLayer (turns CNN features
    into the first capsules) and a RoutingLayer (routes those into the
    final, higher-level output capsules -- e.g. one per class).

    Parameters
    ----------
    rout_capsules : int
        Number of OUTPUT capsules produced by the final RoutingLayer --
        for a classification task, set this to the number of classes
        (e.g. 10 for digit classification 0-9).
    route_dim : int
        Number of numbers in each output capsule's vector (a typical
        choice in capsule-network papers is 16).
    """

    def __init__(self, rout_capsules, route_dim):
        super(EfficientCapsNet, self).__init__()

        # ---- Ordinary CNN "feature extractor" backbone ----
        # Four convolution layers, each followed by Batch Normalization
        # (see explanation #5 near the top of the file). Channel counts
        # grow from 16 -> 32 -> 64 -> 64 -> 128 as we go deeper, which is
        # a common pattern: earlier layers look at raw, simple patterns
        # with fewer channels, later layers combine those into richer,
        # more numerous higher-level patterns.
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding="valid")
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding="valid")
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, padding="valid")
        self.batch_norm3 = nn.BatchNorm2d(64)

        # `stride=2` here means this convolution also DOWNSAMPLES the
        # feature map (skips every other position), roughly halving its
        # height and width -- a common way to progressively shrink the
        # spatial size while increasing channel depth.
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding="valid")
        self.batch_norm4 = nn.BatchNorm2d(128)

        # ---- Capsule layers ----
        # Builds 16 primary capsules, each with 8 numbers, straight out of
        # the 128-channel feature map produced by the backbone above.
        self.primary_caps = PrimaryCapsLayer(in_channels=128, kernel_size=9, num_capsules=16, dim_capsules=8)

        # Routes those 16 primary capsules into `rout_capsules` final,
        # higher-level output capsules, each with `route_dim` numbers.
        self.digit_caps = RoutingLayer(num_capsules=rout_capsules, dim_capsules=route_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes every convolution layer's weights with Kaiming/He
        initialization (see explanation #6 near the top of the file). Bias
        terms are left at PyTorch's own default initialization (small
        random values), and BatchNorm layers keep PyTorch's defaults
        (scale=1, shift=0) since re-initializing those is not typically
        necessary.
        """
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)

    def forward(self, x):
        """
        Defines how data flows through the whole model, from raw input
        `x` all the way to the final output capsules.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped (batch_size, 16, height, width) -- see
            the "Input shape note" in the file's top docstring: this
            model expects 16 input channels, not a typical 1- or 3-channel
            image.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, rout_capsules, route_dim) -- the final
            output capsules. To turn these into class probabilities, pass
            this result through the `length()` function defined above
            (the capsule with the largest length is the predicted class).
        """
        # Each block below is: convolution -> batch normalization -> ReLU
        # activation, repeated 4 times (the standard CNN backbone).
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = torch.relu(self.batch_norm4(self.conv4(x)))

        # Turn the final CNN feature map into the first layer of capsules.
        x = self.primary_caps(x)

        # Route those primary capsules into the final output capsules.
        x = self.digit_caps(x)

        return x
