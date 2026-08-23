"""
==============================================================================
omni_dimensional_dynamic_conv.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file implements "ODConv" (Omni-Dimensional Dynamic Convolution), a
drop-in replacement for an ordinary `nn.Conv2d` layer that can be MUCH more
expressive, at only a small extra computational cost. It is written in
PyTorch.

Background concept #1 -- What is an ordinary convolution, and what's its
limitation?
    A normal convolution layer learns ONE fixed set of filter weights during
    training, and then applies that exact same fixed set of weights to
    every single input image it ever sees, forever (at inference/prediction
    time). This is efficient, but it means the layer cannot adapt its
    behavior based on what a specific input actually looks like -- an image
    of a cat and an image of a car get filtered with exactly the same
    weights.

Background concept #2 -- What is "Dynamic Convolution"?
    Dynamic Convolution is a family of techniques where, instead of
    learning just one fixed filter, the layer learns SEVERAL candidate
    filters (here, `kernel_num` of them), and then, for every individual
    input, computes a small "attention" score for each candidate filter
    and combines them into ONE final filter that is customized to that
    specific input, before running the actual convolution. This lets the
    layer behave differently for different kinds of inputs, without
    needing a hugely bigger fixed filter.

Background concept #3 -- What does "Omni-Dimensional" add on top of that?
    Most earlier dynamic-convolution methods only computed ONE attention
    score per candidate filter (deciding "how much to use each whole
    candidate filter"). ODConv instead computes FOUR separate, more
    fine-grained kinds of attention, covering every dimension a
    convolution filter has:
      1. Which INPUT CHANNELS matter most right now ("channel attention").
      2. Which OUTPUT CHANNELS/FILTERS matter most right now
         ("filter attention").
      3. Which SPATIAL POSITIONS within the kernel matter most right now
         ("spatial attention" -- e.g. is the center of a 3x3 filter more
         important than its corners, for this particular input?).
      4. Which of the `kernel_num` CANDIDATE KERNELS matter most right now
         ("kernel attention").
    Combining attention along all four of these dimensions at once is why
    this technique is called "Omni-Dimensional" (omni = "all").

------------------------------------------------------------------------------
KEY ALGORITHMS/CONCEPTS USED IN THIS FILE (explained in plain language)
------------------------------------------------------------------------------

1) The `Attention` class -- computing all four attention signals
   ------------------------------------------------------------------
   This is a small side-network (much smaller than the main convolution
   itself) whose only job is: "look at the input feature map, and output
   four attention signals". It works like this:
     a) `self.avgpool = nn.AdaptiveAvgPool2d(1)`: "Global Average
        Pooling" -- squashes the entire height x width feature map for
        every channel down to a SINGLE number per channel (its average
        value). This throws away exact spatial detail on purpose, keeping
        only "roughly how strongly active is each channel, overall" --
        this is the classic "Squeeze" step popularized by
        Squeeze-and-Excitation networks (Hu et al., 2018).
     b) `self.fc` (a 1x1 convolution) + `self.bn` (BatchNorm) + `self.relu`:
        a small bottleneck that shrinks the pooled channel vector down to
        a smaller `attention_channel` size and applies a non-linearity --
        this is the "Excitation" step's first half: learn a compact,
        input-dependent summary before branching out into the four
        separate attention heads below.
     c) Four separate small 1x1-convolution "heads" then each read from
        that shared compact summary and produce one of the four attention
        signals described in background concept #3 above:
          - `channel_fc`  -> `get_channel_attention()`
          - `filter_fc`   -> `get_filter_attention()`
          - `spatial_fc`  -> `get_spatial_attention()`
          - `kernel_fc`   -> `get_kernel_attention()`
        Channel and filter attention use a `sigmoid` (squashes each value
        independently into a 0-1 "how much to keep this channel/filter"
        gate). Kernel attention uses a `softmax` instead (forces the
        attention across the `kernel_num` candidate kernels to add up to
        1, since exactly one blended combination of candidates should be
        chosen, similar to a soft "voting" between candidates).

2) The "temperature" parameter
   ---------------------------------
   Both the `sigmoid` and `softmax` calls above first DIVIDE their raw
   input by `self.temperature` before applying the activation. A high
   temperature makes the resulting attention values more "spread out"/
   uniform (harder for the network to strongly prefer one option), while
   a low temperature (closer to the default of 1.0) makes the attention
   values sharper/more decisive (closer to fully picking one option,
   letting some attention values approach 0 or 1). Some training recipes
   for ODConv gradually LOWER the temperature over the course of training
   (starting soft/uniform, ending sharp/decisive) using the
   `update_temperature()` method provided here -- that schedule itself is
   NOT implemented in this file (it lives in whatever separate training
   script uses this model); this file only exposes the knob to set it.

3) The `skip` "no-op" optimization
   ------------------------------------
   `Attention.skip()` is a placeholder function that simply always
   returns the constant number `1.0` and does no computation at all.
   Three separate situations replace one of the four attention heads with
   this simple `skip` shortcut instead of a real learned attention head,
   because computing that particular kind of attention would be
   pointless in those specific situations:
     - If this is a DEPTHWISE convolution (`in_planes == groups and
       in_planes == out_planes`, meaning every input channel gets exactly
       one dedicated output channel, similar to the depthwise convolution
       explained in `capsule_network_model.py` earlier in this project),
       there is no real notion of "choosing between output filters"
       (there's exactly one filter per channel already), so FILTER
       attention is skipped.
     - If `kernel_size == 1` (a "pointwise" convolution -- a filter that
       only ever looks at a single pixel position, with no spatial
       extent), there is no meaningful "which spatial position in the
       kernel matters most" question to ask (there's only one position),
       so SPATIAL attention is skipped.
     - If `kernel_num == 1` (only one candidate kernel exists at all),
       there is nothing to "choose between", so KERNEL attention is
       skipped.
   Skipping these unnecessary computations saves parameters and compute
   time without losing any real expressive power in those specific cases.

4) Building one INPUT-SPECIFIC convolution filter, then applying it
   ---------------------------------------------------------------------
   `ODConv2d._forward_impl_common()` is where the four attention signals
   actually get used. Step by step:
     a) `x = x * channel_attention`: rather than multiplying attention
        into the FILTER WEIGHTS (which would be the most literal
        interpretation of "channel attention"), the code instead
        multiplies it directly into the INPUT feature map. The code
        comment explains this is mathematically equivalent, but runs
        faster and uses less GPU memory in practice.
     b) `aggregate_weight = spatial_attention * kernel_attention *
        self.weight.unsqueeze(dim=0)`, then summed over the `kernel_num`
        candidates (`torch.sum(..., dim=1)`): this is the heart of
        "dynamic convolution" -- it blends together all `kernel_num`
        learned candidate filters into ONE combined filter, weighted by
        how much attention this particular input assigns to each
        candidate kernel (kernel attention) and to each spatial position
        within the kernel (spatial attention). The result is one
        brand-new, custom-built filter, generated fresh for every single
        input example.
     c) Running the convolution with a DIFFERENT filter per example in
        the batch is awkward for a standard convolution operation (which
        normally expects the exact same filter to be used for every
        example). The code works around this using a well-known trick:
        it reshapes the whole batch so that every example's channels get
        stacked one after another into a single, much bigger "combined"
        image (`x.reshape(1, -1, height, width)`), and uses PyTorch's
        `groups` argument (`groups=self.groups * batch_size`) to make sure
        each example's channels are only ever convolved with THAT
        example's own custom filter, never mixing between examples. This
        clever reshape-and-group trick lets one single, ordinary
        `F.conv2d` call efficiently perform a fully-custom-per-example
        convolution, without needing a slow Python loop over the batch.
     d) `output = output * filter_attention`: finally, output-channel
        ("filter") attention is applied directly to the convolution's
        result, gating (scaling) how much each output channel contributes
        overall.

5) A fast-path shortcut for the simplest case (`_forward_impl_pw1x`)
   -----------------------------------------------------------------------
   When `kernel_size == 1` AND `kernel_num == 1` (a plain pointwise
   convolution with just one candidate filter, and no spatial or kernel
   attention to combine — see the `skip` optimization above), all of the
   complicated per-example filter-blending machinery in step 4 above
   becomes unnecessary: there's only one candidate kernel to use, so no
   blending is needed at all. `_forward_impl_pw1x()` handles this simpler
   case directly and more efficiently, applying channel attention to the
   input, running one ordinary shared convolution, then applying filter
   attention to the output. Which of the two forward implementations
   actually gets used is decided ONCE, up front in `__init__`, and stored
   as `self._forward_impl` -- so no extra "if" check is needed on every
   single forward pass afterward.

6) Kaiming ("He") weight initialization, with `mode='fan_out'`
   -------------------------------------------------------------------
   As explained in `capsule_network_model.py` earlier in this project,
   Kaiming initialization picks good starting random weights for
   ReLU-based networks. The `mode='fan_out'` setting used here (instead of
   the more common default `mode='fan_in'`) specifically preserves the
   magnitude of signals as they flow BACKWARD through the network during
   training (useful because this network's candidate filters get
   recombined dynamically at every forward pass, so keeping the
   initialization scale consistent from the OUTPUT side is a common
   choice for this kind of dynamic-filter architecture).

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: torch (PyTorch). No other dependencies.

2. This file only defines the LAYER architecture -- it does not load data
   or run a training loop itself. To reproduce a specific trained model
   you additionally need the exact training script, dataset, and random
   seed used, PLUS (if your original training run used a temperature
   annealing schedule) the exact temperature schedule that calls
   `update_temperature()` over the course of training.

3. Because weight initialization uses random numbers, fix the random seed
   BEFORE creating the model if you want the exact same starting weights
   every run:
       import torch
       torch.manual_seed(0)
       layer = ODConv2d(in_planes=..., out_planes=..., kernel_size=...)

4. This layer is a drop-in replacement for `nn.Conv2d` -- it accepts the
   same core arguments (`in_planes`/`out_planes` instead of
   `in_channels`/`out_channels`, `kernel_size`, `stride`, `padding`,
   `dilation`, `groups`), plus two new ODConv-specific arguments:
   `reduction` (how aggressively to shrink the attention bottleneck size,
   default 0.0625 = 1/16th) and `kernel_num` (how many candidate filters
   to learn and dynamically blend, default 4).

5. For fully bit-for-bit reproducible GPU training (optional, and usually
   somewhat slower), also add:
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
==============================================================================
"""

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    The small "side-network" that looks at an input feature map and
    produces the four ODConv attention signals (channel, filter, spatial,
    kernel). See explanations #1, #2, and #3 near the top of this file for
    a full plain-language walkthrough.

    Parameters
    ----------
    in_planes : int
        Number of input channels the parent ODConv2d layer expects.
    out_planes : int
        Number of output channels the parent ODConv2d layer produces.
    kernel_size : int
        Size (height/width) of the parent layer's convolution filter.
    groups : int
        Number of groups used by the parent layer's convolution (relevant
        for detecting the depthwise-convolution special case below).
    reduction : float
        Fraction used to shrink `in_planes` down to a smaller internal
        "attention_channel" bottleneck size (default 0.0625 = 1/16).
    kernel_num : int
        How many candidate filters the parent layer will dynamically
        blend between.
    min_channel : int
        A floor/minimum value for the internal bottleneck size, so it
        never becomes unreasonably tiny even for very small `in_planes`.
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()

        # The internal "squeeze" bottleneck size: shrink in_planes down by
        # the `reduction` fraction, but never below `min_channel`.
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        # Starting temperature (see explanation #2 above). 1.0 means
        # "neutral" -- no extra softening or sharpening applied yet.
        self.temperature = 1.0

        # ---- Shared "squeeze and reduce" trunk (steps a & b from
        # explanation #1 above) ----
        # Global Average Pooling: (batch, channels, height, width) ->
        # (batch, channels, 1, 1) -- one average value per channel.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # A 1x1 convolution here acts exactly like a small per-channel
        # Linear/Dense layer, shrinking the channel count down to the
        # smaller `attention_channel` bottleneck size.
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU()

        # ---- Channel attention head (always present) ----
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        # ---- Filter attention head (skipped for depthwise convolutions) ----
        # A depthwise convolution (groups == in_planes == out_planes) maps
        # each input channel to exactly one dedicated output channel, so
        # there's no real "choice between output filters" to attend over.
        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        # ---- Spatial attention head (skipped for 1x1 "pointwise" convs) ----
        # A 1x1 filter has only one spatial position, so there's nothing
        # meaningful to attend over spatially.
        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # ---- Kernel attention head (skipped if there's only 1 candidate) ----
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes every convolution layer in this attention sub-network
        with Kaiming/He initialization (see explanation #6 near the top
        of this file), and every BatchNorm layer to its standard "neutral
        start" (scale=1, shift=0). `self.modules()` automatically walks
        through every layer defined above, so this loop applies correctly
        no matter which of the optional attention heads were actually
        created.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        """
        Lets an external training script adjust the temperature value
        used inside every sigmoid/softmax attention computation (see
        explanation #2 near the top of this file). Typically called
        periodically during training as part of a temperature-annealing
        schedule (not implemented in this file).
        """
        self.temperature = temperature

    @staticmethod
    def skip(_):
        """
        The "do nothing" placeholder used when a particular attention
        type doesn't apply (see explanation #3 near the top of this
        file). Always returns the plain number 1.0, regardless of input --
        multiplying anything by 1.0 leaves it completely unchanged, which
        is exactly the desired "no attention effect" behavior.
        """
        return 1.0

    def get_channel_attention(self, x):
        """
        Produces one attention "gate" value (between 0 and 1, via
        sigmoid) per INPUT channel, describing how much each input
        channel should be emphasized or suppressed for this specific
        input.
        """
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        """
        Produces one attention "gate" value (between 0 and 1, via
        sigmoid) per OUTPUT channel/filter, describing how much each
        output filter should be emphasized or suppressed for this
        specific input.
        """
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        """
        Produces one attention "gate" value (between 0 and 1, via
        sigmoid) for EACH position within the kernel_size x kernel_size
        convolution filter, describing which spatial positions in the
        filter matter most for this specific input.
        """
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        """
        Produces one attention weight per CANDIDATE KERNEL (via softmax,
        so all `kernel_num` weights add up to 1 for each example),
        describing how much to rely on each of the several learned
        candidate filters for this specific input.
        """
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        """
        Runs the shared "squeeze and reduce" trunk once, then computes
        all four attention signals (or their `skip` = 1.0 placeholder)
        from that shared summary, and returns all four together.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, in_planes, height, width) -- the raw input
            feature map to the parent ODConv2d layer.

        Returns
        -------
        tuple of 4 torch.Tensor (or float 1.0 for any skipped type)
            (channel_attention, filter_attention, spatial_attention,
            kernel_attention)
        """
        x = self.avgpool(x)   # squeeze: (batch, in_planes, H, W) -> (batch, in_planes, 1, 1)
        x = self.fc(x)        # reduce channel count to the attention bottleneck size
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    """
    Omni-Dimensional Dynamic Convolution layer -- a drop-in replacement
    for `nn.Conv2d` that dynamically builds a custom filter for every
    input example by blending together `kernel_num` learned candidate
    filters, guided by four kinds of attention computed by the `Attention`
    class above. See the full explanation near the top of this file for
    the complete step-by-step walkthrough.

    Parameters
    ----------
    in_planes : int
        Number of input channels (same meaning as `in_channels` in a
        normal `nn.Conv2d`).
    out_planes : int
        Number of output channels (same meaning as `out_channels` in a
        normal `nn.Conv2d`).
    kernel_size : int
        Size (height/width) of the convolution filter.
    stride, padding, dilation, groups : int
        Same meaning as the identically-named arguments in a normal
        `nn.Conv2d`.
    reduction : float
        Passed through to the internal `Attention` module -- controls how
        aggressively the attention bottleneck shrinks the channel count.
    kernel_num : int
        How many separate candidate filters to learn and dynamically
        blend between for every input.
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num

        # The attention sub-network described in the `Attention` class
        # above, configured to match this layer's exact shape.
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)

        # The actual learnable convolution weights: `kernel_num` SEPARATE
        # complete filter sets, each shaped exactly like a normal
        # `nn.Conv2d`'s weight tensor would be
        # (out_planes, in_planes/groups, kernel_size, kernel_size). These
        # get dynamically blended together at every forward pass,
        # weighted by the spatial and kernel attention signals (see
        # explanation #4 near the top of this file).
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        # Decide ONCE, here in __init__ (not on every forward call), which
        # forward-pass implementation to use: the simpler/faster shortcut
        # when there's nothing to dynamically blend (1x1 kernel AND only
        # one candidate filter -- see explanation #5 near the top of the
        # file), or the general-purpose implementation otherwise.
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        """
        Initializes each of the `kernel_num` candidate filter sets with
        Kaiming/He initialization (see explanation #6 near the top of the
        file), one candidate at a time.
        """
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        """
        Forwards a new temperature value down into the internal
        `Attention` module (see `Attention.update_temperature` above).
        """
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        """
        The general-purpose forward pass, used whenever there is real
        blending work to do (more than one candidate kernel, and/or a
        kernel size bigger than 1x1). See explanation #4 near the top of
        this file for the full step-by-step walkthrough of exactly what
        happens here.
        """
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()

        # Apply channel attention directly to the INPUT (see explanation
        # #4a above for why this is done here instead of on the weights).
        x = x * channel_attention

        # Reshape-the-whole-batch-into-one-big-image trick (see
        # explanation #4c above): stack every example's channels
        # end-to-end into a single "batch of 1" image, so a single
        # grouped convolution call can apply a DIFFERENT filter per
        # original example.
        x = x.reshape(1, -1, height, width)

        # Blend together all `kernel_num` candidate filters, weighted by
        # spatial attention (which position in the kernel matters) and
        # kernel attention (which candidate kernel matters), then sum
        # across the candidate-kernel dimension to get ONE final filter
        # per example (see explanation #4b above).
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        # Run ONE grouped convolution across the whole reshaped "batch of
        # 1" image. `groups=self.groups * batch_size` is what makes sure
        # each original example's channels only ever get convolved with
        # THAT example's own custom-built filter (see explanation #4c).
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)

        # Reshape the result back into a normal
        # (batch, out_planes, height, width) tensor.
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))

        # Apply output-channel ("filter") attention to the final result
        # (see explanation #4d above).
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        """
        The simplified/faster forward pass used only when
        `kernel_size == 1` and `kernel_num == 1` (see explanation #5 near
        the top of this file) -- there is exactly one candidate filter
        and no spatial dimension to blend over, so this skips the whole
        per-example filter-blending machinery from
        `_forward_impl_common()` above and just runs one ordinary shared
        convolution.
        """
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        # `self.weight.squeeze(dim=0)` removes the (now-redundant, since
        # kernel_num == 1) candidate-kernel dimension, leaving a normal
        # (out_planes, in_planes/groups, 1, 1) convolution weight tensor.
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        """
        Runs whichever forward-pass implementation was chosen in
        `__init__` (see the `_forward_impl` assignment above).
        """
        return self._forward_impl(x)
