"""
==============================================================================
gene_expression_inference.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file contains the "inference" (also called "prediction") code for this
project: it takes an ALREADY-TRAINED model (built from the pieces defined in
`capsule_network_model.py`, `graph_attention_layer.py`,
`omni_dimensional_dynamic_conv.py`, etc., earlier in this project) and a
"test" dataset (built from `spatial_gene_expression_dataset.py`), and uses
the model to actually PREDICT gene expression values from tissue images --
this is the whole point of the project coming together.

This file does NOT train anything. Training a neural network means
repeatedly showing it examples and adjusting its weights to reduce its
mistakes; inference/prediction means using a model whose weights are
already fixed/finished, and simply running new inputs through it to see
what it outputs.

Background concept #1 -- Why does a prediction function need a completely
different code path than a training function?
    During training, the model needs to remember extra bookkeeping
    information (called "gradients") so it can later figure out how to
    adjust its weights. During inference, none of that bookkeeping is
    needed -- we just want the model's raw output, as fast and
    memory-efficiently as possible. PyTorch lets you explicitly tell it
    "I'm not training right now" in two ways, both used in this file:
      - `model.eval()`: switches certain special layers (like
        `BatchNorm` and `Dropout`, both used elsewhere in this project's
        model layers) into their "inference behavior" -- for example,
        Dropout stops randomly zeroing out values (see
        `graph_attention_layer.py` for a full explanation of Dropout),
        and BatchNorm switches from using each individual batch's
        statistics to using the fixed statistics it accumulated across
        the whole training run.
      - `with torch.no_grad():`: tells PyTorch not to bother tracking
        gradients at all for anything inside this block, since we won't
        need to run backpropagation. This significantly speeds up
        computation and reduces memory use.

Background concept #2 -- What is "AnnData", and why store predictions in it?
    `AnnData` (imported here as `ann`, via the `from utils import *` line,
    and used as `ann.AnnData`) is a standard Python data container from the
    `anndata` library, extremely widely used throughout single-cell and
    spatial genomics analysis (it is also the core data structure behind
    the popular `scanpy` toolkit used earlier in
    `spatial_gene_expression_dataset.py`). An `AnnData` object bundles
    together:
      - a main data matrix (here: rows = tissue spots, columns = genes --
        exactly the same layout as the raw gene-expression tables loaded
        in `spatial_gene_expression_dataset.py`),
      - `.obsm` ("observation-matrix" annotations): extra per-row
        (per-spot) information that isn't a single number, such as each
        spot's spatial (x, y) coordinates -- this is exactly what
        `adata.obsm['spatial'] = ct` stores below.
    Packaging predictions into this same standard format (instead of just
    returning a plain array of numbers) means the predictions can
    immediately be plugged into any of the many existing genomics analysis
    and visualization tools built to work with `AnnData` objects, without
    any extra conversion work.

This file defines two separate prediction functions:
    1. model_predict() -- normal-resolution prediction, for spots where we
                           DO have real measured gene expression to compare
                           against (used to evaluate how accurate the
                           model's predictions are).
    2. sr_predict()     -- "super-resolution" prediction, for the densely
                           spaced VIRTUAL spot grid described in
                           `spatial_gene_expression_dataset.py`'s `sr=True`
                           mode, where there is no real measured gene
                           expression to compare against (used only to
                           generate a fine-grained prediction map, not to
                           measure accuracy).

------------------------------------------------------------------------------
KEY CONCEPTS/ALGORITHMS USED IN THIS FILE (explained in plain language)
------------------------------------------------------------------------------

1) Batching, and why predictions need to be "stitched back together"
   ---------------------------------------------------------------------
   `test_loader` (a PyTorch `DataLoader`, built from a `Dataset` class
   like the ones in `spatial_gene_expression_dataset.py`) doesn't hand
   over the WHOLE test set at once -- it hands it over in smaller pieces
   ("batches"), one batch at a time, inside the `for ... in tqdm(...)`
   loop. Both functions in this file build up ONE big combined result by
   repeatedly using `torch.cat(...)` to "glue" each new batch's results
   onto the end of everything collected so far. The very first batch is
   handled slightly differently (it just becomes the starting point,
   since there's nothing to glue onto yet), which is why you see an
   `if preds is None: ... else: ...` pattern in both functions.

2) `tqdm`
   ----------
   `tqdm(test_loader)` simply wraps the data loader so that a live
   progress bar is printed to the screen while looping through it,
   showing how many batches have been processed so far and an estimated
   time remaining. It has no effect whatsoever on the actual predictions
   -- it exists purely to give a human watching the screen visual
   feedback that the program is working and roughly how much longer it
   will take.

3) Moving data between CPU and GPU, and between PyTorch and NumPy
   ---------------------------------------------------------------------
   - `.to(device)`: PyTorch tensors and models must live on the same
     "device" (either the CPU, or a GPU/CUDA device) to be used together.
     `device` is a function argument here (defaulting to
     `torch.device('cpu')`), so the SAME code works whether you're running
     on a machine with a GPU or not -- just pass in `torch.device('cuda')`
     to use a GPU instead, if one is available.
   - `.cpu()`: moves a tensor back onto the CPU (necessary before
     converting it to a NumPy array, since NumPy doesn't understand
     GPU-resident data).
   - `.numpy()`: converts a PyTorch tensor into a plain NumPy array (the
     data format `AnnData` and most other genomics-analysis libraries
     expect).
   - `.squeeze()`: removes any dimensions of size 1 from a tensor's
     shape. For example, a tensor of shape (1, 500, 32) would become
     (500, 32) -- this is used here to clean up an extra "batch of 1" or
     similar leftover dimension that isn't actually meaningful data.

4) `torch.no_grad()` and memory efficiency
   ------------------------------------------
   As explained in background concept #1 above, this context manager
   turns off PyTorch's gradient-tracking bookkeeping for everything
   inside the `with` block. This is why prediction/inference code is
   almost always noticeably faster and uses less memory than training
   code processing the same amount of data.

------------------------------------------------------------------------------
DIFFERENCES BETWEEN THE TWO FUNCTIONS IN THIS FILE
------------------------------------------------------------------------------
`model_predict()`:
  - Expects each batch from `test_loader` to contain 5 items: `patch`
    (image patches), `position` (grid coordinates), `exp` (the REAL,
    measured gene expression -- the "ground truth" to compare
    predictions against), `center` (pixel coordinates), and `adj` (the
    k-nearest-neighbor spot graph -- see `spot_knn_graph_builder.py`).
    This matches exactly what a non-super-resolution
    `spatial_gene_expression_dataset.py` test split returns.
  - The model itself is called as `model(patch, position, adj)` -- i.e.
    this model architecture expects a graph (`adj`) as one of its
    inputs, meaning it very likely includes a Graph Attention layer (see
    `graph_attention_layer.py`) somewhere inside it.
  - Returns TWO AnnData objects: one holding the model's PREDICTIONS, and
    one holding the REAL ground-truth expression -- both tagged with the
    same spatial coordinates, so they can be directly compared
    spot-by-spot afterward (e.g. to compute a correlation or error
    metric).

`sr_predict()`:
  - Expects each batch to contain only 3 items: `patch`, `position`, and
    `center` -- no `exp` (no real measured expression) and no `adj`
    (no graph) -- matching exactly what `sr=True` mode returns in
    `spatial_gene_expression_dataset.py` (recall: in super-resolution
    mode there is no real ground truth for the made-up virtual spots, and
    that mode does not build/use a k-NN graph at all).
  - The model is called as `model(patch, position)` -- with NO adjacency
    graph passed in -- suggesting this function is meant to be used with
    a (possibly different / graph-free) model variant designed
    specifically for the super-resolution use case.
  - Returns only ONE AnnData object (the predictions) -- there is no
    ground truth available to return alongside it in this mode.

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: torch, tqdm, anndata (imported here as `ann`
   via `from utils import *` -- make sure `utils.py` actually does
   something like `import anndata as ann` internally), plus whatever
   `utils.py` itself needs.

2. To reproduce a specific prediction run exactly:
     - Use the EXACT SAME trained model weights (load them with
       `model.load_state_dict(torch.load(checkpoint_path))` before calling
       either function here -- note the `MODEL_PATH = ''` placeholder near
       the top of this file is currently EMPTY/unused; if your original
       workflow loaded a checkpoint from a specific path, make sure to
       fill this in and actually use it, or pass an already-loaded model
       into these functions directly).
     - Use the EXACT SAME test dataset / test_loader, with the SAME
       `fold` number (see `spatial_gene_expression_dataset.py`'s
       leave-one-sample-out explanation) and with shuffling turned OFF
       (a test/prediction DataLoader should never shuffle -- otherwise
       the order of `preds`/`ct`/`gt` won't line up consistently between
       runs, even though the individual predicted VALUES would still be
       identical).
     - Always call `model.eval()` before predicting (both functions
       already do this for you), so Dropout/BatchNorm behave
       deterministically instead of randomly.
     - Run on the same `device` (CPU vs GPU can sometimes produce tiny
       floating-point differences in the least-significant digits, though
       results should otherwise match).

3. This file calls `warnings.filterwarnings('ignore')` at import time,
   which silences ALL Python warning messages for the remainder of the
   program (not just warnings from this file). This can be convenient to
   avoid noisy log output, but be aware it will also hide any genuinely
   useful warnings (e.g. about deprecated function usage) if you are
   debugging an unrelated problem elsewhere in your pipeline.
==============================================================================
"""

import warnings

import torch
from tqdm import tqdm

# `from utils import *` brings in every public name defined in `utils.py`
# (including, presumably, `ann` -- the `anndata` library import used below
# as `ann.AnnData`). Because this file does not show `utils.py` itself, if
# you see a `NameError: name 'ann' is not defined` when running this file,
# check that `utils.py` actually does `import anndata as ann` (or similar)
# and is importable from wherever you run this script.
from utils import *

# Silences all Python warning messages project-wide from this point
# onward -- see the reproducibility note above about the trade-off this
# involves.
warnings.filterwarnings('ignore')

# Placeholder for a path to a saved model checkpoint file. It is currently
# EMPTY and not actually used anywhere else in this file -- see the
# reproducibility notes above for how you would normally use a variable
# like this (e.g. `model.load_state_dict(torch.load(MODEL_PATH))`) before
# calling the prediction functions below.
MODEL_PATH = ''


def model_predict(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    """
    Runs a trained model over an entire (normal-resolution) test set and
    packages the results into two AnnData objects: one holding the
    model's PREDICTED gene expression, and one holding the REAL, measured
    ("ground truth") gene expression -- both tagged with the same spatial
    coordinates, ready for side-by-side comparison. See the big
    explanation near the top of this file for full background.

    Parameters
    ----------
    model : torch.nn.Module
        An already-TRAINED model (its weights should already be loaded
        before calling this function). Must accept
        `model(patch, position, adj)` as its calling convention -- i.e.
        it expects image patches, grid positions, AND a k-NN spot
        adjacency graph as input (see `spot_knn_graph_builder.py`).
    test_loader : torch.utils.data.DataLoader
        Wraps a test-split Dataset (such as `ViT_HER2ST(train=False, ...)`
        from `spatial_gene_expression_dataset.py`) and yields batches of
        `(patch, position, exp, center, adj)` tuples.
    adata : (unused parameter)
        Accepted for interface flexibility, but not actually read or
        modified anywhere in this function's current implementation --
        this function always builds brand-new AnnData objects from
        scratch instead. Kept here unchanged for compatibility with
        whatever code elsewhere in this project calls this function.
    attention : bool (unused parameter)
        Also accepted but not currently used inside this function's body.
        Likely intended as a hook for a future/alternate code path (e.g.
        to also return attention weights from the model), kept unchanged
        here for interface compatibility.
    device : torch.device
        Which device (CPU or GPU) to run the model on. Defaults to CPU so
        this function works out-of-the-box even on a machine without a
        GPU.

    Returns
    -------
    (adata, adata_gt) : tuple of two anndata.AnnData objects
        `adata`    -- the model's predicted gene expression values.
        `adata_gt` -- the real, measured ("ground truth") gene expression
                      values, for comparison.
        Both have their `.obsm['spatial']` set to the same spot pixel
        coordinates, so predictions and ground truth can be matched up
        spot-by-spot afterward.
    """
    # Switch the model into "inference mode" (see background concept #1
    # near the top of this file) and make sure it lives on the requested
    # device.
    model.eval()
    model = model.to(device)

    # Will accumulate predictions across every batch (see explanation #1
    # near the top of the file for the "stitch batches together" pattern).
    preds = None

    # Turn off gradient tracking for everything below -- we are only
    # doing forward passes, never backpropagation, so this saves memory
    # and computation time (see background concept #1 above).
    with torch.no_grad():
        # `tqdm(test_loader)` just adds a progress bar; the actual
        # looping behavior is identical to `for ... in test_loader`.
        for patch, position, exp, center, adj in tqdm(test_loader):
            # Move this batch's tensors onto the requested device. Note:
            # `exp` (ground truth) and `center` are intentionally NOT
            # moved to `device` here, since they are only used for
            # bookkeeping/output (building the AnnData objects) and never
            # passed into the model itself.
            patch, position, adj = patch.to(device), position.to(device), adj.to(device)

            # Run the model forward to get this batch's predictions.
            pred = model(patch, position, adj)

            if preds is None:
                # First batch: just use it as the starting point.
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                # Every subsequent batch: glue its results onto the end
                # of everything collected so far.
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
                gt = torch.cat((gt, exp), dim=0)  #

    # Move every collected tensor back to CPU, drop any leftover
    # size-1 dimensions, and convert to plain NumPy arrays -- the format
    # AnnData (and most other genomics-analysis tooling) expects.
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()

    # Package the PREDICTED gene expression into an AnnData object (rows
    # = spots, columns = genes), and attach each spot's spatial pixel
    # coordinates as extra per-spot annotation (see background concept #2
    # near the top of this file).
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    # Do the exact same thing for the REAL, measured gene expression, so
    # it can be compared against the predictions above, spot-by-spot.
    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt


def sr_predict(model, test_loader, device=torch.device('cpu')):
    """
    Runs a trained model over a "super-resolution" test set (built using
    `sr=True` in `spatial_gene_expression_dataset.py`) and packages the
    predictions into a single AnnData object. Unlike `model_predict()`
    above, there is no ground-truth gene expression available here (the
    virtual super-resolution spots were never actually measured in the
    lab), so only predictions are returned. See the big explanation near
    the top of this file for full background, including exactly how this
    function differs from `model_predict()`.

    Parameters
    ----------
    model : torch.nn.Module
        An already-TRAINED model. Must accept `model(patch, position)` as
        its calling convention -- note this is only TWO inputs (no
        adjacency graph), unlike `model_predict()`'s three-input model
        above.
    test_loader : torch.utils.data.DataLoader
        Wraps a super-resolution test-split Dataset (such as
        `ViT_HER2ST(train=False, sr=True, ...)` from
        `spatial_gene_expression_dataset.py`) and yields batches of
        `(patch, position, center)` tuples.
    device : torch.device
        Which device (CPU or GPU) to run the model on. Defaults to CPU.

    Returns
    -------
    adata : anndata.AnnData
        The model's predicted gene expression values for every virtual
        super-resolution spot, with `.obsm['spatial']` set to each
        virtual spot's pixel coordinates.
    """
    model.eval()
    model = model.to(device)
    preds = None

    with torch.no_grad():
        for patch, position, center in tqdm(test_loader):
            # Only `patch` and `position` are actually fed into the
            # model; `center` is kept aside purely for building the
            # output AnnData's spatial coordinates afterward.
            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)

            if preds is None:
                preds = pred.squeeze()
                ct = center
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)

    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()

    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    return adata
