"""
==============================================================================
train_and_evaluate_thitogene.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This is the "orchestration" file that ties the ENTIRE project together. It
does not define any new neural network building blocks itself -- instead it:
  1. Loads training/test data using the Dataset classes from
     `spatial_gene_expression_dataset.py` (imported here as `ViT_HER2ST`,
     `ViT_SKIN`).
  2. Builds the full combined model, `THItoGene` (imported from a separate
     file, `vis_model.py`, not shown yet in this project -- this is
     presumably the model that actually STACKS TOGETHER the pieces built
     earlier in this project: the CapsNet-style capsule layers from
     `capsule_network_model.py`, the Graph Attention layer from
     `graph_attention_layer.py`, and the Omni-Dimensional Dynamic
     Convolution from `omni_dimensional_dynamic_conv.py`, using the k-NN
     spot graph from `spot_knn_graph_builder.py`).
  3. TRAINS that model (using the `PyTorch Lightning` framework, explained
     below) on one fold of the data at a time.
  4. TESTS each trained fold's model on its held-out sample, and computes
     how accurately the model predicted real gene expression, using
     Pearson correlation (explained below).
  5. Repeats this entire process once per possible test fold, implementing
     the "leave-one-sample-out cross-validation" scheme first introduced in
     `spatial_gene_expression_dataset.py`.

Background concept #1 -- What is "PyTorch Lightning", and why use it instead
of writing a plain PyTorch training loop by hand?
    Ordinary PyTorch training requires writing a fair amount of repetitive
    "boilerplate" code yourself: looping over epochs, looping over
    batches, calling `optimizer.zero_grad()`, computing the loss, calling
    `.backward()`, calling `optimizer.step()`, moving tensors to the
    correct device, logging progress, saving checkpoints, etc.
    "PyTorch Lightning" (imported here as `pl`) is a popular framework
    that handles ALL of that repetitive plumbing for you automatically,
    as long as your model class follows a specific expected structure
    (defining methods like `training_step()`, `configure_optimizers()`,
    and `predict_step()` -- these are defined inside the `THItoGene` class
    in `vis_model.py`, not in this file). This file interacts with
    Lightning through just two objects:
      - `pl.Trainer`: handles the actual training loop, GPU placement,
        logging, and more, once you call `trainer.fit(model, train_loader)`.
      - `CSVLogger`: tells the Trainer to write training progress/metrics
        out to a plain CSV file on disk (as opposed to, say, TensorBoard
        or Weights & Biases logging, which Lightning also supports).

Background concept #2 -- What is Pearson correlation, and why use it here?
    Pearson correlation is a standard statistics measure of how strongly
    two lists of numbers move together in a straight-line ("linear")
    relationship. It ranges from -1 (perfectly opposite -- when one number
    goes up, the other always goes down) through 0 (no linear
    relationship at all) to +1 (perfectly matching -- when one number goes
    up, the other always goes up by a proportional amount). Here, for
    every individual GENE, the model's PREDICTED expression values (across
    all test spots) are compared against the REAL, measured expression
    values for that same gene, and a Pearson correlation is computed
    between the two. A value near +1 for a gene means the model is doing a
    great job of tracking that gene's real expression pattern across the
    tissue; a value near 0 means the model's predictions for that gene are
    essentially unrelated to reality. Averaging this correlation across
    ALL genes (`np.nanmean(R)`, ignoring any genes where the correlation
    couldn't be computed, e.g. `NaN` values from a gene with zero variance)
    gives one single overall accuracy number for the whole model. The
    actual correlation math itself lives in a function called `get_R()`,
    imported here via `from utils import *` (not shown in this project
    yet).

Background concept #3 -- Reproducibility via random seeds
    Deep learning training involves MANY sources of randomness: how model
    weights are initialized (see the Kaiming/Xavier initialization
    explanations in earlier files of this project), which order training
    examples are shuffled into (`shuffle=True` in the DataLoader below),
    and how Dropout randomly zeroes out values (see
    `graph_attention_layer.py`). Fixing ("seeding") the random number
    generators used by every relevant library, as this file's
    `if __name__ == '__main__':` block does right at the start, is
    ESSENTIAL if you want a training run to be reproducible -- i.e. to
    produce the exact same trained weights and results if you run the
    exact same code again later, or on a different machine.

------------------------------------------------------------------------------
KEY ALGORITHMS/CONCEPTS USED IN THIS FILE (explained in plain language)
------------------------------------------------------------------------------

1) Leave-one-sample-out cross-validation, applied across the WHOLE pipeline
   -----------------------------------------------------------------------
   This file is where the `fold` concept introduced in
   `spatial_gene_expression_dataset.py` actually gets used in a full
   loop: `for i in range(0, 32): train(i, ...)` trains 32 COMPLETELY
   SEPARATE models -- one per possible held-out HER2ST sample -- each
   time training on the other 31 samples and testing only on sample `i`.
   This is a thorough (if computationally expensive) way to evaluate how
   well the model generalizes to tissue it has never seen, since every
   single sample gets a turn being the "unseen" test case exactly once.

2) The `DataLoader` arguments used here
   ------------------------------------------
   - `batch_size=1`: each training/testing step processes exactly ONE
     tissue sample (one whole slide, containing many spots) at a time --
     recall from `spatial_gene_expression_dataset.py` that
     `Dataset.__getitem__()` already returns ALL of one sample's spots at
     once, so a "batch" here effectively means "one slide's worth of
     spots".
   - `num_workers=0`: data loading happens in the SAME process as the
     main training loop, rather than in extra parallel background worker
     processes. Setting this to a higher number can speed up data loading
     on machines with multiple CPU cores, at the cost of some
     complexity/reproducibility subtleties, which is likely why it's
     conservatively left at 0 here.
   - `shuffle=True` (training only): examples are served in a random
     order each epoch, which generally helps training generalize better
     (the model doesn't memorize a fixed example order). Test/prediction
     loaders below correctly leave `shuffle` at its default `False`, since
     the ORDER of test predictions needs to line up consistently with the
     ground truth and spatial coordinates afterward.

3) The `THItoGene` model's hyperparameters, and how they connect to
   earlier files in this project
   -----------------------------------------------------------------------
   `THItoGene(n_genes=..., learning_rate=..., route_dim=64, caps=20,
   heads=[16, 8], n_layers=...)` -- while `THItoGene` itself is defined in
   a separate file (`vis_model.py`), these argument NAMES map naturally
   onto concepts already explained earlier in this project:
     - `n_genes`: how many genes the model should predict expression
       values for (e.g. 785 for the HER2ST dataset's top-785 gene list,
       matching the gene list loaded in `spatial_gene_expression_dataset.py`).
     - `route_dim`: very likely the `dim_capsules` argument for the
       model's final `RoutingLayer` (see `capsule_network_model.py`).
     - `caps`: very likely the number of capsules produced somewhere in
       the capsule-network portion of the model.
     - `heads`: very likely the number of attention heads for one or more
       `MultiHeadGAT` graph-attention blocks (see
       `graph_attention_layer.py`) -- note this is a LIST (`[16, 8]`),
       suggesting the full model actually uses multiple stacked graph
       attention blocks with different head-counts at different depths.
     - `n_layers`: how many times some repeating block (very likely
       including the graph-attention / capsule components) is stacked
       inside the model -- notice this is different for the two datasets
       (4 layers for `her2st`, 8 for `skin`), meaning the two dataset
       configurations use genuinely different model depths/capacities.

4) `sc.pp.scale()` applied to predictions before scoring
   -----------------------------------------------------------
   In the `test()` function, after getting the model's raw predictions,
   `sc.pp.scale(adata_pred)` (from the `scanpy` library, also used back
   in `spatial_gene_expression_dataset.py`) rescales every gene's
   predicted values to have mean 0 and standard deviation 1 across all
   spots (the same "z-score" style normalization explained as an OPTIONAL
   `norm=True` step in the dataset file -- here it is applied
   unconditionally to predictions before scoring). This does not change
   Pearson correlation values at all (Pearson correlation is
   mathematically unaffected by this kind of rescaling), but can matter
   for downstream visualization/interpretation of the predicted values
   themselves.

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: torch, pytorch_lightning, numpy, scanpy (as
   `sc`), plus whatever `utils.py` and `vis_model.py` themselves need.

2. Random seeding: this file ALREADY does the right thing at the very
   start of its `if __name__ == '__main__':` block -- it seeds Python's
   built-in `random` module, NumPy, and PyTorch's CPU AND GPU random
   number generators, all with the same fixed `seed = 0`. Keep this
   exactly as-is (or change the number, but keep ALL FOUR lines together)
   if you want reproducible weight initialization and reproducible
   training-data shuffling.

3. GPU device note: the original code hard-coded
   `pl.Trainer(accelerator="gpu", devices=[7], ...)`, pinning training to
   GPU index 7 specifically -- this would fail outright on any machine
   without at least 8 GPUs. This has been changed (see `train()` below)
   to `accelerator="auto", devices="auto"`, which lets PyTorch Lightning
   automatically detect and use whatever GPU is available (or fall back
   to CPU if none is), so this script now runs out-of-the-box on any
   machine. If you specifically want to pin training to one particular
   GPU on a multi-GPU machine, set `devices=[N]` for whichever index `N`
   you want.

4. IMPORTANT reproducibility gotcha -- checkpoint saving is currently
   DISABLED: notice the line
       # trainer.save_checkpoint(modelsave_address+"/"+"last_train_"+tagname+'_'+str(test_sample_ID)+".ckpt")
   inside `train()` is COMMENTED OUT. This means, as currently written,
   `train()` does NOT explicitly save a checkpoint file to the exact path
   that `test()` later tries to load
   (`model_address + "/last_train_" + tagname + ... + ".ckpt"` for
   her2st). PyTorch Lightning may still auto-save its OWN default
   checkpoint somewhere under a `lightning_logs/` folder depending on
   your Lightning version/configuration, but that automatic path will
   NOT match the path `test()` expects. To reproduce a full train-then-test
   run successfully, you should either:
       (a) uncomment that `trainer.save_checkpoint(...)` line so a
           checkpoint actually gets written to the path `test()` expects, or
       (b) explicitly pass a `ModelCheckpoint` callback into `pl.Trainer`
           configured to write to that same expected path.
   This has been left commented out here, unchanged, to exactly match your
   uploaded code -- but you likely need to fix this before `test()` will
   successfully find a checkpoint to load.

5. Hard-coded, machine-specific file path: inside `test()`, the `else`
   branch (for the `"skin"` dataset) loads its gene list from an absolute
   path:
       '/home/user/jiayuran/code/cond/THItoGene/data/skin_hvg_cut_1000.npy'
   This path is specific to the original author's own computer and will
   almost certainly NOT exist on yours. If you plan to actually run the
   `skin` branch of `test()`, change this to a path that exists on your
   own machine (for consistency, probably
   `'./data/skin_hvg_cut_1000.npy'`, matching how the `her2st` branch
   right above it loads its own gene list). This has been left unchanged
   here to exactly match your uploaded code.

6. Checkpoint filename mismatch between datasets: for `"her2st"`, `test()`
   looks for a file named `"last_train_" + tagname + ...`, but for the
   `else` (skin) branch, it looks for a DIFFERENT prefix,
   `"THItoGene_" + tagname + ...`. Make sure whatever checkpoint-saving
   code you actually use (see point 4 above) writes out files under
   whichever exact name `test()` will later look for, for each dataset.
==============================================================================
"""

# coding:utf-8
import random

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# `ViT_HER2ST` and `ViT_SKIN` are the Dataset classes defined in
# `spatial_gene_expression_dataset.py` earlier in this project.
# NOTE: originally `from dataset import ViT_HER2ST, ViT_SKIN` -- updated
# to match this project's renamed dataset file.
from spatial_gene_expression_dataset import ViT_HER2ST, ViT_SKIN

# `model_predict` is the inference helper function defined in
# `gene_expression_inference.py` earlier in this project.
# NOTE: originally `from predict import model_predict` -- updated to
# match this project's renamed inference file.
from gene_expression_inference import model_predict

# `from genomics_analysis_utils import *` brings in helper functions/
# objects used below, including `get_R` (the Pearson correlation
# computation), `np` (numpy), and `sc` (scanpy). NOTE: originally
# `from utils import *` -- updated to match this project's renamed
# utilities file.
from genomics_analysis_utils import *

# `THItoGene` is the full, combined model architecture, assembling the
# CapsNet, Graph Attention, Omni-Dimensional Dynamic Convolution, and
# Transformer building blocks from earlier files in this project into one
# model. NOTE: originally `from vis_model import THItoGene` -- updated to
# match this project's renamed model-assembly file.
from thitogene_full_model import THItoGene


def train(test_sample_ID, vit_dataset, epochs, modelsave_address, dataset_name):
    """
    Trains ONE `THItoGene` model on all samples EXCEPT `test_sample_ID`
    (leave-one-sample-out cross-validation -- see explanation #1 near the
    top of this file), then immediately evaluates it on the held-out
    `test_sample_ID` and prints its average Pearson correlation score.

    Parameters
    ----------
    test_sample_ID : int
        Which sample index to hold out as this run's TEST sample (passed
        straight through as the `fold` argument to the Dataset classes
        from `spatial_gene_expression_dataset.py`).
    vit_dataset : type
        Which Dataset CLASS to use -- either `ViT_HER2ST` or `ViT_SKIN`
        (both from `spatial_gene_expression_dataset.py`). Note this is the
        class ITSELF (not an already-created instance), since it gets
        called here as `vit_dataset(train=True, fold=test_sample_ID)`.
    epochs : int
        How many full passes over the training data to run.
    modelsave_address : str
        Folder path where logs (and, if the commented-out line were
        enabled, a model checkpoint) should be saved.
    dataset_name : str
        Either `"her2st"` or anything else (treated as `"skin"`) --
        selects which dataset-specific model hyperparameters and gene
        count to use.
    """
    # Build the TRAINING split for this fold (every sample except
    # `test_sample_ID`).
    dataset = vit_dataset(train=True, fold=test_sample_ID)

    # Wrap it in a DataLoader -- see explanation #2 near the top of this
    # file for what each argument means.
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    # Choose dataset-specific model settings: gene count and model depth
    # differ between the HER2ST (breast cancer) and skin datasets. See
    # explanation #3 near the top of this file for what each `THItoGene`
    # argument likely corresponds to.
    if dataset_name == "her2st":
        tagname = "-htg_her2st_785_32_cv"
        model = THItoGene(n_genes=785, learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8], n_layers=4)
    else:
        tagname = "-htg_skin_12_cv"
        model = THItoGene(n_genes=171, learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8], n_layers=8)

    # Set up CSV-file logging of training progress/metrics, into a
    # separate, per-fold-numbered log folder, so multiple folds' logs
    # don't overwrite each other.
    mylogger = CSVLogger(save_dir=modelsave_address + "/../logs/",
                         name="my_test_log_" + tagname + '_' + str(test_sample_ID))

    # Create the PyTorch Lightning Trainer -- see background concept #1
    # near the top of this file.
    # NOTE: the original code hard-coded `accelerator="gpu", devices=[7]`
    # (pinning training to GPU index 7 specifically), which would crash
    # on any machine without at least 8 GPUs. Changed here to
    # `accelerator="auto", devices="auto"` so PyTorch Lightning
    # automatically picks whatever GPU is available, or falls back to
    # CPU if none is -- see reproducibility note #3 near the top of this
    # file. This does not change training behavior on a machine that
    # does have a GPU; it only removes the hard-coded index so the
    # script is portable across machines.
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=epochs,
                         logger=mylogger)

    # Hand everything over to Lightning: it runs the full training loop
    # (looping over `epochs` epochs, calling the model's own
    # `training_step()`/`configure_optimizers()` methods defined inside
    # `vis_model.py`) completely automatically.
    trainer.fit(model, train_loader)

    # Build the TEST split for this fold (just the one held-out sample),
    # at normal resolution (`sr=False` -- i.e. using the real measured
    # spots, not the super-resolution virtual grid described in
    # `spatial_gene_expression_dataset.py`).
    dataset_test = vit_dataset(train=False, sr=False, fold=test_sample_ID)
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=0)

    # `trainer.predict()` runs the model's own `predict_step()` (defined
    # in `vis_model.py`) over the whole test_loader and collects the
    # results into a list -- `[0]` grabs the first (and, since there's
    # only one held-out sample here, only) batch's result, which is
    # expected to be a `(pred, gt)` tuple of AnnData objects (see
    # `gene_expression_inference.py`'s `model_predict()` for the same
    # kind of output format).
    pred, gt = trainer.predict(model=model, dataloaders=test_loader)[0]

    # Compute per-gene Pearson correlation between predicted and real
    # expression (see background concept #2 near the top of this file).
    # `get_R` is defined in `utils.py` (not shown in this project yet).
    R, p_val = get_R(pred, gt)

    # Attach the per-gene statistical results as extra gene-level
    # annotations on the AnnData object (`.var` holds per-COLUMN/per-gene
    # metadata, as opposed to `.obs`, which holds per-ROW/per-spot
    # metadata).
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)

    # Print a single overall accuracy summary number: the average
    # correlation across all genes (ignoring any `NaN` values -- e.g. from
    # genes with zero variance, where correlation is mathematically
    # undefined).
    print('Mean Pearson Correlation:', np.nanmean(R))
    # trainer.save_checkpoint(modelsave_address+"/"+"last_train_"+tagname+'_'+str(test_sample_ID)+".ckpt")


def test(test_sample_ID, vit_dataset, model_address, dataset_name):
    """
    Loads an ALREADY-TRAINED `THItoGene` model checkpoint for a given
    fold, and evaluates it on that fold's held-out test sample, returning
    the predicted and real gene-expression AnnData objects (and, for
    certain HER2ST samples, pathologist region labels too).

    Parameters
    ----------
    test_sample_ID : int
        Which fold's checkpoint to load and which sample to test on.
    vit_dataset : type
        `ViT_HER2ST` or `ViT_SKIN` (see `train()` above for the same
        parameter).
    model_address : str
        Folder containing the saved checkpoint file(s) -- see
        reproducibility note #4 near the top of this file for an
        important caveat about whether such a checkpoint file will
        actually exist.
    dataset_name : str
        Either `"her2st"` or anything else (treated as `"skin"`).

    Returns
    -------
    (adata_pred, adata_truth) or (adata_pred, adata_truth, label)
        The predicted and real ("truth") gene-expression AnnData objects.
        For certain specific HER2ST samples that have pathologist-provided
        tissue region labels available (see
        `spatial_gene_expression_dataset.py`'s `get_lbl()`/`self.label`),
        a third `label` array is also returned.
    """
    if dataset_name == "her2st":
        tagname = "-htg_her2st_785_32_cv"
        # Load the same top-785 gene list used during training, so
        # predicted columns can later be labeled with real gene names.
        g = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))

        # Load a previously-trained model directly from a saved
        # checkpoint file. `load_from_checkpoint` is a PyTorch Lightning
        # convenience method that restores both the model's trained
        # weights AND re-creates the model object using the given
        # hyperparameters (which must match what was used during
        # training, for the architecture to line up correctly).
        model = THItoGene.load_from_checkpoint(
            model_address + "/last_train_" + tagname + '_' + str(test_sample_ID) + ".ckpt", n_genes=785,
            learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8],
            n_layers=4)
    else:
        tagname = "-htg_skin_12_cv"
        # NOTE: this is a hard-coded, machine-specific absolute path from
        # the original author's own computer -- see reproducibility note
        # #5 near the top of this file. You will very likely need to
        # change this to a path that actually exists on your machine
        # (e.g. './data/skin_hvg_cut_1000.npy') before this branch will
        # work.
        g = list(np.load('/home/user/jiayuran/code/cond/THItoGene/data/skin_hvg_cut_1000.npy', allow_pickle=True))

        # NOTE: this checkpoint filename uses a DIFFERENT prefix
        # ("THItoGene_") than the her2st branch above ("last_train_") --
        # see reproducibility note #6 near the top of this file.
        model = THItoGene.load_from_checkpoint(
            model_address + "/THItoGene_" + tagname + '_' + str(test_sample_ID) + ".ckpt", n_genes=171,
            learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8],
            n_layers=8)

    # Automatically use a GPU if one is available, otherwise fall back to
    # CPU -- unlike `train()` above, this does NOT hard-code a specific
    # GPU index, so this function will work on any machine.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the same held-out test sample used to evaluate this fold's
    # model.
    dataset = vit_dataset(train=False, sr=False, fold=test_sample_ID)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    # Run inference using the helper function from
    # `gene_expression_inference.py`. `attention=False` is passed through
    # but (as noted in that file) is not currently used inside
    # `model_predict()`'s implementation.
    adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device=device)

    # Attach real gene names as column labels (previously the predicted
    # AnnData object's columns were just anonymous numbered columns).
    adata_pred.var_names = g

    # Rescale predicted values to mean 0 / std 1 per gene (does not affect
    # Pearson correlation -- see explanation #4 near the top of the file).
    sc.pp.scale(adata_pred)

    # A specific list of HER2ST sample indices are known (from
    # `spatial_gene_expression_dataset.py`'s hard-coded sample-name list)
    # to have pathologist-provided tissue region labels available. For
    # exactly those samples, also return the label array alongside the
    # predictions.
    if test_sample_ID in [5, 11, 17, 23, 26,
                          30] and dataset_name == 'her2st':
        label = dataset.label[dataset.names[0]]
        return adata_pred, adata_truth, label
    else:
        return adata_pred, adata_truth


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Reproducibility: fix EVERY relevant random number generator to the
    # same fixed seed, so that model weight initialization, data
    # shuffling, and any other randomness produce the exact same result
    # every time this script is run. See background concept #3 near the
    # top of this file.
    # ------------------------------------------------------------------
    seed = 0
    random.seed(seed)              # Python's built-in `random` module
    np.random.seed(seed)           # NumPy's random number generator
    torch.manual_seed(seed)        # PyTorch's CPU random number generator
    torch.cuda.manual_seed(seed)   # PyTorch's GPU random number generator (current GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch's GPU random number generator (ALL GPUs)

    # ------------------------------------------------------------------
    # Leave-one-sample-out cross-validation: TRAIN a completely separate
    # model once for every one of the 32 HER2ST samples, holding that
    # single sample out as the test set each time (see explanation #1
    # near the top of this file).
    # ------------------------------------------------------------------
    for i in range(0, 32):
        train(i, ViT_HER2ST, 300, "model", "her2st")

    # An equivalent training loop for the skin dataset (12 samples
    # instead of 32) is left commented out here, exactly as in your
    # uploaded code -- uncomment it if/when you want to train on the skin
    # dataset too.
    # for i in range(12):
    #     train(i, ViT_SKIN, 300, "/home/user/jiayuran/code/cond/THItoGene/model", "skin")

    # ------------------------------------------------------------------
    # Now TEST every one of those 32 just-trained fold models on its own
    # held-out sample, and report each fold's average Pearson correlation.
    # ------------------------------------------------------------------
    for i in range(0, 32):
        dataset = 'her2st'
        test_sample = i
        if dataset == "her2st":
            # Certain samples have pathologist labels available (see
            # `test()`'s docstring above), so they return an extra
            # `label` value.
            if test_sample in [5, 11, 17, 23, 26, 30]:
                pred, gt, label = test(test_sample, ViT_HER2ST, "model",
                                       dataset)
            else:
                pred, gt = test(test_sample, ViT_HER2ST, "model", dataset)

            R, p_val = get_R(pred, gt)
            pred.var["p_val"] = p_val
            pred.var["-log10p_val"] = -np.log10(p_val)
            print('Mean Pearson Correlation:', np.nanmean(R))
        else:
            # (Currently unreachable, since `dataset` is hard-coded to
            # `'her2st'` just above -- kept here unchanged, matching your
            # uploaded code, in case you later want to make `dataset` a
            # variable that can also be set to something else to trigger
            # this skin-dataset branch instead.)
            pred, gt = test(test_sample, ViT_SKIN, "model", dataset)
            R, p_val = get_R(pred, gt)
            pred.var["p_val"] = p_val
            pred.var["-log10p_val"] = -np.log10(p_val)
            print('Mean Pearson Correlation:', np.nanmean(R))
