"""
==============================================================================
spatial_gene_expression_dataset.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file is a "data loader" for a machine learning project.

Background concept #1 -- Spatial Transcriptomics (ST):
    Spatial transcriptomics is a lab technique. Scientists take a thin slice
    of tissue (for example, a piece of a tumor, skin, or brain), put it on a
    glass slide, and photograph it under a microscope (this gives an H&E
    stained tissue IMAGE). On top of that, at many small circular locations
    on the slide (called "SPOTS"), the technology also measures how active
    thousands of GENES are at that exact spot. So, for every spot we end up
    with two things:
        1) a small image patch (what the tissue looks like there), and
        2) a list of numbers, one per gene, saying how much that gene is
           "expressed" (turned on) at that spot.

Background concept #2 -- Why build a PyTorch "Dataset"?
    In PyTorch (a popular deep learning library), a "Dataset" is a Python
    class whose only job is to say:
        - how many samples/examples exist (the __len__ method), and
        - given an index, return one training example (the __getitem__
          method).
    PyTorch's DataLoader then uses this class to automatically create
    batches, shuffle data, and feed it to a neural network during training.
    We are NOT training a model in this file -- we are only PREPARING the
    data so a model (defined elsewhere) can consume it.

Background concept #3 -- The goal of the model that will use this data:
    The overall research goal (this is a common setup in papers such as
    "HisToGene" / "Hist2ST" / "ST-Net") is to train a neural network that
    looks ONLY at the tissue image and tries to PREDICT the gene expression
    values, without needing the expensive lab measurement. If this works,
    doctors/scientists could estimate gene activity just from a normal
    microscope photo.

This single file defines THREE different Dataset classes, one for each
public dataset used in the paper/project:
    1. ViT_HER2ST -> loads the "HER2ST" breast cancer dataset
    2. ViT_SKIN   -> loads the "GSE144240" skin cancer dataset
    3. DATA_BRAIN -> loads the "10x Genomics Visium" human brain dataset

All three classes follow the exact same recipe, just pointed at different
folders and file formats:
    Step A. Find the list of tissue "samples" (each sample = one patient
             slide, containing many spots).
    Step B. Split samples into a TRAIN set and a TEST set (this project
             uses "leave-one-sample-out" cross validation -- see the big
             comment about `fold` below).
    Step C. Load every sample's full microscope image into memory.
    Step D. Load every sample's spot metadata: pixel coordinates (where a
             spot sits on the image, in pixels) and grid coordinates (row,
             column position on the printed spot grid).
    Step E. Load every sample's gene expression table and keep only a
             pre-selected list of "highly variable genes" (the genes that
             differ the most between spots -- the most informative ones).
    Step F. Normalize the gene expression numbers (see the normalization
             explanation below) so the neural network trains more easily.
    Step G. Build a "graph" that connects each spot to its nearest
             neighboring spots (see the k-NN graph explanation below). This
             lets a Graph Neural Network share information between nearby
             spots.
    Step H. When PyTorch asks for item number `i`, cut out an image patch
             (a small square) around each spot's pixel location and return
             it together with its matching gene-expression numbers.

------------------------------------------------------------------------------
KEY ALGORITHMS/CONCEPTS USED IN THIS FILE (explained in plain language)
------------------------------------------------------------------------------

1) Leave-one-sample-out cross-validation via the `fold` argument
   ---------------------------------------------------------------
   Cross-validation is a way to test whether a model generalizes to NEW,
   unseen data instead of just memorizing what it was trained on.
   "Leave-one-out" here means: out of all the tissue samples we have
   (say 32 of them), we pick exactly ONE sample to be the "test" sample,
   and train the model on the remaining ones. The `fold` number (0, 1, 2...)
   simply chooses WHICH sample gets left out this time. If you re-run the
   code with fold=3, a different sample becomes the test sample. Running
   the whole experiment once for every possible fold and averaging the
   results gives a much more trustworthy performance estimate than testing
   on just one split.
   Reproducibility note: because the sample list and the leave-out index
   are both fixed (not random), using the same `fold` number will always
   produce the exact same train/test split.

2) Library-size normalization + log transform (gene expression preprocessing)
   ---------------------------------------------------------------------------
   Raw gene expression counts are not directly comparable between spots,
   because some spots simply captured more total genetic material than
   others (like comparing word counts of two book chapters of different
   length). "Library-size normalization" rescales every spot's gene counts
   so that they all sum to the same total, removing this "how much material
   did we capture" bias. After that, we take the natural logarithm of the
   values. Gene expression data is very "skewed" (a few genes have huge
   counts, most have small counts); the log transform compresses the huge
   values and makes the distribution easier for a neural network to learn
   from. This exact two-step recipe (normalize by library size, then log)
   is standard practice in single-cell/spatial transcriptomics analysis
   (it is implemented here using the `scprep` library's
   `normalize.library_size_normalize` and `transform.log` functions).

3) k-Nearest-Neighbor (k-NN) spatial graph construction (`calcADJ`)
   -------------------------------------------------------------------
   Spots on the tissue slide sit in a 2D grid (each spot has an (x, y)
   grid coordinate). We build a "graph" where each spot is a "node", and
   we draw an "edge" (connection) between a spot and its `k` closest
   neighboring spots (here k=4, meaning: connect each spot to its 4
   nearest neighbors in space). The output is an "adjacency matrix": a
   table of 0s and 1s (or similar) where a 1 in row i, column j means
   "spot i is connected to spot j". This adjacency matrix is later fed
   into a Graph Neural Network (GNN) layer, which lets the model mix
   information between neighboring spots -- similar to how a
   Convolutional Neural Network (CNN) mixes information between
   neighboring pixels, except here the neighborhood is defined by the
   spot graph instead of a fixed pixel grid. The actual neighbor-finding
   math (`calcADJ`) lives in a separate file, `graph_construction.py`,
   which is imported at the top of this file -- if you have not uploaded
   that file yet, this script will not run until you do.

4) Image "patch" extraction
   ---------------------------
   A patch is just a small square crop of the big tissue image, centered
   on one spot's pixel location. `self.r` is the patch "radius" in pixels
   (half of the patch's side length). So a patch is a
   (2*self.r) x (2*self.r) pixel square. We feed one small patch per spot
   into the neural network instead of the whole giant slide image, because
   (a) it is far too large to process at once, and (b) each patch already
   contains the local visual context relevant to predicting that spot's
   gene expression.

5) Super-resolution mode (`sr=True`)
   --------------------------------------
   Normally we only have gene expression measured at the real spot
   locations. In "super-resolution" mode, instead of using the real spot
   centers, the code creates a dense, evenly spaced GRID of fake "virtual
   spot" centers covering the whole tissue (moving 56 pixels at a time in
   both directions). This lets a trained model predict gene expression at
   a much finer resolution than the original spots -- useful for making
   detailed prediction maps, but there is no real gene-expression ground
   truth for these virtual spots (they are only used at prediction time,
   not for computing a training loss).

6) Data augmentation (`aug=True`)
   ------------------------------------
   Data augmentation means applying small random (but label-preserving)
   changes to training images so the model sees more variety and does not
   just memorize the exact pixels. Here it uses:
     - ColorJitter: randomly tweaks brightness/contrast/saturation.
     - RandomHorizontalFlip / RandomRotation (only in the HER2ST class):
       randomly flips/rotates the image, since tissue images don't have a
       single "correct" orientation.
   This is only applied to training data, never to test data (we want test
   evaluation to be on the real, unmodified image).

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Folder layout expected on disk (relative to wherever you run Python):
       ./data/her2st/data/ST-cnts/          (HER2ST gene count tables)
       ./data/her2st/data/ST-imgs/          (HER2ST tissue images)
       ./data/her2st/data/ST-spotfiles/     (HER2ST spot pixel positions)
       ./data/her2st/data/ST-pat/lbl/       (HER2ST expert region labels)
       ./data/her_hvg_cut_1000.npy          (HER2ST top-1000 gene list)
       ./data/GSE144240_RAW/                (skin dataset images+tables)
       ./data/skin_hvg_cut_1000.npy         (skin top-1000 gene list)
       ./data/10X/                          (10x Visium brain dataset)
       ./data/10X/final_gene.npy            (brain gene list)

2. Required Python packages (install exact versions if you want bit-for-bit
   reproducible results, since library defaults can change between
   versions):
       numpy, pandas, scanpy, scprep, torch, torchvision, pillow (PIL)

3. This file also needs a sibling file named `graph_construction.py` in the
   same folder, which must define a function:
       calcADJ(coord, k=4, pruneTag='NA')
   (upload / include that file as well, or the `import` line below will
   fail).

4. Determinism: nothing in this file uses a random-number generator by
   itself (the only "randomness" comes from the `transforms` data
   augmentation, which is applied later, during training, image-by-image).
   For fully reproducible TRAINING runs, remember to set seeds in your
   training script, e.g.:
       import torch, numpy as np, random
       torch.manual_seed(0)
       np.random.seed(0)
       random.seed(0)
   This dataset file itself will always produce the same train/test split
   for a given `fold` number, run after run.
==============================================================================
"""

import glob
import os

import numpy as np
import pandas as pd
import scanpy as sc          # single-cell / spatial-omics analysis toolkit
import scprep as scp         # single-cell preprocessing helper functions
import torch
import torchvision.transforms as transforms
from PIL import ImageFile, Image

# calcADJ builds the k-nearest-neighbor spot graph described above.
# This function lives in graph_construction.py, a separate file.
from graph_construction import calcADJ

# Some whole-slide tissue images are extremely large (way more pixels than
# a typical photo). The next two lines tell the PIL/Pillow image library:
#   1) don't refuse to open very large images (raise the safety pixel limit)
#   2) don't crash if an image file got cut off / truncated on disk
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==============================================================================
# CLASS 1: ViT_HER2ST
# ==============================================================================
# Loads the "HER2ST" breast-cancer spatial transcriptomics dataset.
# "ViT" in the class name is a hint that the image patches produced here are
# meant to be fed into a Vision Transformer (a type of neural network that
# treats an image as a sequence of small patches -- exactly the patches this
# class produces).
class ViT_HER2ST(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        """
        Parameters (all explained from scratch):
        -----------------------------------------------------------------
        train : bool
            If True, build the TRAINING split. If False, build the TEST
            split. See the leave-one-sample-out explanation at the top of
            this file.
        gene_list : (not actually used as an input here -- this class always
            loads its own fixed list of 1000 genes from disk; the parameter
            is kept only so all three classes share the same constructor
            signature).
        ds : unused in this class (kept for interface compatibility with
            the other two dataset classes in this file).
        sr : bool
            "Super-resolution" mode switch. See explanation above. False by
            default (use the real measured spots).
        fold : int
            Which sample index to hold out as the TEST sample. See the
            cross-validation explanation above.
        """
        super(ViT_HER2ST, self).__init__()

        # ---- Where the raw HER2ST files live on disk ----
        self.cnt_dir = r'./data/her2st/data/ST-cnts'      # gene count tables
        self.img_dir = r'./data/her2st/data/ST-imgs'      # tissue images
        self.pos_dir = r'./data/her2st/data/ST-spotfiles'  # spot positions
        self.lbl_dir = r'./data/her2st/data/ST-pat/lbl'   # pathologist labels

        # `self.r` = patch "radius" in pixels. 224 is the classic input
        # image size used by many pretrained vision models (like ResNet /
        # ViT), so dividing by 4 gives a patch size of 224/4 = 56 pixels
        # radius -> a 112x112 pixel patch per spot.
        self.r = 224 // 4

        # Load the fixed list of the 1000 "highly variable genes" (the
        # genes whose expression differs the most across spots -- these
        # are the most informative genes to try to predict).
        gene_list = list(np.load(r'./data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list

        # ---- Figure out which tissue samples exist ----
        # Every gene-count file is named like "A1.tsv", "B2.tsv", etc.
        # os.listdir gives us all filenames in the folder; sorting makes
        # the order deterministic (same every time you run this).
        names = os.listdir(self.cnt_dir)
        names.sort()
        # Keep only the first 2 characters of each filename (the sample ID,
        # e.g. "A1" out of "A1.tsv").
        names = [i[:2] for i in names]

        self.train = train
        self.sr = sr

        # Only samples index 1 through 32 are used here (33 total slides
        # exist in HER2ST, but index 0 is skipped -- likely reserved for a
        # different purpose in the original study).
        samples = names[1:33]

        # ---- Leave-one-sample-out split ----
        # `fold` selects exactly ONE sample name to be the test sample.
        te_names = [samples[fold]]
        # Every other sample becomes a training sample.
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        # ---- Load all tissue images into memory, one per sample ----
        # get_img() (defined further down) opens the full slide image file.
        # We convert it into a PyTorch Tensor (PyTorch's basic
        # multi-dimensional array type, similar to a numpy array but usable
        # directly by neural networks).
        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        # ---- Load metadata (gene counts + spot pixel positions) ----
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        # `self.label` will hold pathologist-provided tissue-region labels
        # (e.g. "invasive cancer", "connective tissue"), if available for
        # that sample. Not every sample has these labels.
        self.label = {i: None for i in self.names}

        # Map each text label to an integer ID. Neural networks work with
        # numbers, not text, so labels must be converted like this before
        # they can be used (e.g. for a classification task).
        self.lbl2id = {
            'invasive cancer': 0,
            'breast glands': 1,
            'immune infiltrate': 2,
            'cancer in situ': 3,
            'connective tissue': 4,
            'adipose tissue': 5,
            'undetermined': -1,
        }

        # Only a subset of samples ('A1', 'B1', ... 'J1') have pathologist
        # labels available on disk. The block below loads labels only for
        # those samples, and fills in "-1" (meaning "no label available")
        # for every other sample.
        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            self.label[self.names[0]] = lbl
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    # Convert each text label into its integer ID using the
                    # mapping dictionary defined above.
                    lbl = torch.Tensor(list(map(lambda i: self.lbl2id[i], lbl)))
                    self.label[i] = lbl
                else:
                    # No label file for this sample -> fill with -1
                    # ("unknown label") for every spot in that sample.
                    self.label[i] = torch.full((len(idx),), -1)

        # ---- Gene expression preprocessing ----
        # For every sample: take only the chosen gene columns, then apply
        # library-size normalization followed by a log transform (see the
        # explanation near the top of this file).
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            for i, m in self.meta_dict.items()
        }

        # `center_dict`: pixel-space (x, y) location of every spot, rounded
        # down to whole pixel numbers -- used later to cut out image
        # patches.
        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }

        # `loc_dict`: grid-space (x, y) coordinate of every spot (its
        # row/column position on the printed spot array, NOT pixels).
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        # Will cache computed image patches per sample so we don't cut them
        # out from the full image more than once (patches are expensive to
        # compute, so we compute them once and reuse them).
        self.patch_dict = {}

        # Number of spots in each sample, and running cumulative total
        # (not directly used inside this file, but handy for anyone
        # building an index across all samples combined).
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)

        # Lets us go from a plain integer index (0, 1, 2, ...) back to the
        # sample name string (e.g. "A1") -- this is what makes
        # __getitem__(index) work.
        self.id2name = dict(enumerate(self.names))

        # Image augmentation pipeline used only for randomly perturbing
        # training images (see explanation above). "Compose" just chains
        # several transformations together, applying them one after another.
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
        ])

        # Build the k-NN spot graph (k=4 nearest neighbors) for every
        # sample, using the grid coordinates in loc_dict. See the k-NN
        # graph explanation near the top of this file.
        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        """
        A small diagnostic/debugging helper. For every sample, it counts
        (per gene) how many spots have a non-zero expression value for that
        gene. NOTE: this function computes the counts into array `a` but
        never returns or uses them elsewhere in this file -- it currently
        has no effect on training. It is kept here unchanged for
        reproducibility with the original code; feel free to extend it
        (e.g. `return a`) if you want to actually inspect gene coverage.
        """
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        """
        This is the method PyTorch calls automatically whenever it needs
        training example number `index`. It must return everything the
        model needs for that one sample (one whole tissue slide, which
        itself contains many spots).
        """
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]
        # Re-order the image's axes. Images are normally stored as
        # (height, width, color-channels); here we swap height and width
        # to (width, height, color-channels) to match how pixel coordinates
        # (x, y) were computed elsewhere in this class.
        im = im.permute(1, 0, 2)

        exps = self.exp_dict[self.id2name[i]]      # gene expression matrix
        centers = self.center_dict[self.id2name[i]]  # pixel spot centers
        loc = self.loc_dict[self.id2name[i]]         # grid spot coordinates
        positions = torch.LongTensor(loc)
        label = self.label[self.id2name[i]]

        # Re-use previously computed patches if we already made them once.
        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None

        adj = self.adj_dict[name]   # the k-NN neighbor graph for this sample

        # Full size (in numbers) of one flattened patch: 3 color channels *
        # patch height * patch width. The "*4" here accounts for the patch
        # being (2*r) x (2*r) in size (2*2 = 4).
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            # -------------------------------------------------------------
            # SUPER-RESOLUTION MODE: build a dense, evenly spaced grid of
            # virtual spot centers instead of using the real measured spots
            # (see explanation near the top of the file).
            # -------------------------------------------------------------
            centers = torch.LongTensor(centers)

            # Find the bounding box (min/max pixel coordinates) covering
            # all real spots, so the virtual grid only covers the tissue
            # area, not the whole (possibly much larger) image.
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()

            # Used to rescale virtual-grid pixel coordinates down into a
            # smaller "position index" range (roughly 0-30), matching the
            # kind of grid-coordinate scale used for the real spots.
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            # Start a running list of virtual centers/positions. We begin
            # with one placeholder row (which we discard at the end).
            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            # Step across the whole bounding box in fixed 56-pixel jumps,
            # generating one virtual spot center at every grid point.
            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat(
                        (positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0
                    )
                    x += 56
                y += 56

            # Drop the placeholder first row we started with.
            centers = centers[1:, :]
            positions = positions[1:, :]

            # Cut out an image patch around every virtual center, then
            # "flatten" it (turn the 3D patch into one long 1D list of
            # numbers) -- useful if the downstream model expects a flat
            # vector input rather than a 3D image.
            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            # In super-resolution mode there is no real gene-expression
            # ground truth for these made-up virtual spots, so we only
            # return the image patches, their grid positions, and their
            # pixel centers.
            return patches, positions, centers

        else:
            # -------------------------------------------------------------
            # NORMAL MODE: use the real, measured spots.
            # -------------------------------------------------------------
            n_patches = len(centers)
            exps = torch.Tensor(exps)

            if patches is None:
                # (channels, height, width) is PyTorch's standard image
                # tensor shape convention for feeding into CNN/ViT models.
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                    # Move color-channel axis to the front to match PyTorch's
                    # (channels, height, width) convention.
                    patches[i] = patch.permute(2, 0, 1)
                # Cache these patches so we never have to re-cut them again
                # for this sample.
                self.patch_dict[name] = patches

            if self.train:
                # Training samples don't need pixel centers returned (we
                # already know exactly where each spot came from during
                # training, and we don't need to draw prediction maps).
                return patches, positions, exps, adj
            else:
                # Test samples also return `centers`, which is handy for
                # later visualizing predictions back onto the original
                # tissue image.
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        """
        PyTorch calls this to know how many "items" this Dataset has.
        Here, one "item" = one whole tissue sample (slide), NOT one spot --
        so the length equals the number of samples in this split
        (train or test), not the number of spots.
        """
        return len(self.exp_dict)

    # -------------------------------------------------------------------
    # Below: small file-reading "helper" methods. Each one knows exactly
    # how to find and parse one particular type of file on disk for the
    # HER2ST dataset.
    # -------------------------------------------------------------------

    def get_img(self, name):
        """Find and open the full tissue slide image for one sample name."""
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        print(path)
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        """Load the raw gene-count table (spots x genes) for one sample."""
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        """
        Load the spot position table for one sample, and build a text `id`
        column ("123x456" style) so it can later be matched up with the
        gene-count table's row names.
        """
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_lbl(self, name):
        """Load pathologist-provided tissue-region labels for one sample."""
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df

    def get_meta(self, name, gene_list=None):
        """
        Combine the gene-count table and the spot-position table into one
        combined table ("meta"), matched up spot-by-spot using the `id`
        column built in get_pos().
        """
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_overlap(self, meta_dict, gene_list):
        """
        Given a wanted gene_list, return only the genes that actually exist
        in EVERY sample's table (the "intersection" / overlap). This avoids
        crashing on a gene that is missing in some samples.
        """
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


# ==============================================================================
# CLASS 2: ViT_SKIN
# ==============================================================================
# Loads the "GSE144240" skin-cancer spatial transcriptomics dataset. The
# overall recipe is identical to ViT_HER2ST above -- only the file locations,
# naming conventions, and a couple of extra options (`aug`, `norm`) differ.
# To avoid repeating the exact same explanations, only the NEW ideas
# compared to ViT_HER2ST are called out below; read ViT_HER2ST's comments
# first if anything here is unclear.
class ViT_SKIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        """
        New parameters compared to ViT_HER2ST:
        -----------------------------------------------------------------
        aug : bool
            If True, apply random image augmentation (ColorJitter) to
            training images -- see the data-augmentation explanation near
            the top of this file.
        norm : bool
            If True, apply an EXTRA normalization step after the usual
            library-size-normalize + log steps: `sc.pp.scale`, which is a
            "z-score" style normalization -- it rescales every gene so it
            has mean 0 and standard deviation 1 across all spots. This can
            help some models train faster/more stably, at the cost of
            losing the original scale of expression values.
        """
        super(ViT_SKIN, self).__init__()
        self.dir = './data/GSE144240_RAW'
        self.r = 224 // 4

        # This dataset has 4 patients, each with 3 tissue replicates, so
        # 4 x 3 = 12 total samples. We build all 12 sample names here
        # (e.g. "P2_ST_rep1") instead of reading them off disk.
        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)

        # Pre-selected top-1000 highly-variable genes for the skin dataset.
        gene_list = list(np.load('./data/skin_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list

        self.train = train
        self.sr = sr
        self.aug = aug
        self.norm = norm

        # A lighter augmentation pipeline than HER2ST's (no flips/rotation
        # here, just color jitter).
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.ToTensor(),
        ])

        # ---- Leave-one-sample-out split (same idea as ViT_HER2ST) ----
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        if self.aug:
            # Keep images as PIL Image objects (not yet converted to
            # tensors) so the torchvision `transforms` pipeline (which
            # expects PIL images) can be applied later, per __getitem__
            # call, giving DIFFERENT random augmentation every time.
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)

        # ---- Gene expression preprocessing (with optional extra scaling) ----
        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(
                    scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
                )
                for i, m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
                for i, m in self.meta_dict.items()
            }

        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.patch_dict = {}

        # Build the same kind of k-NN spot graph as in ViT_HER2ST.
        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        """Same diagnostic helper as in ViT_HER2ST -- see that class for details."""
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]

        if self.aug:
            # Apply the random augmentation transform NOW (this is why we
            # kept images as raw PIL Images above -- so a fresh random
            # augmentation happens every time this sample is requested).
            im = self.transforms(im)
            im = im.permute(2, 1, 0)
        else:
            im = im.permute(1, 0, 2)

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None
        adj = self.adj_dict[name]

        if self.sr:
            # Same super-resolution virtual-grid logic as ViT_HER2ST --
            # see that class's __getitem__ for the full explanation.
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat(
                        (positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0
                    )
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps)
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                    patches[i] = patch.permute(2, 0, 1)
                self.patch_dict[name] = patches
            if self.train:
                return patches, positions, exps, adj
            else:
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        """Find and open the tissue image whose filename ends in `name`.jpg."""
        path = glob.glob(self.dir + '/*' + name + '.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        """Load the raw gene-count table for one sample."""
        path = glob.glob(self.dir + '/*' + name + '_stdata.tsv')[0]
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        """Load the spot position table for one sample and build its `id` column."""
        pattern = f"{self.dir}/*spot*{name}*.tsv"
        path = glob.glob(pattern)[0]
        print(path)
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list=None):
        """
        Join gene counts with spot positions. `how='inner'` means: only
        keep spots that exist in BOTH tables (drop anything that doesn't
        match up), which the HER2ST class above didn't need to specify.
        """
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')
        return meta

    def get_overlap(self, meta_dict, gene_list):
        """Keep only genes present in every sample -- same idea as ViT_HER2ST."""
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


# ==============================================================================
# CLASS 3: DATA_BRAIN
# ==============================================================================
# Loads the "10x Genomics Visium" human brain spatial transcriptomics
# dataset. Structurally this is almost identical to ViT_SKIN above (same
# `aug`/`norm` options, same overall pipeline) -- only file paths, sample
# names, and one metadata-loading quirk differ. Read ViT_HER2ST's comments
# first for full background if needed.
class DATA_BRAIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        super(DATA_BRAIN, self).__init__()
        self.dir = './data/10X'
        self.r = 224 // 4

        # 12 fixed sample IDs for the 10x Visium brain dataset.
        sample_names = [
            '151507', '151508', '151509', '151510', '151669', '151670',
            '151671', '151672', '151673', '151674', '151675', '151676',
        ]

        # Pre-selected gene list for the brain dataset.
        gene_list = list(np.load('./data/10X/final_gene.npy'))
        self.gene_list = gene_list

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.ToTensor(),
        ])
        self.norm = norm

        # ---- Leave-one-sample-out split ----
        samples = sample_names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        if self.aug:
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)

        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(
                    scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
                )
                for i, m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
                for i, m in self.meta_dict.items()
            }

        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.patch_dict = {}

        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        """Same diagnostic helper as in the other two classes."""
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]

        if self.aug:
            im = self.transforms(im)
            im = im.permute(2, 1, 0)
        else:
            im = im.permute(1, 0, 2)

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None
        adj = self.adj_dict[name]

        if self.sr:
            # Same super-resolution logic as the other two classes.
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat(
                        (positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0
                    )
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps)
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                    patches[i] = patch.permute(2, 0, 1)
                self.patch_dict[name] = patches
            if self.train:
                return patches, positions, exps, adj
            else:
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        """Find and open the full-resolution tissue image for one sample."""
        path = glob.glob(self.dir + f'/{name}/{name}_full_image.tif')[0]
        im = Image.open(path)
        return im

    def get_meta(self, name, gene_list=None):
        """
        NOTE (kept exactly as in the original code, for reproducibility):
        this always loads the SAME metadata file (sample "151507"),
        regardless of which sample `name` was actually requested. This
        looks like it may be a bug in the original script -- every sample
        ends up using sample 151507's gene-expression/position table. If
        you intend to use per-sample metadata files, you would need to
        change this line to something like:
            f'./data/10X/{name}/10X_Visium_{name}_meta.csv'
        This has been left unchanged here so behavior exactly matches the
        original code you uploaded; fix it in a later pass if you confirm
        it's unintentional.
        """
        meta = pd.read_csv('./data/10X/151507/10X_Visium_151507_meta.csv', index_col=0)
        return meta

    def get_overlap(self, meta_dict, gene_list):
        """Keep only genes present in every sample -- same idea as the other classes."""
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
