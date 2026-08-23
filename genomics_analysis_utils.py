"""
==============================================================================
genomics_analysis_utils.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file is a "grab-bag" of helper functions used throughout this project
for loading, preprocessing, clustering, visualizing, and scoring spatial
gene-expression data. It is imported elsewhere in this project via
`from utils import *` (see `gene_expression_inference.py` and
`train_and_evaluate_thitogene.py`), which is how functions like `get_R()`
end up available in those files even though they're actually defined here.

This file relies heavily on the `scanpy` and `anndata` libraries, both
already introduced briefly in earlier files of this project
(`spatial_gene_expression_dataset.py`, `gene_expression_inference.py`).
As a quick refresher: `AnnData` is a standard container object that bundles
together a main data matrix (here: rows = tissue spots, columns = genes),
plus extra per-spot and per-gene annotations (`.obs`, `.var`, `.obsm`), and
`scanpy` (imported as `sc`) is the most widely used toolkit for analyzing
that kind of data.

------------------------------------------------------------------------------
KEY CONCEPTS USED THROUGHOUT THIS FILE (explained in plain language)
------------------------------------------------------------------------------

1) Normalization + log transform (`sc.pp.normalize_total`, `sc.pp.log1p`)
   ---------------------------------------------------------------------------
   `sc.pp.normalize_total()` is scanpy's built-in version of the
   "library-size normalization" idea already explained in
   `spatial_gene_expression_dataset.py`: it rescales every spot's total
   gene-count sum to be equal across spots, removing the "some spots just
   happened to capture more material overall" bias. `sc.pp.log1p()` then
   applies a log transform (specifically `log(1 + x)`, which handles zero
   counts gracefully, unlike a plain `log(x)` which would be undefined at
   0) -- the same "compress heavily skewed data" idea explained earlier,
   just using scanpy's own built-in function instead of the `scprep`
   library used in the dataset-loading file.

2) "Marker genes" and cell-type scoring (the `BCELL`, `TUMOR`, `CD4T`, ...
   variables, the `IG` and `LYM` dictionaries, and the `include` logic
   inside `preprocess()`)
   ---------------------------------------------------------------------------
   A "marker gene" is a gene that is well known (from prior biological
   research) to be especially active specifically in ONE particular cell
   type, and not in others. For example, `CD19`/`CD79A`/`CD79B`/`MS4A1`
   are well-known marker genes for B cells (a type of immune cell). By
   averaging together the expression of a cell type's known marker genes
   at every spot, you get a rough "how much does this spot look like it
   contains this cell type" score -- even without doing full, complex
   cell-type classification. The `IG` dictionary here groups marker genes
   for several cell/tissue types relevant to the skin-cancer dataset used
   elsewhere in this project (immune cells, tumor cells, dendritic cells,
   melanoma cells), and `LYM` is a smaller subset covering just the
   lymphocyte (immune cell) types. `MARKERS` is simply every individual
   marker gene from `IG`, flattened into one single list.

3) Principal Component Analysis (PCA) -- `sc.pp.pca()`
   ------------------------------------------------------------
   PCA is a classic dimensionality-reduction technique: given data with
   many columns (here, potentially thousands of genes), it finds a
   smaller number of new, artificial "combined" dimensions (called
   "principal components") that together capture as much of the
   original data's variation/pattern as possible. Working with a much
   smaller number of PCA dimensions afterward (instead of every single
   raw gene) makes later steps like clustering and visualization both
   faster and often more robust to noise.

4) Nearest-neighbor graph, UMAP, and Leiden clustering
   (`sc.pp.neighbors`, `sc.tl.umap`, `sc.tl.leiden`)
   ---------------------------------------------------------------------------
   - `sc.pp.neighbors()`: builds a k-nearest-neighbor graph BETWEEN SPOTS,
     but this time based on how similar their (PCA-reduced) gene
     expression profiles are -- NOT based on physical spatial location
     (contrast this with the spatial k-NN graph built by `calcADJ()` in
     `spot_knn_graph_builder.py` earlier in this project, which connects
     spots that are physically close together on the tissue slide,
     regardless of how similar their gene expression is).
   - `sc.tl.umap()`: "Uniform Manifold Approximation and Projection" --
     another dimensionality-reduction technique (like PCA, but more
     powerful for VISUALIZATION specifically), typically used to squash
     data down to just 2 dimensions so it can be plotted on a simple
     scatter plot, while trying to preserve which points were originally
     close together in the full, high-dimensional gene-expression space.
   - `sc.tl.leiden()`: a graph-based clustering algorithm (i.e., it
     groups together spots that are densely interconnected in the
     nearest-neighbor graph built above) -- a very popular, standard
     choice for automatically discovering distinct cell-type/tissue-region
     groups directly from gene expression data, without needing to
     specify in advance how many groups/clusters should exist (unlike
     KMeans below, which does require a fixed number of clusters).

5) t-SNE and KMeans clustering (`sc.tl.tsne`, `sklearn.cluster.KMeans`)
   ---------------------------------------------------------------------------
   - t-SNE ("t-distributed Stochastic Neighbor Embedding") is yet another
     dimensionality-reduction technique, similar in spirit and purpose to
     UMAP (mainly used for 2D visualization), just using a different
     underlying mathematical approach.
   - KMeans is a classic, simple clustering algorithm: given a FIXED
     number `k` of desired clusters, it repeatedly (a) assigns every
     point to whichever of `k` "cluster center" points it's currently
     closest to, then (b) moves each cluster center to the average
     position of all points currently assigned to it, until the
     assignments stop changing. `init="k-means++"` is a smart, standard
     way of picking good STARTING cluster-center positions (rather than
     purely random ones), which tends to produce better, more consistent
     final results. `random_state=0` fixes KMeans's own internal
     randomness (used both for the "++" initialization and for breaking
     ties), which is essential for reproducibility -- see the
     reproducibility section near the end of this file.

6) Adjusted Rand Index (ARI) -- `sklearn.metrics.adjusted_rand_score`
   ---------------------------------------------------------------------------
   ARI is a standard way to measure how SIMILAR two different groupings
   (clusterings) of the same set of items are -- here, comparing an
   automatically-discovered KMeans clustering against real,
   pathologist-provided tissue-region labels (see
   `spatial_gene_expression_dataset.py`'s `get_lbl()`/`self.label`, from
   earlier in this project). It ranges from roughly 0 (the two groupings
   agree no better than random chance) up to exactly 1 (a perfect match
   between the two groupings) -- and can occasionally go slightly
   negative for groupings that agree even WORSE than random chance. It is
   specifically "adjusted" (as opposed to a simpler, unadjusted
   agreement-counting score) to correct for the fact that, purely by
   luck, some agreement between two random groupings is expected even if
   they have nothing genuinely in common, especially when there are only
   a few possible groups.

7) Pearson correlation for scoring predictions (`get_R()`)
   ---------------------------------------------------------------------------
   As already introduced in `train_and_evaluate_thitogene.py`, Pearson
   correlation measures how strongly two lists of numbers move together
   in a straight-line relationship. `get_R()` here is the actual
   implementation of that computation used throughout this project: for
   every gene (or, alternatively, for every spot -- see the `dim`
   parameter explained in its docstring below), it computes one Pearson
   correlation value between the model's predicted values and the real,
   measured values, using `scipy.stats.pearsonr` (imported here as
   `func`, with `pearsonr` as its default).

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: numpy, pandas, scanpy, scprep, anndata,
   pillow (PIL), scipy, scikit-learn (sklearn).

2. Several functions here (`comp_tsne_km`, `co_embed`, `cluster`) use
   KMeans, which involves randomness. They all correctly pass
   `random_state=0`, which is exactly what's needed for reproducible
   clustering results run after run. If you write any NEW code using
   KMeans elsewhere in this project, make sure to set `random_state` to a
   fixed value the same way.

3. `sc.tl.leiden()` (inside `comp_umap()`) and `sc.tl.umap()` /
   `sc.tl.tsne()` also involve some internal randomness (e.g. for their
   optimization procedures); for fully reproducible plots/clusters
   across runs, consider explicitly passing a `random_state=0` argument
   to those calls as well if your installed scanpy version supports it
   (this file currently calls them with their own defaults, unchanged
   from your uploaded code).

4. Hard-coded, relative file paths: several functions here load files
   from paths like `'data/skin_a.npy'`, `'data/her_g_list.npy'`, and the
   HER2ST folders under `'data/her2st/data/...'`. These are RELATIVE
   paths, meaning they will only resolve correctly if you run your script
   from the correct working directory (the parent folder containing
   `data/`) -- the same requirement noted in
   `spatial_gene_expression_dataset.py` earlier in this project.

5. Two identical imports: this file imports the `anndata` library TWICE,
   under two different names (`import anndata as ad` and
   `import anndata as ann`). This is redundant but harmless -- both names
   point to the exact same library, and different functions in this file
   happen to use one alias or the other (kept exactly as in your uploaded
   code, to avoid changing behavior anywhere they're referenced).

6. A minor inconsistency worth noting for anyone extending this file:
   `build_adata()`'s spot-position loading does NOT round the `x`/`y`
   pixel coordinates to whole numbers before building its `id` strings,
   whereas the equivalent `get_pos()` method in
   `spatial_gene_expression_dataset.py` DOES round them
   (`np.around(...).astype(int)`) before doing the same thing. This is
   left unchanged here to exactly match your uploaded code, but could
   cause `id` string mismatches (e.g. "12.0x34.0" here vs "12x34"
   elsewhere) if you ever mix outputs from both functions together.
==============================================================================
"""

import os

# NOTE: `anndata` is imported twice here, under two different aliases
# (`ad` and `ann`) -- both point to the exact same library. This is
# redundant but harmless; see reproducibility note #5 near the top of
# this file.
import anndata as ad
import anndata as ann
import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
from PIL import Image
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score

# Raises the safety limit on how many total pixels an image is allowed to
# have before PIL/Pillow refuses to open it -- necessary here because
# whole-slide tissue images can be extremely large (see the same kind of
# safety-limit adjustment, with a different numeric value, in
# `spatial_gene_expression_dataset.py` earlier in this project).
Image.MAX_IMAGE_PIXELS = 933120000

# ------------------------------------------------------------------------
# Marker gene lists (see explanation #2 near the top of this file).
# Each all-caps variable below is a list of gene names well known, from
# prior biological research, to indicate the presence of one particular
# cell type. These specific genes correspond to cell types relevant to
# the skin-cancer dataset (`ViT_SKIN` / GSE144240) used elsewhere in this
# project.
# ------------------------------------------------------------------------
BCELL = ['CD19', 'CD79A', 'CD79B', 'MS4A1']              # B cells (immune)
TUMOR = ['FASN']                                          # Tumor cells
CD4T = ['CD4']                                            # CD4+ T cells (immune)
CD8T = ['CD8A', 'CD8B']                                   # CD8+ T cells (immune)
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E']         # Dendritic cells (immune)
MDC = ['LAMP3']                                           # Mature dendritic cells (immune)
CMM = ['BRAF', 'KRAS']                                    # Cutaneous Malignant Melanoma cells

# `IG` ("Interest Genes"?) groups every one of the marker-gene lists
# above under a readable cell-type name.
IG = {'B_cell': BCELL, 'Tumor': TUMOR, 'CD4+T_cell': CD4T, 'CD8+T_cell': CD8T, 'Dendritic_cells': DC,
      'Mature_dendritic_cells': MDC, 'Cutaneous_Malignant_Melanoma': CMM}

# One single flat list containing every individual marker gene from every
# cell type in `IG`, all combined together.
MARKERS = []
for i in IG.values():
    MARKERS += i

# A smaller subset of `IG`, covering only the lymphocyte (a category of
# immune cell) types.
LYM = {'B_cell': BCELL, 'CD4+T_cell': CD4T, 'CD8+T_cell': CD8T}


def read_tiff(path):
    """
    Opens a (possibly very large) TIFF image file, raising PIL's pixel
    safety limit first so it doesn't refuse to open large whole-slide
    tissue images.

    Parameters
    ----------
    path : str
        File path to the .tiff/.tif image.

    Returns
    -------
    PIL.Image.Image
        The opened image object.
    """
    Image.MAX_IMAGE_PIXELS = 933120000
    im = Image.open(path)

    # NOTE: `imarray` is computed here (converting the image into a NumPy
    # pixel array) but never actually used or returned -- this line has
    # no effect on the function's behavior. Kept unchanged here to
    # exactly match your uploaded code; feel free to remove it if you
    # don't need it, or change the `return` statement to `return imarray`
    # if you actually wanted the NumPy array instead of the raw PIL Image
    # object.
    imarray = np.array(im)
    # I = plt.imread(path)
    return im


def preprocess(adata, n_keep=1000, include=LYM, g=True):
    """
    Standard gene-expression preprocessing pipeline for an AnnData object,
    with THREE different possible ways to select which genes/features end
    up in the final result (see explanation #2 near the top of this file
    for background on marker genes). Exactly one of these three paths
    runs, chosen by the `g` and `include` arguments below.

    Parameters
    ----------
    adata : anndata.AnnData
        The raw input gene-expression data to preprocess.
    n_keep : int
        How many "highly variable genes" to keep, ONLY used in the final
        fallback path (see below).
    include : dict or None
        A dictionary mapping cell-type names to lists of marker genes
        (e.g. the `LYM` dictionary defined above), used ONLY in the
        middle path (see below). Defaults to `LYM`.
    g : bool
        If True (the default), use the FIRST path below (a pre-selected,
        fixed gene list loaded from disk), REGARDLESS of what `include`
        was set to -- `include` is silently ignored whenever `g=True`.

    Returns
    -------
    anndata.AnnData
        The preprocessed data, with normalization/log-transform applied,
        gene/feature selection applied (via whichever of the 3 paths ran),
        and normalized spatial coordinates added under
        `adata.obsm['position_norm']`.
    """
    # Make sure every gene name is unique (some datasets can contain
    # duplicate gene symbols) -- required by several downstream scanpy
    # functions that assume unique column labels.
    adata.var_names_make_unique()

    # Standard normalization + log transform (see explanation #1 near the
    # top of this file).
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    if g:
        # ---- Path 1: use a pre-selected, fixed list of gene names ----
        # Loaded from disk -- presumably a hand-picked or
        # previously-computed list of informative genes for the skin
        # dataset (note: `include` is completely ignored in this branch,
        # even if the caller passed something in for it).
        b = list(np.load('data/skin_a.npy', allow_pickle=True))
        adata = adata[:, b]
    elif include:
        # ---- Path 2: replace individual genes with marker-based
        # "cell-type scores" (see explanation #2 near the top of this
        # file) ----
        # Build a brand new matrix with ONE COLUMN PER CELL TYPE (instead
        # of one column per gene), where each cell type's column is the
        # AVERAGE expression of that cell type's marker genes, computed
        # separately for every spot (row).
        exp = np.zeros((adata.X.shape[0], len(include)))
        for n, (i, v) in enumerate(include.items()):
            tmp = adata[:, v].X               # pull out just this cell type's marker genes
            tmp = np.mean(tmp, 1).flatten()   # average them together, per spot
            exp[:, n] = tmp
        # Shrink `adata` down to the right NUMBER of columns (matching
        # how many cell types there are), purely so its shape lines up --
        # its actual column CONTENTS get completely overwritten just
        # below.
        adata = adata[:, :len(include)]
        adata.X = exp
        # Relabel the (now cell-type-score) columns with their
        # corresponding cell-type names instead of gene names.
        adata.var_names = list(include.keys())

    else:
        # ---- Path 3 (fallback): automatically select the most
        # "highly variable" genes ----
        # Same core idea as the pre-selected 1000-gene lists loaded from
        # disk in `spatial_gene_expression_dataset.py` earlier in this
        # project, except computed fresh here via scanpy's own built-in
        # highly-variable-gene detection, keeping the top `n_keep` genes.
        sc.pp.highly_variable_genes(adata, n_top_genes=n_keep, subset=True)

    # ---- Normalize spatial coordinates ----
    # `StandardScaler` rescales each coordinate axis (x and y separately)
    # to have mean 0 and standard deviation 1 -- the same "z-score"
    # normalization idea explained for gene expression in
    # `spatial_gene_expression_dataset.py`, applied here instead to the
    # raw pixel coordinates, so spatial positions from different samples
    # (which might otherwise span very different pixel ranges) become
    # more directly comparable.
    c = adata.obsm['spatial']
    scaler = preprocessing.StandardScaler().fit(c)
    c = scaler.transform(c)
    adata.obsm['position_norm'] = c

    return adata


def comp_umap(adata):
    """
    Runs a standard scanpy "dimensionality reduction + clustering"
    pipeline on an AnnData object: PCA -> nearest-neighbor graph -> UMAP
    (for 2D visualization) -> Leiden clustering (to automatically group
    similar spots together). See explanations #3 and #4 near the top of
    this file for what each step does and why.

    Parameters
    ----------
    adata : anndata.AnnData
        The (already preprocessed) gene-expression data.

    Returns
    -------
    anndata.AnnData
        The same object, now with `.obsm['X_pca']`, `.obsm['X_umap']`,
        and a new `.obs['clusters']` column added, containing each
        spot's automatically-discovered Leiden cluster label.
    """
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters")
    return adata


def comp_tsne_km(adata, k=10):
    """
    Runs PCA followed by t-SNE (for 2D visualization), then clusters the
    PCA-reduced data into a FIXED number `k` of groups using KMeans. See
    explanations #3 and #5 near the top of this file.

    Parameters
    ----------
    adata : anndata.AnnData
        The (already preprocessed) gene-expression data.
    k : int
        How many KMeans clusters to create (default: 10).

    Returns
    -------
    anndata.AnnData
        The same object, now with `.obsm['X_pca']`, `.obsm['X_tsne']`,
        and a new `.obs['kmeans']` column containing each spot's assigned
        cluster label (stored as a text string, e.g. `"0"`, `"1"`, ...).
    """
    sc.pp.pca(adata)
    sc.tl.tsne(adata)

    # `random_state=0` fixes KMeans's internal randomness for
    # reproducibility (see reproducibility note #2 near the top of this
    # file).
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(
        adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans.labels_.astype(str)
    return adata


def co_embed(a, b, k=10):
    """
    Combines TWO separate AnnData objects (typically: real "ground truth"
    expression, and a model's "predicted" expression -- the same pairing
    produced by `model_predict()` in `gene_expression_inference.py`
    earlier in this project) into ONE joint dataset, tags each row with
    which of the two it originally came from, then runs the same
    PCA + t-SNE + KMeans pipeline as `comp_tsne_km()` on the combined
    result. This is useful for visually and quantitatively checking
    whether the model's predicted expression patterns cluster similarly
    to the real, measured patterns.

    Parameters
    ----------
    a : anndata.AnnData
        The first dataset (e.g. ground-truth expression). Gets tagged
        `'Truth'`.
    b : anndata.AnnData
        The second dataset (e.g. predicted expression). Gets tagged
        `'Pred'`.
    k : int
        How many KMeans clusters to create in the combined embedding.

    Returns
    -------
    anndata.AnnData
        The combined dataset (`a` and `b` stacked together, row-wise),
        with a `.obs['tag']` column marking which original dataset each
        row came from, plus PCA/t-SNE/KMeans results just like
        `comp_tsne_km()`.
    """
    a.obs['tag'] = 'Truth'
    b.obs['tag'] = 'Pred'

    # `ad.concat([a, b])` stacks the two AnnData objects together,
    # row-wise (spot-wise), into one combined object -- this requires
    # both `a` and `b` to have the same columns (genes), which they
    # should, since they represent predictions vs. ground truth for the
    # exact same genes.
    adata = ad.concat([a, b])
    sc.pp.pca(adata)
    sc.tl.tsne(adata)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans.labels_.astype(str)
    return adata


def build_adata(name='H1'):
    """
    Manually builds a single AnnData object for one HER2ST sample,
    directly from the raw image/count/position files on disk -- covering
    much of the same ground as the file-reading helper methods
    (`get_img`, `get_cnt`, `get_pos`, `get_meta`) inside the
    `ViT_HER2ST` class in `spatial_gene_expression_dataset.py`, but
    packaged here as one standalone, simpler function (with no
    train/test splitting, graph construction, or image-patch extraction
    -- this is likely meant for quick, ad-hoc inspection/visualization of
    one sample at a time, rather than for feeding a model).

    Parameters
    ----------
    name : str
        Which HER2ST sample to load (e.g. `'H1'`).

    Returns
    -------
    (adata, im) : (anndata.AnnData, PIL.Image.Image)
        `adata` holds this sample's (normalized + log-transformed) gene
        expression for a pre-selected gene list, with spot pixel
        coordinates attached under `.obsm['spatial']`. `im` is the raw,
        full tissue slide image.
    """
    cnt_dir = 'data/her2st/data/ST-cnts'
    img_dir = 'data/her2st/data/ST-imgs'
    pos_dir = 'data/her2st/data/ST-spotfiles'

    # Load the full tissue slide image (same folder-naming convention as
    # `ViT_HER2ST.get_img()` in `spatial_gene_expression_dataset.py`).
    pre = img_dir + '/' + name[0] + '/' + name
    fig_name = os.listdir(pre)[0]
    path = pre + '/' + fig_name
    im = Image.open(path)

    # Load the raw gene-count table for this sample.
    path = cnt_dir + '/' + name + '.tsv'
    cnt = pd.read_csv(path, sep='\t', index_col=0)

    # Load the spot pixel-position table for this sample.
    path = pos_dir + '/' + name + '_selection.tsv'
    df = pd.read_csv(path, sep='\t')

    x = df['x'].values
    y = df['y'].values

    # Build a text `id` string ("123x456" style) per spot, so the count
    # table and position table can be matched up row-by-row. NOTE: unlike
    # `ViT_HER2ST.get_pos()` in `spatial_gene_expression_dataset.py`, this
    # version does NOT round x/y to whole numbers first -- see
    # reproducibility note #6 near the top of this file for why this
    # could matter if you ever mix outputs from both functions.
    id = []
    for i in range(len(x)):
        id.append(str(x[i]) + 'x' + str(y[i]))
    df['id'] = id

    # Join the gene-count table and the position table together,
    # matched up by their shared `id` column.
    meta = cnt.join((df.set_index('id')))

    # Load a pre-selected gene list for this sample (a different, likely
    # larger/differently-curated list than the 1000-gene list used in
    # `spatial_gene_expression_dataset.py`).
    gene_list = list(np.load('data/her_g_list.npy'))

    # Apply the same library-size-normalize + log-transform preprocessing
    # explained in `spatial_gene_expression_dataset.py`, then wrap the
    # result in a fresh AnnData object.
    adata = ann.AnnData(scp.transform.log(scp.normalize.library_size_normalize(meta[gene_list].values)))
    adata.var_names = gene_list
    adata.obsm['spatial'] = np.floor(meta[['pixel_x', 'pixel_y']].values).astype(int)

    return adata, im


def get_data(dataset='bc1', n_keep=1000, include=LYM, g=True):
    """
    Downloads (or loads, if already cached locally by scanpy) one of
    several PUBLIC 10x Genomics Visium spatial-transcriptomics sample
    datasets, using scanpy's own built-in dataset-fetching function
    `sc.datasets.visium_sge()`, then runs it through the same
    `preprocess()` pipeline defined above.

    Parameters
    ----------
    dataset : str
        Which public dataset to load. `'bc1'` and `'bc2'` are two
        specific named shortcuts (both official 10x Genomics
        "Breast Cancer" demo datasets); any other string is passed
        straight through as the `sample_id` to
        `sc.datasets.visium_sge()`, letting you load ANY dataset name
        that function supports.
    n_keep, include, g : (see `preprocess()`'s docstring above)
        Passed straight through to `preprocess()`.

    Returns
    -------
    (adata, img_path) : (anndata.AnnData, str)
        The preprocessed gene-expression data, and the file path to this
        sample's high-resolution tissue image (extracted from
        scanpy's own metadata about the downloaded dataset).
    """
    if dataset == 'bc1':
        # `include_hires_tiff=True` tells scanpy to also download the
        # full-resolution microscope image, not just a smaller preview.
        adata = sc.datasets.visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1', include_hires_tiff=True)
        adata = preprocess(adata, n_keep, include, g)
        img_path = adata.uns["spatial"]['V1_Breast_Cancer_Block_A_Section_1']["metadata"]["source_image_path"]
    elif dataset == 'bc2':
        adata = sc.datasets.visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_2', include_hires_tiff=True)
        adata = preprocess(adata, n_keep, include, g)
        img_path = adata.uns["spatial"]['V1_Breast_Cancer_Block_A_Section_2']["metadata"]["source_image_path"]
    else:
        # Fallback: treat `dataset` itself as the literal sample ID to
        # request from scanpy.
        adata = sc.datasets.visium_sge(sample_id=dataset, include_hires_tiff=True)
        adata = preprocess(adata, n_keep, include, g)
        img_path = adata.uns["spatial"][dataset]["metadata"]["source_image_path"]

    return adata, img_path


def get_R(data1, data2, dim=1, func=pearsonr):
    """
    Computes a Pearson correlation (by default) between two AnnData
    objects' data matrices, either PER GENE (comparing predicted vs. real
    values for one gene, across all spots) or PER SPOT (comparing
    predicted vs. real values for one spot, across all genes), depending
    on `dim`. This is the exact function used in
    `train_and_evaluate_thitogene.py` to score how accurately a trained
    model's predictions matched real, measured gene expression -- see
    explanation #7 near the top of this file for the full background on
    what Pearson correlation measures.

    Parameters
    ----------
    data1, data2 : anndata.AnnData
        Two datasets to compare (e.g. predicted expression and real
        expression), expected to have the SAME shape (same number of
        spots and same number of genes, in the same order).
    dim : int
        Which axis to loop over and compare "across":
          - `dim=1` (default): loop over GENES (columns) -- for each
            gene, compare its values across ALL spots between `data1` and
            `data2`. This is the mode used elsewhere in this project.
          - `dim=0`: loop over SPOTS (rows) instead -- for each spot,
            compare its values across ALL genes between `data1` and
            `data2`.
    func : callable
        Which correlation function to use, defaulting to
        `scipy.stats.pearsonr` (which returns both a correlation
        coefficient AND a p-value for each comparison). Any other
        function with the same `(x, y) -> (statistic, p_value)` calling
        convention could be substituted here instead.

    Returns
    -------
    (r1, p1) : (np.ndarray, np.ndarray)
        `r1` holds one correlation value per gene (or per spot, if
        `dim=0`); `p1` holds the matching statistical p-value for each of
        those correlations (a smaller p-value means the observed
        correlation is less likely to have arisen purely by chance).
    """
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []

    # Loop over either genes (dim=1) or spots (dim=0), one at a time.
    for g in range(data1.shape[dim]):

        if dim == 1:
            # Compare gene `g`'s values across all spots.
            r, pv = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            # Compare spot `g`'s values across all genes.
            r, pv = func(adata1[g, :], adata2[g, :])

        r1.append(r)
        p1.append(pv)

    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1


def cluster(adata, label):
    """
    Evaluates how well UNSUPERVISED clustering (KMeans, on
    PCA-reduced data) recovers a set of REAL, known tissue-region labels
    (e.g. the pathologist-provided labels from
    `spatial_gene_expression_dataset.py`'s `get_lbl()`/`self.label`),
    using the Adjusted Rand Index (see explanation #6 near the top of
    this file). Spots explicitly marked `'undetermined'` (no confident
    real label available) are excluded from the comparison entirely.

    Parameters
    ----------
    adata : anndata.AnnData
        The (already preprocessed) gene-expression data for one sample.
    label : array-like of str
        One real, known tissue-region label per spot (same length/order
        as `adata`'s rows), with the special value `'undetermined'` used
        for spots that should be excluded from scoring.

    Returns
    -------
    (p, ari) : (np.ndarray of str, float)
        `p` is the KMeans-assigned cluster label for every KEPT
        (non-'undetermined') spot. `ari` is the Adjusted Rand Index
        score (rounded to 3 decimal places) comparing those KMeans
        clusters against the real labels for those same spots.
    """
    # Only keep spots that actually have a confident, real label.
    idx = label != 'undetermined'
    tmp = adata[idx]
    l = label[idx]

    # Standard PCA + t-SNE pipeline (same as `comp_tsne_km()` above),
    # applied only to the kept spots.
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)

    # Cluster into EXACTLY as many groups as there are distinct REAL
    # labels (`len(set(l))`) -- a fair, apples-to-apples comparison
    # between the automatic clustering and the real labeling scheme.
    # `n_init=20` runs KMeans 20 separate times (each with different
    # random starting points) and automatically keeps the best result,
    # reducing the chance of landing in a poor local solution purely by
    # bad luck; `random_state=0` still keeps the whole process
    # reproducible (see reproducibility note #2 near the top of this
    # file).
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0, n_init=20).fit(
        tmp.obsm['X_pca'])
    p = kmeans.labels_.astype(str)

    # Build a label array covering EVERY spot in the original (full,
    # unfiltered) `adata` -- excluded ('undetermined') spots get filled
    # in with a placeholder string (the string version of how many
    # distinct real labels there are), while kept spots get their actual
    # KMeans cluster assignment.
    lbl = np.full(len(adata), str(len(set(l))))
    lbl[idx] = p
    adata.obs['kmeans'] = lbl

    # Compare the KMeans clustering (`p`) against the real labels (`l`),
    # for just the kept spots, using the Adjusted Rand Index (see
    # explanation #6 near the top of this file).
    return p, round(ari_score(l, p), 3)
