"""
==============================================================================
spot_knn_graph_builder.py
==============================================================================

WHAT IS THIS FILE FOR? (explained from scratch, no prior background assumed)
------------------------------------------------------------------------------
This file builds a GRAPH out of a set of 2D (or higher-dimensional)
coordinates -- for example, the grid positions of tissue "spots" used
earlier in this project's `spatial_gene_expression_dataset.py`. This is the
missing piece that file depends on: it imports and calls `calcADJ()`
(defined below) to build the k-nearest-neighbor spot graph that later gets
fed into the Graph Attention layer (`graph_attention_layer.py`) elsewhere in
this project.

Background concept #1 -- What is a graph, and what is an "adjacency matrix"?
    As introduced in `graph_attention_layer.py`, a graph is a set of
    "nodes" (here: one node per tissue spot / one node per input
    coordinate) plus "edges" connecting certain pairs of nodes together. A
    convenient way to store WHICH pairs are connected is an "adjacency
    matrix": a square table with one row and one column per node, where
    the entry at row i, column j is 1 if node j is a neighbor of node i,
    and 0 otherwise. This file's whole job is to build exactly this table.

Background concept #2 -- What does "k-nearest-neighbors" (k-NN) mean?
    Instead of connecting every node to every other node (which would
    create a huge, mostly-uninformative graph), a much more useful
    strategy is: for every node, only connect it to its `k` CLOSEST
    neighbors (measured by some distance, e.g. straight-line/Euclidean
    distance). This keeps the graph "local" -- each node is only directly
    connected to the handful of other nodes that are physically nearest to
    it (e.g., for a tissue spot, its nearest neighboring spots on the
    slide) -- which is exactly the kind of local, spatially-meaningful
    connectivity a Graph Neural Network layer (like the GAT layer used
    elsewhere in this project) is designed to take advantage of.

Background concept #3 -- Why might we want to further "prune" (remove) some
of those k nearest-neighbor connections?
    Sometimes even a node's "k nearest" neighbors are still quite far away
    in absolute terms (for example, near the edge of the tissue, where
    there may not be many real neighbors nearby, so the k-th closest
    neighbor might genuinely be quite distant). Connecting to a neighbor
    that is unusually far away compared to everyone else's neighbors can
    introduce noisy, not-very-meaningful edges into the graph. This file
    supports three different strategies (controlled by the `pruneTag`
    argument) for whether to keep every one of the k nearest connections,
    or to additionally filter out ones that are "too far" by some rule --
    see the full explanation of each strategy below.

------------------------------------------------------------------------------
THE ALGORITHM, EXPLAINED STEP BY STEP
------------------------------------------------------------------------------
`calcADJ()` takes in `coord` (one row of coordinates per node -- e.g. one
(x, y) grid position per tissue spot) and returns a square adjacency matrix.
Here is exactly what it does, for EVERY node `i`, one at a time:

1) Measure the distance from node `i` to every other node (INCLUDING
   itself).
   ------------------------------------------------------------------------
   `distance.cdist(tmp, spatialMatrix, distanceType)` (from the `scipy`
   library) computes the distance between one point (`tmp`, node i's own
   coordinates) and every point in `spatialMatrix` (all nodes, node i
   included), all in a single efficient call. The result, `distMat`, is a
   1-row table of distances: `distMat[0, j]` = distance from node i to
   node j. The default `distanceType='euclidean'` means ordinary
   straight-line distance (the same "as the crow flies" distance you'd
   compute with the Pythagorean theorem), but `scipy.spatial.distance.cdist`
   also supports many other distance definitions if you ever need them
   (e.g. `'cityblock'`, `'cosine'`).

2) Sort all other nodes by distance, closest first, and take the top `k`.
   ------------------------------------------------------------------------
   `distMat.argsort()` returns the INDEX of each node in order from
   closest to farthest (rather than the distances themselves). The very
   first entry (index 0 after sorting) will always be node `i` itself,
   since a node's distance to itself is 0 -- the smallest possible
   distance. That's why the code takes indices `res[0][1:k+1]` (skipping
   position 0, which is always itself) to get the `k` TRUE nearest
   neighbors, excluding the node itself.
   Special case: if `k == 0` is passed in, the code instead treats that as
   "connect to literally every other node" by setting
   `k = spatialMatrix.shape[0] - 1` (every node except itself).

3) Compute a distance "boundary" for the optional statistical pruning mode.
   ------------------------------------------------------------------------
   `boundary = mean(neighbor distances) + std(neighbor distances)`.
   This is a common, simple statistical rule of thumb: "how far is
   further than about one standard deviation above the average neighbor
   distance for this particular node". It's computed for every node
   regardless of which `pruneTag` mode is actually used, but it is only
   ever actually USED when `pruneTag == 'STD'` (see step 4c below).

4) For each of the `k` nearest neighbors, decide whether to actually
   create an edge, depending on `pruneTag`:
   ------------------------------------------------------------------------
   a) `pruneTag == 'NA'` ("No prune"):
        Always create the edge. Every one of the k nearest neighbors gets
        connected, no matter how far away it is. This is the simplest and
        most permissive option, and (looking at how the dataset-loading
        code elsewhere in this project calls `calcADJ`) it is the default
        behavior actually used throughout this project.
   b) `pruneTag == 'STD'` ("Standard-deviation prune"):
        Only create the edge if this particular neighbor's distance is
        less than or equal to the `boundary` computed in step 3. This
        removes unusually-far-away neighbors on a PER-NODE basis (each
        node gets its own boundary, based on its own neighbors' spread),
        which can help avoid connecting a node to a neighbor that, while
        technically one of its k closest, is still much farther away than
        its other nearby neighbors (e.g. near the edge of the tissue).
   c) `pruneTag == 'Grid'` ("Fixed grid-distance prune"):
        Only create the edge if the neighbor's distance is less than or
        equal to a fixed threshold of `2.0`. This mode makes the most
        sense when your input coordinates are already whole-number GRID
        positions (like row/column indices on a regular lattice), where
        immediately-adjacent grid cells sit at a known, fixed distance
        apart -- so a fixed cutoff (rather than the per-node statistical
        `STD` rule) reliably keeps only true "physically touching"
        grid neighbors.

5) Repeat this whole process independently for every node, filling in one
   row of the adjacency matrix at a time, and return the completed matrix.

Important note on symmetry: because each node's k nearest neighbors are
decided independently, it's possible for node A to consider node B one of
its k nearest neighbors, while node B does NOT consider node A one of its
own k nearest neighbors (this can happen in regions of uneven point
density). This means the resulting adjacency matrix is NOT guaranteed to
be perfectly symmetric (i.e. `Adj[i][j]` is not always equal to
`Adj[j][i]`). This is expected, standard behavior for a plain k-NN graph,
and downstream layers (like the GAT layer used elsewhere in this project)
are built to handle a directed/asymmetric adjacency matrix without any
issue.

------------------------------------------------------------------------------
COMPUTATIONAL COST (for anyone reproducing on a large dataset)
------------------------------------------------------------------------------
For every one of the `nodes` input points, this function computes its
distance to ALL other points (an O(nodes) operation) and sorts them
(an O(nodes * log(nodes)) operation). Doing this once per node gives an
overall cost that grows roughly with `nodes^2 * log(nodes)`. This is
perfectly fine for a single tissue sample's spot count (typically a few
hundred to a few thousand spots), but would become slow for extremely
large point sets (tens of thousands or more), in which case a specialized
approximate/optimized k-NN library (e.g. `scikit-learn`'s
`NearestNeighbors`, or `scipy.spatial.cKDTree`) would compute the same
kind of result much faster.

------------------------------------------------------------------------------
HOW TO REPRODUCE RESULTS WITH THIS FILE
------------------------------------------------------------------------------
1. Required Python packages: numpy, torch, scipy.

2. This function is FULLY DETERMINISTIC -- it uses no random numbers
   anywhere. Given the exact same `coord` input array and the exact same
   `k` / `distanceType` / `pruneTag` arguments, it will always produce
   the exact same adjacency matrix, every single time, on any machine.
   (The only theoretical source of run-to-run variation would be if two
   or more neighbors are tied at EXACTLY the same distance from a node,
   in which case `argsort()`'s tie-breaking order could depend on the
   internal memory order of the input array -- an extremely rare edge
   case in real-world floating point coordinate data.)

3. To match how this function is actually called elsewhere in this
   project (inside `spatial_gene_expression_dataset.py`), use:
       calcADJ(coord=spot_xy_coordinates, k=4, pruneTag='NA')
   i.e. connect every spot to its 4 nearest neighbors, with no additional
   pruning.
==============================================================================
"""

import numpy as np
import torch
from scipy.spatial import distance


def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    """
    Builds a k-nearest-neighbor adjacency matrix from a set of coordinates.
    See the big step-by-step explanation near the top of this file for the
    full plain-language walkthrough of everything below.

    Parameters
    ----------
    coord : array-like, shape (num_nodes, num_dimensions)
        One row of coordinates per graph node (e.g. one (x, y) grid
        position per tissue spot). `num_dimensions` is usually 2, but this
        function works for any number of coordinate dimensions.
    k : int
        How many nearest neighbors to connect each node to. If `k == 0`,
        this is treated as a special case meaning "connect every node to
        every other node" (see step 2 in the explanation above).
    distanceType : str
        Which distance metric to use when measuring "closeness" between
        two points. Default `'euclidean'` = ordinary straight-line
        distance. Passed straight through to `scipy.spatial.distance.cdist`,
        which supports many other options too (e.g. `'cityblock'`,
        `'cosine'`, `'chebyshev'`) if a different notion of distance is
        ever needed.
    pruneTag : str
        Which pruning strategy to apply on top of the k-nearest-neighbor
        selection. One of:
          - `'NA'`   : no pruning -- always connect all k nearest
                       neighbors (this is the default used throughout the
                       rest of this project).
          - `'STD'`  : only connect neighbors whose distance is within one
                       standard deviation above this node's own average
                       neighbor distance.
          - `'Grid'` : only connect neighbors whose distance is <= 2.0
                       (intended for regular whole-number grid
                       coordinates).

    Returns
    -------
    torch.Tensor, shape (num_nodes, num_nodes)
        The adjacency matrix. Entry [i, j] == 1.0 means "node j is one of
        node i's kept nearest neighbors"; entry [i, j] == 0.0 means "not
        connected". Note this matrix is not guaranteed to be symmetric --
        see the "Important note on symmetry" paragraph near the top of
        this file.
    """
    # Just an alias/rename for clarity -- `coord` holds the raw spatial
    # coordinates for every node.
    spatialMatrix = coord

    # Total number of nodes (graph points) we need to build connections
    # for -- one row of `coord` per node.
    nodes = spatialMatrix.shape[0]

    # Start with an all-zeros (nodes x nodes) table. We will fill in a 1.0
    # wherever we decide an edge should exist. Using a `torch.Tensor` here
    # (rather than a plain numpy array) is what lets this adjacency matrix
    # be fed directly into a PyTorch-based Graph Neural Network layer
    # afterward, with no extra conversion step needed.
    Adj = torch.zeros((nodes, nodes))

    # Process one node at a time, filling in that node's row of the
    # adjacency matrix (i.e. deciding which OTHER nodes it connects to).
    for i in np.arange(spatialMatrix.shape[0]):
        # This node's own coordinates, reshaped into a single 1-row table
        # (required shape for `distance.cdist` below).
        tmp = spatialMatrix[i, :].reshape(1, -1)

        # Distance from this one node to EVERY node (including itself).
        # Shape: (1, nodes) -- one row, one distance value per node.
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)

        # Special case: k == 0 means "no fixed neighbor count -- just
        # connect to literally every other node in the graph".
        if k == 0:
            k = spatialMatrix.shape[0] - 1

        # Sort ALL node indices by distance, closest first. `res[0]` is
        # this ordering; the very first entry is always this node itself
        # (distance 0), so we grab the closest `k` neighbors AFTER
        # skipping that self-reference in the slicing steps below.
        res = distMat.argsort()[:k + 1]

        # Grab the actual distance VALUES (not just indices) for this
        # node's true k nearest neighbors (again skipping index 0, which
        # is the node itself).
        tmpdist = distMat[0, res[0][1:k + 1]]

        # Statistical "how far is unusually far" cutoff for this node,
        # only actually used if pruneTag == 'STD' (see step 3/4b in the
        # explanation above).
        boundary = np.mean(tmpdist) + np.std(tmpdist)

        # Walk through this node's k nearest neighbors (positions 1
        # through k in the sorted order -- again, position 0 is skipped
        # because it is the node itself) and decide, one at a time,
        # whether to actually record that connection in the adjacency
        # matrix, according to the chosen pruning rule.
        for j in np.arange(1, k + 1):
            if pruneTag == 'NA':
                # No pruning: always connect.
                Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'STD':
                # Only connect if this neighbor isn't unusually far away
                # compared to this node's other neighbors.
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'Grid':
                # Only connect if this neighbor is within a fixed distance
                # of 2.0 (meant for regular whole-number grid
                # coordinates).
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0

    return Adj
