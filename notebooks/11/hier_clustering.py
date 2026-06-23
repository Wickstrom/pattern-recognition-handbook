# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///
#
# Marimo version of the Hierarchical clustering lecture.
# Same content as notebooks/11/hier_clustering.ipynb, but authored as a
# reactive Marimo app (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/11/hier_clustering.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/hier_clustering.slides.json",
)


@app.cell
def _():
    import os
    from pathlib import Path

    # Make `mo.image(src="media/foo.png")` resolve regardless of which
    # directory `marimo edit` is launched from. We chdir to the notebook's
    # own directory so the relative `media/` path works the same way it
    # does for the deployed WASM build (where the page sits next to
    # `media/`). In WASM, `__file__` may not be set and the chdir becomes
    # a no-op — the browser still fetches the images via the page URL.
    if "__file__" in globals() and __file__:
        try:
            os.chdir(Path(__file__).resolve().parent)
        except OSError:
            pass

    import marimo as mo
    return Path, mo, os


@app.cell
def _(mo):
    mo.md(
        r"""
    # Clustering 2 - Hierarchical clustering
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Last lecture we looked at sequential clustering.
        - Start from on data point and sequentially determine clusterings.
    - Today, we take alternative routes:
        - Divisive
        - Agglomerative
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example

    - Clustering in modern computer vision
    - [**DINOv3**](https://arxiv.org/pdf/2304.07193)
    - [**Automatic Data Curation for Self-Supervised Learning:A Clustering-Based Approach**](https://arxiv.org/pdf/2405.15613)
        """
    )
    return


@app.cell
def _(mo):
    # Cluster example from the DINOv3 / automatic-data-curation paper:
    # shows what large-scale clustering produces on real image data.
    mo.image(src="media/cluster_ex.png", width="600px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Divisive

    - Core idea: start will all points in one clusters and split based on proxmity.
    - Draw:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Agglomerative

    - Core idea: all points are a cluster and merge hierarchically.
    - Draw:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks on agglomerative

    - Nested clusterings!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Agglomerative algorithms

    - $g(c_i, cj) \rightarrow$ proximity between clusters $c_i$ and $c_j$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Generalized agglomerative scheme (GAS)

    - Initialize: $R_0 = \{ c_i = \{ \mathbf{x_i}\}, i=1,\cdots,N\}$
    - Pseudocode:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Matrix-based implementation

    - Note: $g(c_i, cj)$ formed from proximity between data points.
    - Will focus on dissimilarity
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Dissimilarity matrix

    - Define $\mathbf{P} (N \times N)$ such that $P_{ij}=d(\mathbf{x}_i, \mathbf{x}_j)$.
        - Symmetric
        - $P_ii=d_0$ (smallest value along diagonal)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Matrix updating algorithmic scheme (MUAS)

    - Pseudocode:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Single link versus complete link

    - Single link algorithm:
    - Complete link algorithm:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example single link

    - $$ \mathbf{P}_0 =  $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Single link dendrogram

    - Draw dendrogram
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example complete link

    - $$ \mathbf{P}_0 =  $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Complete link dendrogram

    - Draw dendrogram
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Computational complexity

    - For large datasets, this procedure can become computationally demanding.
    - Many splits if N gets large.
        """
    )
    return


if __name__ == "__main__":
    app.run()
