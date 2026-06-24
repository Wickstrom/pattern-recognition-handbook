# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///
#
# Marimo version of the Function-optimization clustering lecture
# (K-means + spectral clustering).
# Same content as notebooks/12/clustering.ipynb, but authored as a
# reactive Marimo app (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/12/clustering.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/clustering.slides.json",
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
    # Clustering 3 - Function optimization
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Previous lectures:
        - Sequential and hierarchical clustering
    - Today, clustering through function optimization:
        - Generalized hard algorithmic scheme (GHAS) -> K-means.
        - Spectral clustering (Laplace).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Generalized hard algorithmic scheme (GHAS)

    - Recall:
    - Hard memberships: $ \mu_{ij} \in \{0, 1\}, \hspace{1.0cm} \sum_{j=1}^m mu_{ij} = 1$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Clustering process

    - Determine "correct" $\mu_{ij}$
    - Find representative cluster **parameters** or **points**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Cost function

    - Let $\mathbf{U} = [\mathbf{u}_1, \cdots , \mathbf{u}_N]$ and $\boldsymbol{\Theta}=[\boldsymbol{\theta}_1, \cdots , \boldsymbol{\theta}_N]$
    - $ \underset{\mathbf{U}, \boldsymbol{\Theta}}{\operatorname{min}}\ J(\mathbf{U}, \boldsymbol{\Theta}) \sum\limits_{i=1}^{N} \sum\limits_{j=1}^M n_{ij} d(\mathbf{x}_i, \boldsymbol{\theta}_j)$
    - where $d(\mathbf{x}_i, \boldsymbol{\theta}_j)$: Dissimilarity measure
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks to cost function

    - Will assing $\mathbf{x}_i$ to $\boldsymbol{\theta}_j$ if they are **close**.
    - Discrete variables - not differentiable!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Key idea - two-stage clustering

    - First step:
        - Assume $\boldsymbol{\Theta}$ fixed: $\boldsymbol{\Theta}^*$.
        - $ \underset{\mathbf{U}}{\operatorname{min}}\ J(\mathbf{U}, \boldsymbol{\Theta}^*):$
    - Second step:
        - Fix $\mathbf{U}$: $\mathbf{U}^*$
        - $ \underset{\mathbf{\boldsymbol{\Theta}}}{\operatorname{min}}\ J(\mathbf{U}^*, \boldsymbol{\Theta}):$
    - Have:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## K-means

    - Let $d(\mathbf{x}_i, \boldsymbol{\theta}_j) = \| \mathbf{x}_i - \boldsymbol{\theta}_j \|^2$
    - Hence: $\frac{\partial}{\partial \boldsymbol{\theta}_k} J(\mathbf{U}^*, \boldsymbol{\Theta})$
    - $\boldsymbol{\theta}_k=$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### K-means algorithm

    - Initialize $\boldsymbol{\theta}_j (\mathbf{m}_j)$.
    - Assign $\mathbf{x}_i$ to **closest** $\boldsymbol{\theta}_j$ (finds $u_{ij}$)
    - Re-compute $\boldsymbol{\theta}_j$ as mean of cluster.
    - Iterate for numerous iterations until cluster assignments stop changing.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks to K-means

    - Ideal for spherical and compact clusters
    - Creates linear cluster boundaries.
    - Can have scale problems:
        - But modern implementation have batched versions.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Spectral clustering (Laplace)

    - Graph: V $\rightarrow$
        - Undirected
        - Connected
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Proximity (affinty) matrix

    - $ \mathbf{W} = [W(i,j)], \hspace{1.0cm} i,j=1,\cdots,N $
    - Symmetric!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example of spectral clustering

    - Cut the graph in two parts!
    - Let $W(i,j)=\exp(-\frac{1}{2\sigma^2}\| \mathbf{x_i} - \mathbf{x_j} \|^2)$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Different ways to cut graph

    - RatioCut
    - NormalizedCut
    - InformationCut
        """
    )
    return


if __name__ == "__main__":
    app.run()
