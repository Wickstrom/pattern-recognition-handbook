# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
#
# Marimo version of the Data transformation and dimensionality reduction
# II lecture (Kernel PCA + Laplacian Eigenmaps).
# Same content as notebooks/08/non_linear_dim_reduction.ipynb, but
# authored as a reactive Marimo app (one slide per concept, matching
# the density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/08/non_linear_dim_reduction.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `X`) can be redefined
# in any number of cells, but Marimo requires each name to be owned by
# exactly one cell. Below, every cell-local dataset and figure is
# given a unique suffix (`_moons_kpca`, `_swiss_roll`) so the slides
# stay independent and the data each figure displays is reproducible.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/non_linear_dim_reduction.slides.json",
    css_file="../_shared/math.css",
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
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Data transformation and dimensionality reduction (DTDR) II
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Last lecture we looked at compression of data.
        - Reduce the dimensionality.
    - Learned about two methods:
        - Fisher Discriminant Analysis (need labeled data).
        - Principal Component Analysis (PCA).
    - Both methods use a linear transformation.
    - Today, we will learn about non-linear DTDR techniques:
        - Kernel PCA (1998)
        - Laplacian Eigenmaps (2003)
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Kernel PCA

    - Non-linear operations with respect to $\mathbf{x}$.
    - Will only need inner-products in $\mathbb{H}$
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Kernel matrix
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Eigenvectors of autocorrelation matrix R

    - PCA idea:
        Diagonal covariance matrix for transformed data.
    - Revisit to motivate kernel PCA.
    - Easier notation to work with autocorrelation matrix $\mathbf{R}$.
    - Eigendecomposition: $\mathbf{R} \mathbf{v} = \lambda \mathbf{v}$
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Eigenvectors of autocorrelation matrix R

    - First: $\lambda \mathbf{v}=$
    - Reordering for $\mathbf{v}= $
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Projection onto eigenvector

    - $\mathbf{v}^T\phi(\mathbf{x}_t) = $
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Finding the weights

    - Given by solving $\mathbf{K} \mathbf{a} = \tilde{\lambda} \mathbf{a}$
    - $\tilde{\lambda}=\lambda N$

    ---

    - Remark: Eigenvectors of symmetric $\mathbf{R}$ assumed $\|\mathbf{v}_i\|_2 = 1$ and $\mathbf{v}_i^T \mathbf{v}_j = 1$:

    - Have $\mathbf{v}^T \mathbf{v} = $
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Visualizing in $\mathbb{H}$
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 23</div>
    """
    )
    return


@app.cell
def _(mo, plt):
    # Side-by-side comparison of PCA vs Kernel PCA on the two-moons
    # dataset: PCA cannot separate the half-moons in 2-D while an
    # RBF-kernel PCA can unroll them into a linearly separable shape.
    from sklearn.datasets import make_moons
    from sklearn.decomposition import PCA, KernelPCA

    X_moons_kpca, y_moons_kpca = make_moons(n_samples=200, noise=0.05, random_state=42)

    pca_moons = PCA(n_components=2)
    x_pca_moons = pca_moons.fit_transform(X_moons_kpca)

    kpca_moons = KernelPCA(n_components=2, kernel="rbf", gamma=28)
    x_kpca_moons = kpca_moons.fit_transform(X_moons_kpca)

    fig_moons_kpca, (ax_moons_orig, ax_moons_pca, ax_moons_kpca) = plt.subplots(1, 3, figsize=(15, 5))
    ax_moons_orig.scatter(X_moons_kpca[:, 0], X_moons_kpca[:, 1], c=y_moons_kpca,
                          cmap="viridis", edgecolor="k", s=50)
    ax_moons_orig.set_title("Original Space")
    ax_moons_pca.scatter(x_pca_moons[:, 0], x_pca_moons[:, 1], c=y_moons_kpca,
                         cmap="viridis", edgecolor="k", s=50)
    ax_moons_pca.set_title("PCA")
    ax_moons_kpca.scatter(x_kpca_moons[:, 0], x_kpca_moons[:, 1], c=y_moons_kpca,
                          cmap="viridis", edgecolor="k", s=50)
    ax_moons_kpca.set_title("Kernel PCA")

    plt.close(fig_moons_kpca)
    mo.vstack([
        mo.as_html(fig_moons_kpca),
        mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 23</div>"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Laplacian Eigenmaps

    - Make use of the eigenvalues (spectrum) of the similarity matrix of the data
        - Spectral methods.
    - Motivated from the perspective of manifold learning.
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 23</div>
    """
    )
    return


@app.cell
def _(mo, plt):
    # Swiss roll in 3-D — the canonical example for manifold learning.
    # The colour encodes the position along the roll and helps show how
    # Laplacian Eigenmaps can "unroll" it back to 2-D.
    from sklearn.datasets import make_swiss_roll

    X_swiss, y_swiss = make_swiss_roll(n_samples=2000, noise=0.01, random_state=42)

    fig_swiss = plt.figure(figsize=(8, 6))
    ax_swiss = fig_swiss.add_subplot(projection="3d")
    ax_swiss.scatter3D(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=y_swiss,
                       cmap="viridis", edgecolor="k", s=50)

    plt.close(fig_swiss)
    mo.vstack([
        mo.as_html(fig_swiss),
        mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 23</div>"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Laplacian Eigenmaps - key idea

    - Transform $\mathbf{x} \in \mathbb{R}^d$ into $\mathbf{z} \in \mathbb{R}^k$, where $k \leq d$
    - Such that: neighbors in $\mathbb{R}^d$ stay neighbors in $\mathbb{R}^k$
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Adjacency matrix

    - What makes points neighbors?
        - Similarity!
    - Draw $\rightarrow$
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Mapping $\mathbf{x}_i$ to $\mathbf{z}_i$

    - Need a way to optimize the position of points in $\mathbb{R}^k$.
    - Consider the following: $J(\mathbf{z}) = \frac{1}{2} \sum_i \sum_j (\mathbf{z}_i-\mathbf{z}_j)W_{ij}$
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Optimizing the position of points in $\mathbb{R}^k$

    - Reasonable to minimize $J(\mathbf{z})$ with respect to $\mathbf{z}$
    - Obvious solution not interesting.
        - What is the obvious solution?
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Optimizing the position of points in $\mathbb{R}^k$

    - Have: $J(\mathbf{z}) = \frac{1}{2}$
    - Let $D_{ii} = $
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Optimizing the position of points in $\mathbb{R}^k$

    - Hence: $J(\mathbf{z}) = \mathbf{z}^T (\mathbf{D}-\mathbf{W}) \mathbf{z} = $
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Laplacian matrix $\mathbf{L}$

    - Deep theory!
    - The Laplacian tells us the divergence of the gradients.
        - How much is the steepness changing.
    - Analogous to the second derivative for a continuous, single variate function.
    - [**Nice resource for further information**](https://mbernste.github.io/posts/laplacian_matrix/)
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Optimizing the position of points in $\mathbb{R}^k$

    - Constrain $\mathbf{z}$: $\mathbf{z}^T \mathbf{D} \mathbf{z} = 1$
    - Removes arbitrary scaling in embeddings.
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Laplacian eigenmaps optimization problem

    - Minimize $\mathbf{z}^T \mathbf{L} \mathbf{z}$
    - Subject to $\mathbf{z}^T \mathbf{D} \mathbf{z} = 1$
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Lagrange function for Laplacian Eigenmaps

    Lagrange function:
    $$
    L(\mathbf{z}, \lambda) = \mathbf{z}^T \mathbf{L} \mathbf{z} - \lambda \mathbf{z}^T \mathbf{D} \mathbf{z}
    $$

    Set derivative to zero:
    $$
    \frac{\partial L}{\partial \mathbf{z}} = 2\mathbf{L}\mathbf{z} - 2\lambda \mathbf{D}\mathbf{z} = 0
    $$

    Which gives:
    $$
    \mathbf{L}\mathbf{z} = \lambda \mathbf{D}\mathbf{z}
    $$

    - The minimization of this objective is given by eigenvector correspond to the second smallest eigenvalue
    - Assuming a connected graph.
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 23</div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Closing perspective

    Let $\mathbf{P} = \mathbf{D}^{-1} \mathbf{W}$.

    **Example:**

    $$
    \mathbf{W} =
    \begin{bmatrix}
    2 & 3 & 1 \\
    3 & 4 & 0 \\
    1 & 0 & 2
    \end{bmatrix}
    ,\quad
    \mathbf{D} =
    \begin{bmatrix}
    6 & 0 & 0 \\
    0 & 7 & 0 \\
    0 & 0 & 3
    \end{bmatrix}
    $$

    $$
    \mathbf{D}^{-1} =
    \begin{bmatrix}
    \frac{1}{6} & 0 & 0 \\
    0 & \frac{1}{7} & 0 \\
    0 & 0 & \frac{1}{3}
    \end{bmatrix}
    $$

    $$
    \mathbf{P} =
    \begin{bmatrix}
    \frac{2}{6} & \frac{3}{6} & \frac{1}{6} \\
    \frac{3}{7} & \frac{4}{7} & 0 \\
    \frac{1}{3} & 0 & \frac{2}{3}
    \end{bmatrix}
    $$

    Probabilities of a "walk" from $x_i$ to $x_j$.
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 23</div>
    """
    )
    return


if __name__ == "__main__":
    app.run()
