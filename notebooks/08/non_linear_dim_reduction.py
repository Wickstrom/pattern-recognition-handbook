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

__generated_with = "0.23.13"
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

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt


    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Data transformation and dimensionality reduction (DTDR) II


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Recall PCA
    - Dataset with $N$ observations $\mathbf{x}_1,...,\mathbf{x}_N$ with $\mathbf{x}_i\in\mathbb{R}^d$
    - Aim: reduce dimensionality!
    - How: project $\mathbf{x}_i$ into the direction $\mathbf{v}$ along with the variance of the data is maximum $\rightarrow$ ${z}_i=\mathbf{v}^T\mathbf{x}_i$
    - Direction that maximise the variance: $\Sigma\mathbf{v}=\lambda\mathbf{v}$, with $\mathbf{\Sigma}$ covariance matrix $\rightarrow$ **first eigenvector of the covariance matrix**.
    """)
    return


@app.cell
def _(make_moons, mo):
    X_demo, y_demo = make_moons(n_samples=90, noise=0.055, random_state=7)

    pca_angle = mo.ui.slider(start=0, stop=180, step=5, value=45,
                             label="Angle", show_value=True)
    pca_play = mo.ui.switch(value=False, label="Play rotation")
    pca_clock = mo.ui.refresh(options=["1s"], default_interval="1s",
                              label="Animation clock")
    return X_demo, pca_angle, pca_clock, pca_play, y_demo


@app.cell
def _(X_demo, mo, np, pca_angle, pca_clock, pca_play, plt, y_demo):
    _ticks = int(pca_clock.value.split("(")[-1].rstrip(")")) if pca_clock.value else 0
    _theta_deg = (5 * _ticks) % 180 if pca_play.value else pca_angle.value
    _theta = np.deg2rad(_theta_deg)
    _e = np.array([np.cos(_theta), np.sin(_theta)])
    _xc = X_demo - X_demo.mean(axis=0)
    _scores = _xc @ _e
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _ax0.scatter(_xc[:, 0], _xc[:, 1], c=y_demo, cmap="coolwarm", s=19)
    _line = np.linspace(-1.8, 1.8, 2)
    _ax0.plot(_line*_e[0], _line*_e[1], color="black", lw=2)
    _ax0.set_aspect("equal"); _ax0.set_title("Centered data and axis")
    _ax1.scatter(_scores, np.zeros_like(_scores), c=y_demo, cmap="coolwarm", s=21)
    _ax1.set_yticks([]); _ax1.set_title(f"Angle {_theta_deg}° · variance {np.var(_scores):.3f}")
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### PCA projections are linear wrt the original datapoints; this can be not enough when data represents non linear structures.

    $$\mathbf{e}(\theta)=(\cos\theta,\sin\theta)^T,\qquad
    z_i=\mathbf{e}(\theta)^T\widetilde{\mathbf{x}}_i.$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 26</div>
    """),
               mo.hstack([pca_angle, pca_play, pca_clock], justify="start"),
               mo.as_html(_fig)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Kernel PCA

    - Non-linear operations with respect to $\mathbf{x}$.
    - Let $\phi:\mathbb{R}^d\rightarrow \mathbb{H}$ be a data transformation function from $\mathbb{R}^d$ to the feature space $\mathbb{H}$
    - KPCA $\rightarrow$ apply PCA to transformed points $\phi (\mathbf{x}_1),...,\phi (\mathbf{x}_N)$


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Eigenvectors of autocorrelation matrix R

    - Let $\mathbf{R}=\frac{1}{N}\sum_{i=1}^N \phi(\mathbf{x}_i)\phi(\mathbf{x}_i)^T$ be the autocorrelation matrix in the feature space $\mathbb{H}$;
    - Eigendecomposition: $\mathbf{R} \mathbf{v} = \lambda \mathbf{v}$
    - $\mathbf{R}$ can have very high dimension $\rightarrow$ construct it and diagonalize it can be computationally infeasible!
    - **Trick**: 1) we can show that the eigenvectors associated to non-null eigenvalues of $\mathbf{R}$ are linear combination of the features; 2) the coefficient of this linear combination can be obtained from the kernel matrix $\mathbf{K}$.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1) Eigenvectors of $\mathbf{R}$ are linear combination of the features.

    - $\mathbf v=\sum_{i=1}^N a_i\phi(\mathbf x_i)$ with $a_i=\frac1{N\lambda} \phi(\mathbf x_i)^T\mathbf{v}$


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2) How to obtain the coefficient of this linear combination

    - Introduce the kernel function $k(\mathbf x_i,\mathbf x_j)=\phi(\mathbf x_i)^T\phi(\mathbf x_j)$ and the associated kernel matrix $\mathbf{K}$.
    - $\mathbf{K}\mathbf{a}=N\lambda\mathbf{a}$


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### KPCA \[
    \mathbf R\mathbf v=\lambda\mathbf v
    \]



    \[
    \mathbf v=\sum_i a_i\phi(\mathbf x_i)
    \]

    For finding the coefficients:

    \[
    \mathbf K\mathbf a
    =
    \tilde\lambda\mathbf a
    \]

    with:

    \[
    \tilde\lambda=N\lambda
    \]

    Finally, we project:

    \[
    z(\mathbf x)=\mathbf v^T\phi(\mathbf x)
    \]

    Substituting:

    \[
    \boxed{
    z(\mathbf x)=
    \sum_i a_i k(\mathbf x_i,\mathbf x)
    }
    \]


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Visualizing in $\mathbb{H}$


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 23</div>
    """)
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
    return (make_moons,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Laplacian Eigenmaps

    - Transform $\mathbf{x} \in \mathbb{R}^d$ into $\mathbf{z} \in \mathbb{R}^k$, where $k \leq d$
    - Such that: neighbors in $\mathbb{R}^d$ stay neighbors in $\mathbb{R}^k$


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 23</div>
    """)
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
    mo.md(r"""
    ### Adjacency matrix

    - What makes points neighbors?
        - Similarity!
    - Construct graph $\mathcal{G}$ where each node $i$ represents a datapoint $\mathbf{x}_i$; an edge connects two nodes if the two correspondent points are close (e.g $||\mathbf{x}_i-\mathbf{x}_j||<\epsilon)$
    - The correspondent adjacency matrix is a weighted symmetric matrix $\mathbf{W}\in \mathbb{R}^{N\times N}$ with
      \[{W}_{ij}={W}_{ji}\geq 0
      \]


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 23</div>
    """)
    return


@app.cell
def _(mo):
    from scipy.sparse.csgraph import connected_components
    neighbor_control = mo.ui.slider(start=1, stop=14, step=1,
                                     value=5, label="Nearest neighbors",
                                     show_value=True)
    return connected_components, neighbor_control


@app.cell
def _(X_demo, connected_components, mo, neighbor_control, np, plt, y_demo):

    _d2 = ((X_demo[:, None, :] - X_demo[None, :, :])**2).sum(axis=2)
    np.fill_diagonal(_d2, np.inf)
    _neighbors = np.argsort(_d2, axis=1)[:, :neighbor_control.value]
    _W = np.zeros((len(X_demo), len(X_demo)), dtype=int)
    for _i in range(len(X_demo)):
        _W[_i, _neighbors[_i]] = 1
    _W = np.maximum(_W, _W.T)
    _components, _labels = connected_components(_W, directed=False)
    _fig, _ax = plt.subplots(figsize=(8, 3.9))
    _ax.scatter(X_demo[:, 0], X_demo[:, 1], c=y_demo, cmap="coolwarm", s=23, zorder=2)
    for _i, _j in zip(*np.triu_indices(len(X_demo), 1)):
        if _W[_i, _j]:
            _ax.plot(X_demo[[_i, _j], 0], X_demo[[_i, _j], 1],
                     color="gray", alpha=.24, lw=.7, zorder=1)
    _ax.set_aspect("equal")
    _ax.set_title(f"{_components} connected component(s)")
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### 
    Change the number of nearest neighbors.

    - Too few edges: disconnected graph.
    - Too many edges: unwanted shortcuts.

    The number of zero Laplacian eigenvalues equals the number of components.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 26</div>
    """), neighbor_control, mo.as_html(_fig)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Mapping $\mathbf{x}_i$ to $\mathbf{z}_i$

    - Need a way to optimize the position of points in $\mathbb{R}^k$.
    - Consider the following: $J(\mathbf{z}) = \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N W_{ij}||\mathbf{z}_i-\mathbf{z}_j||^2$


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Optimizing the position of points in $\mathbb{R}^k$

    - Reasonable to minimize $J(\mathbf{z})$ with respect to $\mathbf{z}$
    - Obvious solution not interesting.
        - What is the obvious solution?


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Optimizing the position of points in $\mathbb{R}^k$

    \[\min_{z\in \mathbb{R}^N}\mathbf{z}^T\mathbf{L}\mathbf{z}
    \]
    with
    \[\mathbf{z}^T\mathbf{D}\mathbf{z}=1
    \]
    and
    \[\mathbf{z}^T\mathbf{D}\mathbf{1}=0
    \]


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Laplacian matrix $\mathbf{L}$

    - Deep theory!
    - The Laplacian tells us the divergence of the gradients.
        - How much is the steepness changing.
    - Analogous to the second derivative for a continuous, single variate function.
    - [**Nice resource for further information**](https://mbernste.github.io/posts/laplacian_matrix/)


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 23</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Lagrange function for Laplacian Eigenmaps

    Lagrange function:
    $$
    L(\mathbf{z}, \lambda) = \mathbf{z}^T \mathbf{L} \mathbf{z} - \lambda (\mathbf{z}^T \mathbf{D} \mathbf{z}-1)
    $$

    Set derivative to zero:
    $$
    \frac{\partial L}{\partial \mathbf{z}} = 2\mathbf{L}\mathbf{z} - 2\lambda \mathbf{D}\mathbf{z} = 0
    $$

    Which gives:
    $$
    \mathbf{L}\mathbf{z} = \lambda \mathbf{D}\mathbf{z}
    $$

    - The minimization of this objective is given by eigenvector corresponding to the second smallest eigenvalue
    - Assuming a connected graph.


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 23</div>
    """)
    return


if __name__ == "__main__":
    app.run()
