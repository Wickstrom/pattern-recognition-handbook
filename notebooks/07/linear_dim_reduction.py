# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo version of the Data transformation and dimensionality reduction
# I lecture (FDA + PCA).
# Same content as notebooks/07/linear_dim_reduction.ipynb, but authored
# as a reactive Marimo app (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/07/linear_dim_reduction.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/linear_dim_reduction.slides.json",
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
    # Data transformation and dimensionality reduction (DTDR) I
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - So far, the course has been problem oriented (mainly classification).
    - However, an essential part of pattern recognition is data analysis.
    - In this lecture, we will look at methods that compress data into a more compact representation thorugh linear transformations.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Analyzing data

    - Show example
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Problems

    - The data dimensionality may be very large.
        - Computationally demanding.
        - Curse of dimensionality
    - Some parts of the data may not be discriminative / be redundant.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remedies

    - May pick only parts of the data to use.
    - May transform the data!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Approaches to DTDR

    - To better discriminate between classes (supervised).
    - To remove redundancy (unsupervised).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Fisher discriminant analysis (FDA)

    - Transform (project) to 1D:
    - Start with the two class case and $P(w_1)=P(w_2)$
    - Fisher discriminant ratio:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Scatter matrices

    - **Within class:** $\boldsymbol{S}_w = \sum_{i=1}^M P(\boldsymbol{w}_i) \boldsymbol{\Sigma}_i$

    - where $\boldsymbol{\Sigma}_i = \mathbb{E}[(\boldsymbol{x} - \boldsymbol{\mu}_i)(\boldsymbol{x} - \boldsymbol{\mu}_i)^T]$

    - Example:

    ---

    - **Between class:** $\boldsymbol{S}_B = \sum_{i=1}^M P(\boldsymbol{w}_i) (\boldsymbol{\mu}_i - \boldsymbol{\mu})^2$

    - Example:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remark

    - Class separability measures in $\mathbf{x}$ by e.g. $\frac{\text{trace}(\boldsymbol{S}_w)}{\text{trace}(\boldsymbol{S}_B)}$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Fisher discriminant analysis

    - Remember; want to learn a transformation into 1D $z = \mathbf{w}^T \mathbf{x} \Rightarrow \mu = \mathbf{w}^T \boldsymbol{\mu}$

    - $S_B = (\mu_1 - \mu_2)^2 = $

    - $\sigma^2 = \mathbb{E}[(x - \mu)^2] = $

    - $S_w = \sigma_1^2 + \sigma_2^2 = $

    - Hence: Fisher discriminant ratio (FDR):
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks

    - Don't need $P(w_1) = P(w_2)$, but easy to solve.

    - If $P(w_1) \neq P(w_2)$: $\mathbf{w}$is the leading eigenvector of $\boldsymbol{S}_w^{-1} \boldsymbol{S}_B$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### FDR

    - Want: $\arg\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{S}_B \mathbf{w}}{\mathbf{w}^T \boldsymbol{S}_w \mathbf{w}}$

    - At solution (problem x.x): $\boldsymbol{S}_w \mathbf{w} = \lambda \boldsymbol{S}_B \mathbf{w}$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##

    I. If $P(w_1) = P(w_2)$:
    $$
    \lambda \boldsymbol{S}_w \mathbf{w} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \mathbf{w}
    $$
    $\implies$ $\mathbf{w} \propto \boldsymbol{S}_w^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$

    ---

    II. If $P(w_1) \neq P(w_2)$:
    $$
    \mathbf{w} = \text{leading eigenvector of } \boldsymbol{S}_w^{-1} \boldsymbol{S}_B
    $$

    ---

    **Example:**  
    $P(w_1) = P(w_2)$

    $$
    S_w = \frac{1}{2} \sigma_1^2 + \frac{1}{2} \sigma_2^2
    $$

    $$
    S_B = \frac{1}{2} (\mu_1 - \mu)^2 + \frac{1}{2} (\mu_2 - \mu)^2
    $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remark

    - Generalized to $z = \mathbf{w}^T \mathbf{x} \in \mathbb{R}^k$ where $k \leq d$.
        - More complex (pages 291-297 in book).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Principal Component Analysis (PCA)

    - First: $\mathbf{z} = \mathbf{A} \mathbf{x}$ such that $\mathbf{z} \in \mathbb{R}^d$, $\mathbf{x} \in \mathbb{R}^d$, and $\mathbf{A} \in \mathbb{R}^{d \times d}$
    - Want: $\boldsymbol{\Sigma}_y$ diagonal!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### A closer look at the covariance matrix

    - Have:

    $$
    \boldsymbol{\Sigma}_y = \mathbb{E}[(\mathbf{y} - \boldsymbol{\mu}_y)(\mathbf{y} - \boldsymbol{\mu}_y)^T]
    $$

    $$
    = \mathbb{E}[(\mathbf{A}\mathbf{x} - \mathbf{A}\boldsymbol{\mu}_x)(\mathbf{A}\mathbf{x} - \mathbf{A}\boldsymbol{\mu}_x)^T]
    $$

    ---

    - $\boldsymbol{\Sigma}_x$: symmetric and positive semi-definite $\implies $orthogonal eigenvectors and non-negative eigenvalues.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Eigendecomposition of the covariance matrix

    - Let $\mathbf{E} = [\mathbf{e}_1, \ldots, \mathbf{e}_d]$

    $$
    \boldsymbol{\Sigma}_x \mathbf{E} = \mathbf{E} \boldsymbol{\Lambda}
    $$

    where
    $$
    \boldsymbol{\Lambda} = \begin{bmatrix}
    \lambda_1 & 0 & \cdots & 0 \\
    0 & \lambda_2 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & \lambda_d
    \end{bmatrix}
    $$
    (diagonal matrix of eigenvalues)

    ---

    - Have:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interpreting the eigenvalues and eigenvectors

    - Note: $\boldsymbol{\Sigma}_y=$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Variance maximally preserved

    - First: $\sum_{i=1}^d \text{Var}(y_i) = \sum_{i=1}^d \lambda_i$

    - Thus: Let $\mathbf{A} = \mathbf{E} = [\mathbf{e}_1, \ldots, \mathbf{e}_d]$

    - Remark: Let $\mathbb{E}[\mathbf{x}] = 0$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## PCA is reconstruction / compression

    - Let $\mathbf{z} \in \mathbb{R}^d = [z(0), z(1), \ldots, z(d-1)]^T$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### PCA is reconstruction / compression

    - Have $\mathbf{x} = \mathbf{A} \mathbf{z}$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### PCA is reconstruction / compression

    - **MSE:** $\mathbb{E}\left[\|\mathbf{x} - \hat{\mathbf{x}}\|^2\right]$

    For $\mathbf{z} \in \mathbb{R}^k$: $\hat{\mathbf{x}} = \sum_{i=0}^{k-1} z(i) \mathbf {e}_i$

    If $y_i = 0$ for $i \geq k$:
    $$
    \mathbb{E}\left[\|\mathbf{x} - \hat{\mathbf{x}}\|^2\right] = \sum_{i=k}^{d-1} \lambda_i
    $$

    $\implies$ **MSE is minimized**.

    ---

    - **Compression:** Store/save $\mathbf{z} \in \mathbb{R}^k $instead of $\mathbf{x}$ (e.g. images).
    - **Reconstruct:** $\hat{\mathbf{x}}$ using $\mathbf{z}$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
