# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo version of the RBF Networks and Prototypical Learning lecture.
# Same content as notebooks/05/rbf_networks.ipynb, but authored as a
# reactive Marimo app so each matplotlib figure stays tied to the data
# generation that produced it (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/05/rbf_networks.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `X`) can be redefined
# in any number of cells, but Marimo requires each name to be owned by
# exactly one cell. Below, the cell-local dataset and figure use the
# `_xor` suffix so they stay independent of any future cells that
# might want similar names.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/rbf_networks.slides.json",
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
    # RBF Networks and Prototypical Learning
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - We have learned how neural networks transform data into new representation to solve non-linear problems.
    - We will now look at non-linear classification from the perspective of function approximation.
    - Key idea:
        - Transform data in terms of a preselected class of interpolation functions.
    - We will also see how RBF networks relate to the field of prototypical learning.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Function approximation

    - Recall the XOR example:
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # XOR dataset: four small Gaussian blobs at the corners of the
    # unit square with labels {0, 1, 1, 0}. Used here as a small,
    # already-familiar running example to motivate the function-
    # approximation framing of RBF networks.
    np.random.seed(42)

    n_samples_xor = 50
    std_xor = 0.15

    centers_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels_xor = np.array([0, 1, 1, 0])

    X_xor = []
    y_xor = []
    for center_xor, label_xor in zip(centers_xor, labels_xor):
        X_xor.append(center_xor + std_xor * np.random.randn(n_samples_xor, 2))
        y_xor.append(np.full(n_samples_xor, label_xor))
    X_xor = np.vstack(X_xor)
    y_xor = np.concatenate(y_xor)

    fig_xor, (ax_xor_1, ax_xor_2) = plt.subplots(1, 2, figsize=(8, 4))
    ax_xor_1.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], color="blue", label="Class 0")
    ax_xor_1.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], color="red", label="Class 1")
    ax_xor_1.set_xlabel("x1")
    ax_xor_1.set_ylabel("x2")
    ax_xor_2.set_ylim(-0.5, 1.5)
    ax_xor_2.set_xlim(-0.5, 1.5)
    ax_xor_2.set_xlabel("x1")
    ax_xor_2.set_ylabel("x2")

    mo.as_html(fig_xor)
    plt.close(fig_xor)
    return X_xor, y_xor


@app.cell
def _(mo):
    mo.md(
        r"""
    ### General idea

    - Let (non-linear) $f_i: \mathbb{R}^d \rightarrow \mathbb{R}, i=1,\cdots,k$
    - Look at $\mathbf{x} \in \mathbb{R}^d$. (Usually $k>l$)
    - Draw example:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Network

    - The function approximation setup described above can be thought of as a network:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Radial basis function (RBF) networks

    - General idea: $ f(\|\mathbf{x} - \mathbf{c}_i\|)$
    - RBF network: $ f(\mathbf{x})= \exp(-\frac{1}{2\sigma_i^2}(\|\mathbf{x} - \mathbf{c}_i\|))$
    - Activation at node $f_i$ given by distance to $\mathbf{c}_i$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Estimating the weights

    - Any linear method will do!
    - Let $\mathbf{y}$ be the desired outputs. Construct matrix $\mathbf{G}$:
    - $ G = \begin{bmatrix} \phi(x_1, c_1) & \cdots & \phi(x_1, c_k) \\ \vdots & & \vdots \\ \phi(x_N, c_1) & \cdots & \phi(x_N, c_k) \end{bmatrix}_{N \times k} $
    - So:
    - $ \mathbf{g} = G \mathbf{w} = \begin{bmatrix} g(x_1) \\ \vdots \\ g(x_N) \end{bmatrix} $
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Estimating the weights

    - Choose $\mathbf{w}$ such that $J = \|\mathbf{y} - G\mathbf{w}\|^2$ is minimized.
    - $ \frac{\partial J}{\partial \mathbf{w}} = 0 \implies \boxed{\mathbf{w} = (G^\top G)^{-1} G^\top \mathbf{y}} $
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How to choose the centers?

    - **Fixed centres:** Centres selected randomly from the training set.
       - Simple, but not optimum!
    - **Training of the centres:**  Let $\sigma_i^2$, $\mathbf{c}_i$, $w_i$, $i=1,\ldots,k$, be free parameters learned from the training set.

       $$ \frac{\partial J}{\partial \mathbf{c}_i} = 0 \implies \mathbf{c}_i^{(\text{new})} = \mathbf{c}_i^{(\text{old})} + \mu \frac{\partial J}{\partial \mathbf{c}_i} $$
    - Choose centres according to how the data are distributed in space. Draw ->
       - More about this later.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Prototypical learning

    - Consider the following example
        """
    )
    return


@app.cell
def _(mo):
    # Uncertainty figure: motivates why prototypes / typical examples
    # help when a single decision boundary isn't enough to communicate
    # what the model has learned.
    mo.image(src="media/uncertaintyex.jpg", width="400px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Prototypical learning

    - Define / learn prototypes and classifiy new samples based on similarity.
    - Prototypes can be defined in input space or in latent space!
        """
    )
    return


@app.cell
def _(mo):
    # ProtoVAE figure: prototypes learned in the latent space of a
    # variational autoencoder.
    mo.image(src="media/protovae.png", width="700px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Prototypical learning

    - Using prototypes can make it easier to understand the behaviour of your model.
    - Prototypes represent "typical" examples in the data.
    - Example from Kingma and Welling, 2016.
        """
    )
    return


@app.cell
def _(mo):
    # VAE figure: original Kingma & Welling VAE diagram that the
    # ProtoVAE builds on.
    mo.image(src="media/vae.png", width="700px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Prototypical learning

    - Using prototypes can make it easier to understand the behaviour of your model.
    - Prototypes represent "typical" examples in the data.
        """
    )
    return


@app.cell
def _(mo):
    # Archetype figure: an alternative way of summarising a class with
    # a single "typical" example rather than a centroid.
    mo.image(src="media/archetype.png", width="400px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Challenges with prototypical learning

    - Curse of dimensionality.
        - Need many prototypes in high dimensional space. Can be computationally expensive.
    - Optimization can be slow, and can be memory intensive.
        """
    )
    return


if __name__ == "__main__":
    app.run()
