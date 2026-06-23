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
# Marimo exercise notebook for the Hierarchical clustering lecture.
# Programming exercises associated with notebooks/11/hier_clustering.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/11/hier_clustering_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/hier_clustering_exercise.slides.json",
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
    # Hierarchical clustering — programming exercises
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Programming exercises

    Below are programming exercises associated with this lecture. These cell blocks are starting points that load the data and prepare the problem such that you can get going with the implementation. There are also theoretical exercises, but due to copyright we cannot share them here. They will be made available in a private repository connected to the course.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Agglomorative clustering

    Consider the dissimilarity matrix in the cell below. Perform (manually) agglomerative clustering using the GAS algorithm, and comment on the result.

    Tip: you are free to use existing code to make nice dendrograms, for example with: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Hand-built 5x5 symmetric dissimilarity matrix: students run
    # the GAS algorithm by hand and (optionally) render the dendrogram
    # with sklearn's plot_agglomerative_dendrogram helper.
    P_hc = np.array([
        [0.0, 1.8, 2.4, 2.3, 4.5],
        [1.8, 0.0, 2.5, 2.7, 4.0],
        [2.4, 2.5, 0.0, 1.2, 3.5],
        [2.3, 2.7, 1.2, 0.0, 3.0],
        [4.5, 4.0, 3.5, 3.0, 0.0],
    ])

    fig_hc, ax_hc = plt.subplots(figsize=(5, 4))
    im_hc = ax_hc.imshow(P_hc, cmap="viridis")
    ax_hc.set_title("Dissimilarity matrix P")
    fig_hc.colorbar(im_hc, ax=ax_hc)

    mo.as_html(fig_hc)
    plt.close(fig_hc)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Visualizing the Wine classification dataset

    Perform dimensionality reduction on the Wine classification dataset using PCA and Kernel PCA. Visualize the results in 2D plots, coloring the points based on their class labels to observe how well the classes are separated in the reduced dimensions.

    Hint: normalizing X can make hyperparameter tuning easier here.
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_wine
    return (load_wine,)


@app.cell
def _(load_wine, mo, np, plt):
    # Wine starter: load all 13 features (no slicing) so students can
    # exercise PCA / Kernel PCA on the full feature set.
    wine_data = load_wine()

    X_wine = wine_data.data
    y_wine = wine_data.target
    y_names_wine = np.unique(y_wine)
    colors_wine = ["black", "blue", "red"]

    fig_wine, ax_wine = plt.subplots(figsize=(6, 6))
    for class_value_wine, color_wine in zip(y_names_wine, colors_wine):
        ax_wine.scatter(
            X_wine[y_wine == class_value_wine, 0],
            X_wine[y_wine == class_value_wine, 1],
            label=f"Class {class_value_wine+1}",
            color=color_wine,
            edgecolor="k", s=50,
        )
    ax_wine.set_xlabel("Feature 1")
    ax_wine.set_ylabel("Feature 2")
    ax_wine.legend()

    mo.as_html(fig_wine)
    plt.close(fig_wine)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Unrolling the Swiss roll.

    Transform the Swiss roll data into a 2-dimensional representation using Laplacian Eigenmaps
        """
    )
    return


@app.cell
def _(mo, plt):
    # Swiss roll starter (5000 points): students unroll it to 2-D
    # with Laplacian Eigenmaps.
    from sklearn.datasets import make_swiss_roll

    X_swiss_ex, t_swiss_ex = make_swiss_roll(n_samples=5000, noise=0.05, random_state=0)

    fig_swiss_ex = plt.figure(figsize=(8, 6))
    ax_swiss_ex = fig_swiss_ex.add_subplot(projection="3d")
    ax_swiss_ex.scatter(X_swiss_ex[:, 0], X_swiss_ex[:, 1], X_swiss_ex[:, 2],
                        c=t_swiss_ex, marker="o")

    mo.as_html(fig_swiss_ex)
    plt.close(fig_swiss_ex)
    return


if __name__ == "__main__":
    app.run()
