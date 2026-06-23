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
# Marimo exercise notebook for the Function-optimization clustering
# lecture. Programming exercises associated with notebooks/12/clustering.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/12/clustering_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/clustering_exercise.slides.json",
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
    # Function-optimization clustering — programming exercises
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
    ### Making the Two-Moons dataset linearly separable

    The code below loads a classic synthetic machine learning dataset, the Two Moons dataset, that we have looked at before. Cluster the dataset using both K-means clustering and Spectral clustering and discuss the results.
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    return (make_moons,)


@app.cell
def _(make_moons, mo, plt):
    # Two-moons starter: students run both K-means and spectral
    # clustering on the same data and compare the boundaries.
    X_moons_ex, y_moons_ex = make_moons(n_samples=200, noise=0.05, random_state=42)

    fig_moons_ex, ax_moons_ex = plt.subplots(figsize=(6, 6))
    ax_moons_ex.scatter(X_moons_ex[:, 0], X_moons_ex[:, 1], c=y_moons_ex,
                        cmap="viridis", edgecolor="k", s=50)

    mo.as_html(fig_moons_ex)
    plt.close(fig_moons_ex)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Clustering toy data with K-means

    Perform K-means clustering on the simple toy dataset provided below. Plot the cluster centroids at different iterations. Experiment with different initializations of centroids. Discuss what you observe.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Toy starter for K-means: three Gaussian clusters with anisotropic
    # covariances. The original notebook referenced an undefined `N`;
    # we default to 150 (50 per cluster) so the cell actually runs.
    np.random.seed(0)
    N_toy = 150

    means_true_toy = np.array([[2, 3], [6, 5], [4, 1]])
    covs_true_toy = [
        np.array([[1.5, 0.5], [0.5, 1]]),
        np.array([[1.0, -0.3], [-0.3, 1.2]]),
        np.array([[0.8, 0.2], [0.2, 0.5]]),
    ]
    data_toy = np.vstack([
        np.random.multivariate_normal(means_true_toy[i], covs_true_toy[i], N_toy // 3)
        for i in range(3)
    ])

    fig_toy, ax_toy = plt.subplots(figsize=(6, 6))
    ax_toy.scatter(data_toy[:, 0], data_toy[:, 1], c="gray")

    mo.as_html(fig_toy)
    plt.close(fig_toy)
    return


if __name__ == "__main__":
    app.run()
