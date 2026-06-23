# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "ucimlrepo",
# ]
# ///
#
# Marimo exercise notebook for the Sequential clustering lecture.
# Programming exercises associated with notebooks/10/seq_clustering.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/10/seq_clustering_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/seq_clustering_exercise.slides.json",
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
    # Sequential clustering — programming exercises
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
    ### BSAS on simple 2-dimensional data

    This first exercise is focused on simple 2-dimensional data such that you can familiarize yourself with the BSAS algorithm. Cluster the data stored in X using different thresholds and order, and observe how it affects the clustering.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # BSAS starter: 14 hand-picked 2-D points, a default threshold
    # of sqrt(2), and three sample traversal orders students can
    # experiment with.
    X_p1 = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5],
                     [5, 5], [5, 6], [-4, 5], [-3, 5], [-4, 4], [-3, 4]])

    threshold_p1 = np.sqrt(2)

    order_a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    order_b = [0, 9, 1, 2, 3, 10, 11, 4, 5, 6, 12, 7, 13, 8]
    order_c = [0, 9, 4, 1, 2, 10, 11, 3, 5, 6, 12, 13, 7, 8]

    fig_p1, ax_p1 = plt.subplots(figsize=(6, 5))
    ax_p1.scatter(X_p1[:, 0], X_p1[:, 1], s=80, edgecolor="k")
    ax_p1.set_title("Hand-picked 2-D points for BSAS")
    ax_p1.set_aspect("equal")
    ax_p1.grid(alpha=0.3)

    mo.as_html(fig_p1)
    plt.close(fig_p1)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### BSAS on the Iris dataset.

    Use the BSAS algorithm to cluster the Iris dataset. Experiment with order and threshold to see how you can fit the known number of classes (3) best.
        """
    )
    return


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo
    return (fetch_ucirepo,)


@app.cell
def _(fetch_ucirepo, mo, np, plt):
    # Iris starter: first two features, all 3 classes. Students apply
    # their BSAS implementation to recover the natural 3 clusters.
    iris = fetch_ucirepo(id=53)

    X_iris = iris.data.features.iloc[:, :2]
    feature_1_name_iris = "sepal length (cm)"
    feature_2_name_iris = "sepal width (cm)"
    y_iris = np.zeros(150)
    y_iris[50:100] = 1
    y_iris[100:150] = 2
    y_names_iris = np.unique(iris.data.targets)

    fig_iris, ax_iris = plt.subplots(figsize=(6, 6))
    for class_i, class_name in enumerate(y_names_iris):
        ax_iris.scatter(
            X_iris.iloc[np.where(y_iris == class_i)[0], 0],
            X_iris.iloc[np.where(y_iris == class_i)[0], 1],
            label=class_name,
        )
    ax_iris.set_xlabel(feature_1_name_iris)
    ax_iris.set_ylabel(feature_2_name_iris)
    ax_iris.legend()

    mo.as_html(fig_iris)
    plt.close(fig_iris)
    return


if __name__ == "__main__":
    app.run()
