# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "ucimlrepo",
# ]
# ///
#
# Marimo exercise notebook for the Clustering evaluation lecture.
# Programming exercises associated with notebooks/13/clustering_eval.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/13/clustering_eval_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/clustering_eval_exercise.slides.json",
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
    # Clustering evaluation — programming exercises
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
    ### Cluster the Iris dataset and evaluate the clustering quality

    Cluster the Iris data below with your clustering algorithm of choice. Evaluate the resulting clusters with your own implementation of clustering accuracy, normalized mutual information, and the silhouette score. You can use the Scipy function for solving the assignment problem in the clustering accuracy computation. Compare your implementation with the output of existing implementations.

    Cluster the Iris data with different clustering methods and compare the performance. Which method gives you the best clustering performance?
        """
    )
    return


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo
    return (fetch_ucirepo,)


@app.cell
def _(fetch_ucirepo, mo, np, plt):
    # Iris starter: first two features, 3 classes. Students run any
    # clustering algorithm and report accuracy / NMI / silhouette.
    iris = fetch_ucirepo(id=53)

    X_iris = iris.data.features.iloc[:, :2]
    X_1_name_iris = "sepal length (cm)"
    X_2_name_iris = "sepal width (cm)"
    y_iris = np.zeros(150)
    y_iris[50:100] = 1
    y_iris[100:150] = 2

    fig_iris, ax_iris = plt.subplots(figsize=(6, 6))
    ax_iris.scatter(X_iris.iloc[:, 0], X_iris.iloc[:, 1], c=y_iris,
                    cmap="viridis", edgecolor="k", s=50)
    ax_iris.set_xlabel(X_1_name_iris)
    ax_iris.set_ylabel(X_2_name_iris)

    mo.as_html(fig_iris)
    plt.close(fig_iris)
    return


if __name__ == "__main__":
    app.run()
