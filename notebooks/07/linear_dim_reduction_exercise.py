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
# Marimo exercise notebook for the Data transformation and dimensionality
# reduction I lecture. Programming exercises associated with
# notebooks/07/linear_dim_reduction.py, split out into their own Marimo
# app so the lecture deck stays focused on concepts while the exercises
# can be worked on interactively.
# Run locally with `marimo edit notebooks/07/linear_dim_reduction_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/linear_dim_reduction_exercise.slides.json",
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
    # Linear dimensionality reduction — programming exercises
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
    ### Dimensionality reduction on the Iris dataset

    We will now revisit the Iris dataset from the first lectures. Instead of simply selecting 2 features, you will now perform either FDA or PCA to reduce the dimensionality down to 2 and visualize the data. Do you observe any difference between the two methods?
        """
    )
    return


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo
    return (fetch_ucirepo,)


@app.cell
def _(fetch_ucirepo, mo, np, plt):
    # Iris starter: load all 4 features and the 3-class targets so
    # students can run either FDA or PCA on the full feature set and
    # project down to 2-D for visualisation.
    iris = fetch_ucirepo(id=53)

    X_iris = iris.data.features
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
