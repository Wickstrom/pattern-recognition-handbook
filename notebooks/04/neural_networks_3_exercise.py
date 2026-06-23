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
# Marimo exercise notebook for the Neural Networks III lecture.
# Programming exercises associated with notebooks/04/neural_networks_3.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/04/neural_networks_3_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks_3_exercise.slides.json",
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
    # Neural Networks III — programming exercises
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
    ### Wine classification with neural networks.

    This problem is focused on multi-class classification of wine types based on 13 features. Below, we have simplified the problem by extracting 2 features. Take this as a starting point, but when you feel confident in your implementation, train your classifier on all 13 features. Does performance change?
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_wine
    return (load_wine,)


@app.cell
def _(load_wine, mo, np, plt):
    # Wine starter: 2 of the 13 features (alcohol, malic acid), all 3
    # classes. Targets are 0/1/2; legend is shifted to 1/2/3 to match
    # the original notebook's "Class N" labeling.
    wine_data = load_wine()

    X_wine = wine_data.data[:, :2]
    feature_1_name_wine = "alcohol"
    feature_2_name_wine = "malic acid"
    y_wine = wine_data.target
    y_names_wine = np.unique(y_wine)
    colors_wine = ["black", "blue", "red"]

    fig_wine, ax_wine = plt.subplots(figsize=(5, 5))
    for class_value_wine, color_wine in zip(y_names_wine, colors_wine):
        ax_wine.scatter(
            X_wine[y_wine == class_value_wine, 0],
            X_wine[y_wine == class_value_wine, 1],
            s=120, facecolors="none",
            edgecolors=color_wine, linewidth=3.0,
            label=f"Class {class_value_wine+1}",
        )
    ax_wine.set_xlabel(feature_1_name_wine)
    ax_wine.set_ylabel(feature_2_name_wine)
    ax_wine.legend()

    mo.as_html(fig_wine)
    plt.close(fig_wine)
    return


if __name__ == "__main__":
    app.run()
