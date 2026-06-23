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
# Marimo exercise notebook for the support vector machine lecture.
# Programming exercises associated with notebooks/03/support_vector_machine.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/03/support_vector_machine_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/support_vector_machine_exercise.slides.json",
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
    # Support Vector Machine — programming exercises
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Programming exercises

    Below are programming exercises assocaited with this lecture. These cell blocks are starting points that loads the data and prepares the problem such that you can get going with the implementation. There are also theoretical exercsies, but due to copyright we cannot shared them here. They will be made available in a private repository connected to the course.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Wine classification with SVM

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
    # Wine exercise starter: same 2-feature projection as the lecture
    # example. Students extend this by fitting an SVM on all 13
    # features and comparing multi-class performance.
    wine_data_ex = load_wine()

    X_wine_ex = wine_data_ex.data[:, :2]
    feature_1_name_wine_ex = "alcohol"
    feature_2_name_wine_ex = "malic acid"
    y_wine_ex = wine_data_ex.target
    y_names_wine_ex = np.unique(y_wine_ex)
    colors_wine_ex = ["black", "blue", "red"]

    fig_wine_ex, ax_wine_ex = plt.subplots(figsize=(5, 5))
    for class_value_wine_ex, color_wine_ex in zip(y_names_wine_ex, colors_wine_ex):
        ax_wine_ex.scatter(
            X_wine_ex[y_wine_ex == class_value_wine_ex, 0],
            X_wine_ex[y_wine_ex == class_value_wine_ex, 1],
            s=120, facecolors="none",
            edgecolors=color_wine_ex, linewidth=3.0,
            label=f"Class {class_value_wine_ex+1}",
        )
    ax_wine_ex.set_xlabel(feature_1_name_wine_ex)
    ax_wine_ex.set_ylabel(feature_2_name_wine_ex)
    ax_wine_ex.legend()

    mo.as_html(fig_wine_ex)
    plt.close(fig_wine_ex)
    return


if __name__ == "__main__":
    app.run()
