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
# Marimo exercise notebook for the linear classifiers lecture.
# Programming exercises associated with notebooks/03/linear_classifiers.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/03/linear_classifiers_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/linear_classifiers_exercise.slides.json",
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

    # Seed once so the same samples back every figure in this notebook —
    # otherwise the slides re-roll data on each render and the figures
    # change between class sessions.
    np.random.seed(0)
    return np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Linear classifiers — programming exercises
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
    ### Problem 2.12 from the book

    Repeat problem 2.12 from the book, but using the MSE, SE2, and Perceptron classifier. How do they compare to the density-based classifiers from the previous sessions?

    The the Perceptron algorithm to convergence, the data needs to be linearly separable. The easiest way to accomplish this is to change the mean of one of the classes until the two clasess are completely separable.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Problem 2.12 starter: two overlapping Gaussians with the same
    # covariance. The Bayes-optimal boundary is the perpendicular
    # bisector of the line between the means, but MSE/LS won't recover
    # that exactly — that's the point of the exercise.
    number_of_samples_p212 = 100

    mu1_p212 = np.array([1, 1])
    mu2_p212 = np.array([1.5, 1.5])
    sigma_p212 = np.array([[0.2, 0.0], [0.0, 0.2]])

    x_train_1_p212 = np.random.multivariate_normal(mu1_p212, sigma_p212, number_of_samples_p212)
    x_train_2_p212 = np.random.multivariate_normal(mu2_p212, sigma_p212, number_of_samples_p212)

    fig_p212, ax_p212 = plt.subplots(figsize=(8, 8))
    ax_p212.scatter(
        x_train_1_p212[:, 0], x_train_1_p212[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_p212.scatter(
        x_train_2_p212[:, 0], x_train_2_p212[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2",
    )
    ax_p212.set_xlabel("x1")
    ax_p212.set_ylabel("x2")
    ax_p212.legend()

    mo.as_html(fig_p212)
    plt.close(fig_p212)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Wine classification with LS classifiers

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
    for class_value, color in zip(y_names_wine, colors_wine):
        ax_wine.scatter(
            X_wine[y_wine == class_value, 0],
            X_wine[y_wine == class_value, 1],
            s=120, facecolors="none",
            edgecolors=color, linewidth=3.0,
            label=f"Class {class_value+1}",
        )
    ax_wine.set_xlabel(feature_1_name_wine)
    ax_wine.set_ylabel(feature_2_name_wine)
    ax_wine.legend()

    mo.as_html(fig_wine)
    plt.close(fig_wine)
    return


if __name__ == "__main__":
    app.run()
