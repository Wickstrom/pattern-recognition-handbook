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
# Marimo exercise notebook for the density estimation lecture.
# Programming exercises associated with notebooks/02/density_estimation.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/02/density_estimation_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/density_estimation_exercise.slides.json",
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
    # Density estimation — programming exercises
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
    ### Problem 2.36 from the book

    - The cell below generates data from a uniform distribution. Use the Parzen
      window method to estimate the probability density function. Consider the
      following settings:
        - Varying number of samples (32, 256, 5000).
        - Use Gaussian kernel for the Parzen window with smoothing parameters
          of 0.05 and 0.2.
        """
    )
    return


@app.cell
def _():
    from scipy.stats import uniform
    return (uniform,)


@app.cell
def _(mo, np, plt, uniform):
    # Problem 2.36 starter: uniform samples on [0, 2] with the true PDF
    # overlaid. Students vary the sample count and the kernel bandwidth.
    number_of_samples = 32
    uniform_data_p36 = np.random.uniform(low=0, high=2, size=number_of_samples)

    x_uniform_p36 = np.linspace(-1, 3, 100)
    p_uniform_p36 = uniform.pdf(x_uniform_p36, loc=0, scale=2)

    fig_uni, ax_uni = plt.subplots(figsize=(8, 4))
    ax_uni.plot(x_uniform_p36, p_uniform_p36, "k", linewidth=2, label="PDF")
    ax_uni.set_xlabel("Value")
    ax_uni.set_title("Uniform distribution")
    ax_uni.legend()
    ax_uni.grid(True)

    mo.as_html(fig_uni)
    plt.close(fig_uni)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Problem 2.12 from the book

    - Below is code that generates the data associated with problem 2.12
      from the book. Your task is to:
        - Design a Bayesian classifier.
        - Design a Bayesian classifier using the following risk parameters.
          How does this change the decision boundary?
            - $\lambda_{12}$: 1.0
            - $\lambda_{21}$: 0.5
        - Experiment with changing the mean and variance of each class.
          What do you observe?
    - However, this time you will estimate the probability density functions.
      Use both a parametric and non-parametric approach. How do they compare
      to Bayes classifier?
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Problem 2.12 starter: two well-overlapping Gaussians with the same
    # covariance. The Bayes-optimal boundary is the perpendicular
    # bisector of the line between the means, but density-estimation
    # classifiers won't recover that exactly — that's the point of the
    # exercise.
    number_of_samples_in_each_class = 100

    mu1 = np.array([1, 1])
    mu2 = np.array([1.5, 1.5])
    sigma = np.array([[0.2, 0.0], [0.0, 0.2]])

    x_train_1 = np.random.multivariate_normal(mu1, sigma, number_of_samples_in_each_class)
    x_train_2 = np.random.multivariate_normal(mu2, sigma, number_of_samples_in_each_class)
    x_train = np.concatenate((x_train_1, x_train_2))
    y_train = np.concatenate(
        (np.ones(number_of_samples_in_each_class), np.zeros(number_of_samples_in_each_class))
    )

    fig_p212, ax_p212 = plt.subplots(figsize=(8, 8))
    ax_p212.scatter(
        x_train_1[:, 0], x_train_1[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_p212.scatter(
        x_train_2[:, 0], x_train_2[:, 1],
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
    ### Classification using Bayes classifiers and density estimation

    You will now get your first experince working with real-world data. Below is a code block that loads the Iris dataset, one of the earliest known datasets used for evaluating classification methods and a classic introductory dataset in pattern recognition and machine learning. [See here for more information about the dataset](https://archive.ics.uci.edu/dataset/53/iris).

    The data is 4-dimensional with 3 classes. Here, we have simplifed the problem a bit and extracted 2 features such that the dataset can be easily visualized. Use the density estimation approach you deem the most suitable and combine it with a Bayesian classifier to tackle the problem of Iris plant classification.
        """
    )
    return


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo
    return (fetch_ucirepo,)


@app.cell
def _(fetch_ucirepo, mo, np, plt):
    # Iris starter: 2 of the 4 features (sepal length, sepal width),
    # all 3 classes. Students pick a density estimator and combine it
    # with a Bayes classifier to draw a decision boundary.
    iris = fetch_ucirepo(id=53)

    X_iris = iris.data.features.iloc[:, :2]
    feature_1_name = "sepal length (cm)"
    feature_2_name = "sepal width (cm)"
    y_iris = np.zeros(150)
    y_iris[50:100] = 1
    y_iris[100:150] = 2
    y_names = np.unique(iris.data.targets)

    fig_iris, ax_iris = plt.subplots(figsize=(6, 6))
    for class_i, class_name in enumerate(y_names):
        ax_iris.scatter(
            X_iris.iloc[np.where(y_iris == class_i)[0], 0],
            X_iris.iloc[np.where(y_iris == class_i)[0], 1],
            label=class_name,
        )
    ax_iris.set_xlabel(feature_1_name)
    ax_iris.set_ylabel(feature_2_name)
    ax_iris.legend()

    mo.as_html(fig_iris)
    plt.close(fig_iris)
    return


if __name__ == "__main__":
    app.run()
