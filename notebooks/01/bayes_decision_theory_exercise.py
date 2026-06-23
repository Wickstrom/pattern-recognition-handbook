# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo exercise notebook for the Bayes decision theory lecture.
# Programming exercises associated with notebooks/01/bayes_decision_theory.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/01/bayes_decision_theory_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/bayes_decision_theory_exercise.slides.json",
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
    # Bayes decision theory — programming exercises
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Programming exercises

    Below are programming exercises associated with this lecture. These
    cell blocks are starting points that load the data and prepare the
    problem so that you can get going with the implementation. There are
    also theoretical exercises, but due to copyright we cannot share them
    here. They will be made available in a private repository connected
    to the course.

    For the content covered in this lecture, we assume that we have
    knowledge about the probability density function of the data.
    Therefore, these programming exercises will focus on toy data where
    we have control over the data distribution. In future exercises, we
    will see how we can move on to real-world data.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Problem 2.12 starter: two overlapping Gaussians with the same
    # covariance. The Bayes-optimal boundary is the perpendicular
    # bisector of the line between the means — students design a
    # Bayesian classifier (with and without risk parameters) and
    # visualise the resulting decision boundary.
    number_of_samples_in_each_class = 100

    mu1_toy = np.array([1.0, 1.0])
    mu2_toy = np.array([1.5, 1.5])
    sigma_toy = np.array([[0.2, 0.0], [0.0, 0.2]])

    x_train_1 = np.random.multivariate_normal(mu1_toy, sigma_toy, number_of_samples_in_each_class)
    x_train_2 = np.random.multivariate_normal(mu2_toy, sigma_toy, number_of_samples_in_each_class)
    x_train = np.concatenate((x_train_1, x_train_2))
    y_train = np.concatenate(
        (np.ones(number_of_samples_in_each_class), np.zeros(number_of_samples_in_each_class))
    )

    fig_toy, ax_toy = plt.subplots(figsize=(6, 6))
    ax_toy.scatter(x_train_1[:, 0], x_train_1[:, 1], s=120, facecolors="none",
                   edgecolors="black", linewidth=3.0, label="Class 1")
    ax_toy.scatter(x_train_2[:, 0], x_train_2[:, 1], s=120, facecolors="none",
                   edgecolors="blue", linewidth=3.0, label="Class 2")
    ax_toy.set_xlabel("x1")
    ax_toy.set_ylabel("x2")
    ax_toy.legend()
    fig_toy.tight_layout()

    mo.vstack(
        [
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
                """
            ),
            mo.as_html(fig_toy),
        ],
        gap=2,
    )
    plt.close(fig_toy)
    return


if __name__ == "__main__":
    app.run()
