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
# Marimo exercise notebook for the Neural Networks I lecture.
# Programming exercises associated with notebooks/04/neural_networks_1.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/04/neural_networks_1_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks_1_exercise.slides.json",
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
    # Neural Networks I — programming exercises
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
    ## Solving the XOR problem with a 2LP

    In this problem, you will solve the XOR problem using a 2LP. Experiment with the number of neurons in the hidden layer and the learning rate in your gradient descent algorithm.
        """
    )
    return


@app.cell
def _():
    import random
    return (random,)


@app.cell
def _(mo, np, plt, random):
    # XOR exercise starter: four Gaussian blobs at the unit-square
    # corners, augmented with a bias column, then shuffled. Plot shows
    # the original (unshuffled) blobs so students can see the class
    # assignment before training.
    N_tr = 100

    mu1_xor_ex = np.array([0.0, 0.0])
    mu2_xor_ex = np.array([1.0, 1.0])
    mu3_xor_ex = np.array([0.0, 1.0])
    mu4_xor_ex = np.array([1.0, 0.0])
    sigma_xor_ex = np.array([[0.01, 0.0], [0.0, 0.01]])

    x1_tr_xor_ex = np.random.multivariate_normal(mu1_xor_ex, sigma_xor_ex, N_tr)
    x2_tr_xor_ex = np.random.multivariate_normal(mu2_xor_ex, sigma_xor_ex, N_tr)
    x3_tr_xor_ex = np.random.multivariate_normal(mu3_xor_ex, sigma_xor_ex, N_tr)
    x4_tr_xor_ex = np.random.multivariate_normal(mu4_xor_ex, sigma_xor_ex, N_tr)

    x_xor_ex = np.concatenate((x1_tr_xor_ex, x2_tr_xor_ex, x3_tr_xor_ex, x4_tr_xor_ex))
    y_xor_ex = np.concatenate((np.zeros(N_tr), np.zeros(N_tr),
                                np.ones(N_tr), np.ones(N_tr)))

    shuffle_idx_xor_ex = random.sample(range(0, 4 * N_tr), 4 * N_tr)

    # X = 400 x 3 (last column is the bias)
    x_xor_ex = np.c_[x_xor_ex[shuffle_idx_xor_ex], np.ones(x_xor_ex.shape[0])]
    y_xor_ex = y_xor_ex[shuffle_idx_xor_ex]

    fig_xor_ex, ax_xor_ex = plt.subplots(figsize=(8, 8))
    ax_xor_ex.scatter(x1_tr_xor_ex[:, 0], x1_tr_xor_ex[:, 1], s=120, facecolors="none",
                      edgecolors="blue", linewidth=3.0, label="Class 1")
    ax_xor_ex.scatter(x2_tr_xor_ex[:, 0], x2_tr_xor_ex[:, 1], s=120, facecolors="none",
                      edgecolors="blue", linewidth=3.0)
    ax_xor_ex.scatter(x3_tr_xor_ex[:, 0], x3_tr_xor_ex[:, 1], s=120, facecolors="none",
                      edgecolors="black", linewidth=3.0, label="Class 2")
    ax_xor_ex.scatter(x4_tr_xor_ex[:, 0], x4_tr_xor_ex[:, 1], s=120, facecolors="none",
                      edgecolors="black", linewidth=3.0)
    ax_xor_ex.legend()

    mo.as_html(fig_xor_ex)
    plt.close(fig_xor_ex)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Non-linear classification

    The code below loads a classic synthetic machine learning dataset, the Two Moons dataset. This is a binary classification problem that is not solvable with a linear classifier (try one of your classifiers from previous notebooks and see for yourself.) Design and train a MLP to solve the task. You can experiment with making the problem more or less challenging by changing the "noise" parameter in the "make_moons" function.
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    return (make_moons,)


@app.cell
def _(make_moons, mo, plt):
    # Two-moons starter: same noise level as last week's exercise so
    # students can compare a perceptron-based approach against the MLP
    # they implement here.
    X_moons, y_moons = make_moons(n_samples=400, noise=0.11, random_state=42)

    fig_moons, ax_moons = plt.subplots(figsize=(6, 6))
    ax_moons.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="viridis",
                     edgecolor="k", s=50)

    mo.as_html(fig_moons)
    plt.close(fig_moons)
    return


if __name__ == "__main__":
    app.run()
