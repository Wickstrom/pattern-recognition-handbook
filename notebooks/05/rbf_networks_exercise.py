# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo exercise notebook for the RBF Networks and Prototypical Learning
# lecture. Programming exercises associated with
# notebooks/05/rbf_networks.py, split out into their own Marimo app so
# the lecture deck stays focused on concepts while the exercises can be
# worked on interactively.
# Run locally with `marimo edit notebooks/05/rbf_networks_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/rbf_networks_exercise.slides.json",
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
    # RBF Networks — programming exercises
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
    ## Problem 4.20 from book

    Use an RBF network to approximate the function plotted below. Select the centers from a regular grid between 0 and 1. Repeat the experiments with different numbers of Gaussian functions and bandwidths for the Gaussian function. Estimate the unknown weights using the least squares method.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Problem 4.20 starter: 200 noisy samples from f(x) = 0.3 + 0.2·cos(2πx)
    # on [0, 1] with 10 candidate centers on a regular grid shown in red.
    # Students fit an RBF network by solving the least-squares system
    # built from the Gram matrix.
    N_tr = 200
    c_p420 = np.linspace(0, 1, 10).reshape(10, 1)

    def cos_func(x): return 0.3 + 0.2 * np.cos(2 * np.pi * x)

    x_tr_p420 = np.sort(np.random.uniform(size=N_tr))
    y_tr_p420 = cos_func(x_tr_p420)

    fig_p420, ax_p420 = plt.subplots(figsize=(8, 4))
    ax_p420.scatter(x_tr_p420, y_tr_p420)
    ax_p420.scatter(c_p420, np.zeros(10), color="red")

    mo.as_html(fig_p420)
    plt.close(fig_p420)
    return


if __name__ == "__main__":
    app.run()
