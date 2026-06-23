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
# Marimo exercise notebook for the Neural Networks II lecture.
# Programming exercises associated with notebooks/04/neural_networks_2.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/04/neural_networks_2_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks_2_exercise.slides.json",
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
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Neural Networks II — programming exercises
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Non-linear classification I

    The code below loads a classic synthetic machine learning dataset, the Two Moons dataset. This is a binary classification problem that is not solvable with a linear classifier (try one of your classifiers from previous notebooks and see for yourself.) Design and train a MLP to solve the task. You can experiment with making the problem more or less challenging by changing the "noise" parameter in the "make_moons" function.

    This is the same problem as last week, but this time you will optimize with momentum included in the gradient descent algorithm.
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    return (make_moons,)


@app.cell
def _(make_moons, mo, plt):
    # Same two-moons starter as last week — students reuse their MLP
    # here, this time with momentum.
    X_moons, y_moons = make_moons(n_samples=400, noise=0.11, random_state=42)

    fig_moons, ax_moons = plt.subplots(figsize=(6, 6))
    ax_moons.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="viridis",
                     edgecolor="k", s=50)

    mo.as_html(fig_moons)
    plt.close(fig_moons)
    return


if __name__ == "__main__":
    app.run()
