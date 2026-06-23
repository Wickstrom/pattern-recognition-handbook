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
# Marimo exercise notebook for the Non-linear SVM / Kernel Methods lecture.
# Programming exercises associated with notebooks/06/kernel_methods.py,
# split out into their own Marimo app so the lecture deck stays focused
# on concepts while the exercises can be worked on interactively.
# Run locally with `marimo edit notebooks/06/kernel_methods_exercise.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/kernel_methods_exercise.slides.json",
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
    # Non-linear SVM — programming exercises
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
    ### Non-linear classification with a SVM

    The code below loads a classic synthetic machine learning dataset, the Two Moons dataset, that we have looked at before. Traing a SVM with a non-linear kernel to tackle this non-linearly separable classification task.
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    return (make_moons,)


@app.cell
def _(make_moons, mo, plt):
    # Two-moons starter: students reuse the kernel SVM from the
    # previous notebook and compare against an RBF-kernel SVM.
    X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)

    fig_moons, ax_moons = plt.subplots(figsize=(6, 6))
    ax_moons.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="viridis",
                     edgecolor="k", s=50)

    mo.as_html(fig_moons)
    plt.close(fig_moons)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Wine classification with SVM

    This problem is revisting the Wine classification problem from the linear SVM notebook. Now, turn the SVM into a non-linear classifier through a Gaussian kernel. How does this affect performance?
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
