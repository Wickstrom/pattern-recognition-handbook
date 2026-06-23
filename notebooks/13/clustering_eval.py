# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo version of the Clustering evaluation lecture.
# Same content as notebooks/13/clustering_eval.ipynb, but authored as a
# reactive Marimo app (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/13/clustering_eval.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `clusters1`) can be
# redefined in any number of cells, but Marimo requires each name to
# be owned by exactly one cell. Below, the two motivating figures use
# the `_motivation_a` and `_motivation_b` suffixes so they stay
# independent of any future cells that might want similar names.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/clustering_eval.slides.json",
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
    # Clustering 4 - Evaluating clustering
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - During the last couple of weeks we have learned about different methods for clustering data.
    - But how do we know if the clusters we obtain are good?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Motivation

    - Consider the following examples:
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Motivating example (well-separated clusters): same three Gaussian
    # clusters, plotted with two different colour assignments. The
    # accuracy of a clustering should be invariant to the colour
    # permutation.
    rng_mot_a = np.random.default_rng(42)
    n_per_mot_a = 60
    means_mot_a = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
    covs_mot_a = [0.4 * np.eye(2), 0.6 * np.eye(2), 0.35 * np.eye(2)]
    clusters_mot_a = [rng_mot_a.multivariate_normal(mean, cov, size=n_per_mot_a)
                      for mean, cov in zip(means_mot_a, covs_mot_a)]

    colors_mot_a = ["tab:blue", "tab:orange", "tab:green"]
    colors_cluster_mot_a = ["tab:orange", "tab:blue", "tab:green"]

    fig_mot_a, axes_mot_a = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
    ax_mot_a_left = axes_mot_a[0]
    for k_mot_a in range(3):
        ax_mot_a_left.scatter(clusters_mot_a[k_mot_a][:, 0], clusters_mot_a[k_mot_a][:, 1],
                              c=colors_mot_a[k_mot_a], s=40, edgecolor="k")
    ax_mot_a_left.set_aspect("equal", "box")
    ax_mot_a_left.grid(alpha=0.25)

    ax_mot_a_right = axes_mot_a[1]
    for k_mot_a in range(3):
        ax_mot_a_right.scatter(clusters_mot_a[k_mot_a][:, 0], clusters_mot_a[k_mot_a][:, 1],
                               c=colors_cluster_mot_a[k_mot_a], s=40, edgecolor="k")
    ax_mot_a_right.set_aspect("equal", "box")
    ax_mot_a_right.grid(alpha=0.25)

    fig_mot_a.tight_layout(rect=[0, 0.03, 1, 0.95])

    mo.as_html(fig_mot_a)
    plt.close(fig_mot_a)
    return


@app.cell
def _(mo, np, plt):
    # Same idea but with overlapping clusters: shows that the
    # colour-permutation invariance matters even more when the
    # clusters are noisy.
    rng_mot_b = np.random.default_rng(42)
    n_per_mot_b = 60

    means_mot_b = np.array([[1.5, 0.5], [4.0, -0.2], [2.7, 2.2]])
    covs_mot_b = [2.0 * np.eye(2), 1.7 * np.eye(2), 1.5 * np.eye(2)]
    clusters_mot_b = [rng_mot_b.multivariate_normal(mean, cov, size=n_per_mot_b)
                      for mean, cov in zip(means_mot_b, covs_mot_b)]

    colors_mot_b = ["tab:blue", "tab:orange", "tab:green"]
    colors_cluster_mot_b = ["tab:orange", "tab:blue", "tab:green"]

    fig_mot_b, axes_mot_b = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
    ax_mot_b_left = axes_mot_b[0]
    for k_mot_b in range(3):
        ax_mot_b_left.scatter(clusters_mot_b[k_mot_b][:, 0], clusters_mot_b[k_mot_b][:, 1],
                              c=colors_mot_b[k_mot_b], s=40, edgecolor="k", alpha=0.75)
    ax_mot_b_left.set_aspect("equal", "box")
    ax_mot_b_left.grid(alpha=0.25)

    ax_mot_b_right = axes_mot_b[1]
    for k_mot_b in range(3):
        ax_mot_b_right.scatter(clusters_mot_b[k_mot_b][:, 0], clusters_mot_b[k_mot_b][:, 1],
                               c=colors_cluster_mot_b[k_mot_b], s=40, edgecolor="k", alpha=0.75)
    ax_mot_b_right.set_aspect("equal", "box")
    ax_mot_b_right.grid(alpha=0.25)

    fig_mot_b.tight_layout(rect=[0, 0.03, 1, 0.95])

    mo.as_html(fig_mot_b)
    plt.close(fig_mot_b)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Clustering accuracy

    - Standard accuracy is computed as:
    - Index for a particular class is (usually) arbitrary!
        - No reason the cluster predictions should have the same value as the true labels.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### An assignment problem

    - A fundamental combinatorial optimization problem:
        - :
    - Standard approach in clustering: [**Hungarian algorithm**](https://en.wikipedia.org/wiki/Hungarian_algorithm)
    - [**Scipy implementation**](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
    - General procedure:
        - Solve assignment problem with Hungarian algorithm.
        - Compute accuracy as usual.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Normalized mutual information

    - An alternative quality measure is normalized mutual information: $$ NMI(Y, \hat{Y})=\frac{2 I(Y, \hat{Y})}{H(Y) + H(\hat{Y})}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Normalized mutual information

    - Understanding $I(Y, \hat{Y})$:
    - [**Scikit learn implementation**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Silhouette score

    - Both clustering accuracy and normalized mutual information assumes access to the true labels.
    - What if that is not available?
    - Several options, but the Silhouette score is widely used.
    - Measure how similar an object is to its own cluster compare to other clusters.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Silhouette score

    - Computation:
    - [**Scikit learn implementation**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
        """
    )
    return


if __name__ == "__main__":
    app.run()
