# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo version of the Sequential clustering lecture.
# Same content as notebooks/10/seq_clustering.ipynb, but authored as a
# reactive Marimo app (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/10/seq_clustering.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `X`) can be redefined
# in any number of cells, but Marimo requires each name to be owned by
# exactly one cell. Below, every cell-local dataset and figure is given
# a unique suffix (`_three`, `_bridge`, `_norms`) so the slides stay
# independent and the data each figure displays is reproducible.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/seq_clustering.slides.json",
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
    # Clustering 1 - Sequential clustering
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Often important to find groups in data.
    - Vastly important in data processing.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example

    - Clustering in modern computer vision
    - [**DINOv2: Learning Robust Visual Features without Supervision**](https://arxiv.org/pdf/2304.07193)
        """
    )
    return


@app.cell
def _(mo):
    # DINOv2 cluster figure: motivates why clustering matters in
    # modern computer-vision pipelines (self-supervised features
    # cluster semantically meaningful regions).
    mo.image(src="media/dino_cluster.png", width="600px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Clustering: Science or Art?

    - [**Is there such a thing as a "correct" clustering?**](https://proceedings.mlr.press/v27/luxburg12a/luxburg12a.pdf)
        """
    )
    return


@app.cell
def _(mo):
    # Three images from Luxburg's "Is there a correct clustering?"
    # paper that frame the discussion: thesis statement, what
    # different algorithms produce, and how to read them.
    mo.hstack(
        [
            mo.image(src="media/thesis.png", width="200px"),
            mo.image(src="media/cluster_results.png", width="400px"),
            mo.image(src="media/explanation.png", width="400px"),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Key points

    - Often no prior knowledge about data -> unsupervised
    - Different ways of specifying groups / clusters -> different answers.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Loose idea of clustering

    - Find natural groups such that:
        - Points in the same cluster "similar".
        - Points in different clusters are "dissimilar".
    - Draw example:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Different types of clustering

    - Will look at:
        - Sequential
        - Hierarchical
        - Function optimization
            - K-means
            - Spectral
        - Clustering evaluation
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Membership functions

    - Let $$X = \{\mathbf{x_1}, \cdots, \mathbf{x_N}\}$$
    - Task: Assign cluster memberships to $\mathbf{x}_i$ for $i=1,\cdots,N$.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Toy three-cluster dataset (4 points each) used to motivate
    # hard membership functions; the same setup is reused in later
    # slides with bridge points added.
    rng_three = np.random.default_rng(42)
    centers_three = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
    points_per_cluster_three = 4

    X_three = []
    labels_three = []
    for k_three, c_three in enumerate(centers_three):
        pts_three = c_three + 0.3 * rng_three.normal(size=(points_per_cluster_three, 2))
        X_three.append(pts_three)
        labels_three += [k_three] * points_per_cluster_three
    X_three = np.vstack(X_three)
    labels_three = np.array(labels_three)

    fig_three, ax_three = plt.subplots(figsize=(6, 5))
    for k_three in range(3):
        ix_three = labels_three == k_three
        ax_three.scatter(X_three[ix_three, 0], X_three[ix_three, 1], s=100,
                         label=f"cluster {k_three+1}", edgecolor="k")
    ax_three.set_title("Three distinct clusters (4 points each)")
    ax_three.legend()
    ax_three.set_aspect("equal")
    ax_three.grid(alpha=0.3)

    mo.as_html(fig_three)
    plt.close(fig_three)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Membership functions

    - Assume $m$ clusters.
    - Let $u_j: \mathbf{x} \rightarrow \{0, 1\}$.
    - Subject to: $\sum_{j=1}^{^m} u_j (\mathbf{x}_i)=1$ for $i=1,\cdots,N$
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Same toy three-cluster dataset but with two "bridge" points
    # added between the clusters — used to motivate the need for
    # fuzzy / continuous memberships.
    rng_bridge = np.random.default_rng(42)
    centers_bridge = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
    points_per_cluster_bridge = 4

    X_bridge = []
    labels_bridge = []
    for k_bridge, c_bridge in enumerate(centers_bridge):
        pts_bridge = c_bridge + 0.3 * rng_bridge.normal(size=(points_per_cluster_bridge, 2))
        X_bridge.append(pts_bridge)
        labels_bridge += [k_bridge] * points_per_cluster_bridge

    bridge_pts = np.array([[2.5, 0.0], [2.5, 2.0]])
    X_bridge.append(bridge_pts)
    labels_bridge += [3, 3]

    X_bridge = np.vstack(X_bridge)
    labels_bridge = np.array(labels_bridge)

    fig_bridge, ax_bridge = plt.subplots(figsize=(6, 5))
    for k_bridge in np.unique(labels_bridge):
        ix_bridge = labels_bridge == k_bridge
        lbl = f"cluster {k_bridge+1}" if k_bridge < 3 else "bridge"
        ax_bridge.scatter(X_bridge[ix_bridge, 0], X_bridge[ix_bridge, 1], s=100,
                          label=lbl, edgecolor="k")
    ax_bridge.set_title("Three distinct clusters (4 points each) + 2 bridge points")
    ax_bridge.legend()
    ax_bridge.set_aspect("equal")
    ax_bridge.grid(alpha=0.3)

    mo.as_html(fig_bridge)
    plt.close(fig_bridge)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Fuzzy membership

    - Let $u_j: \mathbf{x} \rightarrow [0, 1]$.
    - Subject to: $\sum_{j=1}^{^m} u_j (\mathbf{x}_i)=1$ for $i=1,\cdots,N$
    - Going from discrete to continuous assignments
        - Opens the door for derivative-based optimization.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Proximity measures

    - Both dissimilarity and similarity
        - Between pairs of vectors
        - Between vector and set
        - Between pairs of sets
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Proximity measures between pairs of vectors

    - Let d : X × X → ℝ be a dissimilarity (distance) between vectors.

    - Non‑negativity: d(x, y) ≥ d_0 for all x, y.

    - Identity: d(x, x) = d_0 for all x.

    - Symmetry: d(x, y) = d(y, x) for all x, y.

    - Triangle inequality: d(x, z) ≤ d(x, y) + d(y, z) for all x, y, z.

    - A function satisfying 1–4 is a metric.

    - Note: some useful proximity measures (similarities, asymmetric scores, etc.) violate one or more of these properties.
        - Divergence measures
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Examples for dissimilarity measure

    - Weighted $l_p$ metric dissimilarity measure: $d(\mathbf{x}, \mathbf{y}) = \big( \sum_{i=1}^{l} w_i|x_i-y_i|^p\big)^{1/p}$ for $\mathbf{x}, \mathbf{y} \in \mathbb{R}$.
    - Euclidean distance:
        - $d_2(\mathbf{x}, \mathbf{y}) = \sqrt{{\sum_{i=1}^{l} (x_i-y_i)^2}}$
    - Manhattan:
        - $d_1(\mathbf{x}, \mathbf{y}) = {\sum_{i=1}^{l} |x_i-y_i|}$
    - l-infinity:
        $d_\infty(\mathbf{x}, \mathbf{y}) = \text{max} |x_i-y_i|$ for $1\leq i \leq l$
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Unit-ball visualisation for the L2, L1 and L-infinity norms so
    # students can see the geometric differences between common
    # dissimilarity measures.
    xs = np.linspace(-1.5, 1.5, 401)
    ys = np.linspace(-1.5, 1.5, 401)
    X_norms, Y_norms = np.meshgrid(xs, ys)

    L2 = np.sqrt(X_norms ** 2 + Y_norms ** 2)
    L1 = np.abs(X_norms) + np.abs(Y_norms)
    Linf = np.maximum(np.abs(X_norms), np.abs(Y_norms))

    fig_norms, ax_norms = plt.subplots(figsize=(6, 4))
    ax_norms.contour(X_norms, Y_norms, L2, levels=[1.0], colors="C0",
                     linewidths=2, linestyles="-")
    ax_norms.contour(X_norms, Y_norms, L1, levels=[1.0], colors="C1",
                     linewidths=2, linestyles="--")
    ax_norms.contour(X_norms, Y_norms, Linf, levels=[1.0], colors="C2",
                     linewidths=2, linestyles=":")
    ax_norms.scatter(0, 0, color="k", zorder=5)
    ax_norms.set_aspect("equal")
    ax_norms.set_xlim(-1.5, 1.5)
    ax_norms.set_ylim(-1.5, 1.5)
    ax_norms.set_xlabel("x")
    ax_norms.set_ylabel("y")
    ax_norms.set_title("Unit balls: L2 (Euclidean), L1 (Manhattan), L∞ (Chebyshev)")
    ax_norms.grid(alpha=0.3)

    mo.as_html(fig_norms)
    plt.close(fig_norms)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Examples of similarity measures

    - Inner-product: $$ s_i(\mathbf{x},\mathbf{y}) = \mathbf{x}^T \mathbf{y}$$
    - Cosine: $$ s_i(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^T \mathbf{y}}{||\mathbf{x}|| ||\mathbf{y}||} $$
    - From dissimilarity measure: $s = \frac{a}{d}$ or $s = d_{\text{max}}-d$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Between vector and set C

    - Max proximity: $$\rho_{\text{max}}(\mathbf{x}, C) = \underset{ \mathbf{y} \in C}{\max}\; \rho(\mathbf{x}, \mathbf{y})$$
    - Min proximity: $$\rho_{\text{min}}(\mathbf{x}, C) = \underset{ \mathbf{y} \in C}{\min}\; \rho(\mathbf{x}, \mathbf{y})$$
    - Average proximity: $$\rho_{\text{avg}}(\mathbf{x}, C) = \frac{1}{n_c}\sum_{\mathbf{y} \in C} \rho(\mathbf{x}, \mathbf{y}) $$

    - And similarly for proximity between set $C_i$ and $C_j$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Basic sequential algorithmic scheme (BSAS)

    - Examine all samples in $X = \{\mathbf{x_1}, \cdots, \mathbf{x_N}\}$ sequentially.
    - Pseudo-code:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks to BSAS

    - Order important!
    - Threshold important!
    - Vectors are assigned to clusters before the number of clusters are determined.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Modified Basic sequential algorithmic scheme (MBSAS)

    - First; determine clusters.
    - Second: pattern classification.
    - Requires two passes through the data.
    - Pseudo-code:
        """
    )
    return


if __name__ == "__main__":
    app.run()
