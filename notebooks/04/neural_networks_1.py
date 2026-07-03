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
# Marimo version of the Neural Networks I lecture.
# Same content as notebooks/04/neural_networks_1.ipynb, but authored as
# a reactive Marimo app so each matplotlib figure stays tied to the data
# generation that produced it (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/04/neural_networks_1.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `x_inner`) can be
# redefined in any number of cells, but Marimo requires each name to be
# owned by exactly one cell. Below, every cell-local dataset and figure
# is given a unique suffix (`_circles`, `_xor`, `_xor_ex`,
# `_moons`) so the slides stay independent and the data each figure
# displays is reproducible.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks_1.slides.json",
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
    # Neural Networks I
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Our focus has so far been on linear classifiers. Now, our focus will shift to **non-linear classifiers**.
    - We will start with learning about neural networks.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Two concentric noisy circles — the canonical "non-linearly
    # separable" running example used to motivate everything that
    # follows in the lecture.
    n_samples_circles = 200

    r_inner_circles = 1
    theta_inner_circles = 2 * np.pi * np.random.rand(n_samples_circles // 2)
    x_inner_circles = r_inner_circles * np.cos(theta_inner_circles) + 0.2 * np.random.randn(n_samples_circles // 2)
    y_inner_circles = r_inner_circles * np.sin(theta_inner_circles) + 0.2 * np.random.randn(n_samples_circles // 2)

    r_outer_circles = 2.5
    theta_outer_circles = 2 * np.pi * np.random.rand(n_samples_circles // 2)
    x_outer_circles = r_outer_circles * np.cos(theta_outer_circles) + 0.2 * np.random.randn(n_samples_circles // 2)
    y_outer_circles = r_outer_circles * np.sin(theta_outer_circles) + 0.2 * np.random.randn(n_samples_circles // 2)

    fig_circles, ax_circles = plt.subplots(figsize=(8, 8))
    ax_circles.scatter(x_inner_circles, y_inner_circles, color="blue", edgecolor="k",
                       s=80, label="Class 1")
    ax_circles.scatter(x_outer_circles, y_outer_circles, color="red", edgecolor="k",
                       s=80, label="Class 2")
    ax_circles.set_aspect("equal")
    ax_circles.set_xlabel("x1")
    ax_circles.set_ylabel("x2")
    ax_circles.legend()
    fig_circles.tight_layout()

    plt.close(fig_circles)
    mo.as_html(fig_circles)
    return


@app.cell
def _(mo):
    # Timeline slide: heading + photo strip in one cell. Heights are
    # uniform so the row of mixed-aspect images (one portrait, one
    # wide landscape, two near-square) lines up cleanly.
    mo.vstack(
        [
            mo.md(
                r"""
            ## Neural networks - a brief history

            - McCulloch - Pitts neuron (1946).
            - The Perceptron (Rosenblatt, 1950's).
            - The XOR problem (Minsky and Papert, 1969) - the first AI winter.
            - (Debated topic) Backpropagation (Rumelhart, Hinton, Williams, 1986) - a new hope.
            - Lacked data and compute, difficult to train (1990s) - the second AI winter.
            - AlexNet (Krizhevsky, Sutskever, Hinton, 2013) - the deep learning revolution.
                """
            ),
            mo.hstack(
                [
                    mo.image(src="media/mcpitts.png", height="180px"),
                    mo.image(src="media/rosenblatt.jpg", height="180px"),
                    mo.image(src="media/mnp.png", height="180px"),
                    mo.image(src="media/alexnet.jpeg", height="180px"),
                ],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The XOR problem (Minsky and Papert, 1969)

    - The Perceptron caused a lot of excitement.
    - Researchers started pushing the limit of the Perceptron.
    - Problem:
        - The Perceptron cannot solve the XOR problem!
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # XOR dataset: four small Gaussian blobs at the corners of the
    # unit square with labels {0, 1, 1, 0}. The original notebook sets
    # seed 42 for reproducibility across renders.
    np.random.seed(42)

    n_samples_xor = 50
    std_xor = 0.15

    centers_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels_xor = np.array([0, 1, 1, 0])

    X_xor = []
    y_xor = []
    for center_xor, label_xor in zip(centers_xor, labels_xor):
        X_xor.append(center_xor + std_xor * np.random.randn(n_samples_xor, 2))
        y_xor.append(np.full(n_samples_xor, label_xor))
    X_xor = np.vstack(X_xor)
    y_xor = np.concatenate(y_xor)

    fig_xor, ax_xor = plt.subplots(figsize=(8, 8))
    ax_xor.scatter(
        X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
        color="blue", s=180, edgecolor="k", alpha=0.75, label="Class 0",
    )
    ax_xor.scatter(
        X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
        color="red", s=180, edgecolor="k", alpha=0.75, label="Class 1",
    )

    ax_xor.set_aspect("equal")
    ax_xor.set_xlabel("x1")
    ax_xor.set_ylabel("x2")
    ax_xor.set_xlim(-0.55, 1.55)
    ax_xor.set_ylim(-0.55, 1.55)
    ax_xor.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig_xor.tight_layout()

    plt.close(fig_xor)
    mo.as_html(fig_xor)
    return X_xor, y_xor


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Transforming into a linearly separable representation

    - The perceptron requires linearly separable data.
    - What if we could transform the data to accomplish this?
        """
    )
    return


@app.cell
def _(X_xor, mo, plt, y_xor):
    # Side-by-side: original XOR (no linear boundary works) and an
    # empty canvas the lecturer can draw the desired transformation on
    # by hand. Right panel is intentionally blank.
    fig_xform, (ax_xform_1, ax_xform_2) = plt.subplots(1, 2, figsize=(12, 5))
    ax_xform_1.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], color="blue", label="Class 0")
    ax_xform_1.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], color="red", label="Class 1")
    ax_xform_1.set_aspect("equal")
    ax_xform_1.set_xlabel("x1")
    ax_xform_1.set_ylabel("x2")
    ax_xform_1.legend()
    ax_xform_2.set_ylim(-0.5, 1.5)
    ax_xform_2.set_xlim(-0.5, 1.5)
    ax_xform_2.set_aspect("equal")
    ax_xform_2.set_xlabel("x1")
    ax_xform_2.set_ylabel("x2")
    fig_xform.tight_layout()

    plt.close(fig_xform)
    mo.as_html(fig_xform)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The big idea

    - Transform into a new representation where data is linearly separable.
    - Transformation is done by another Perceptron!
    - Stack layers of Perceptrons to form a **Multilayer Perceptron** (MLP).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### A two-layer Perceptron (2LP)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Classification capabilites of 2LP

    - A hidden layer maps $\mathbf{x}\in \mathcal{R}^d$ onto vertices of a p-dimensional cube (hidden layer with p neurons).
    - A 2LP can separate classes represented as union of polyhedral regions.
    - Not all regions can be separated!
        - More layers increases the capacity of the network.
        - But how to train?
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 3D illustration for "classification capabilities of a 2LP".
    # Draws the input cube with Gaussian clusters at its corners —
    # labelled as class 0 (blue, even-sum corners) and class 1
    # (red, odd-sum corners). The figure is intentionally large and
    # uncluttered so the lecturer can sketch polyhedral separating
    # regions (e.g. planes / convex hulls) on top of it during class.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    corners_2lp = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=float,
    )
    labels_2lp = np.array([0, 1, 1, 0, 1, 0, 0, 1])

    rng_2lp = np.random.default_rng(7)
    n_per_corner_2lp = 25
    std_2lp = 0.10

    X_2lp = np.vstack([
        c + std_2lp * rng_2lp.standard_normal((n_per_corner_2lp, 3))
        for c in corners_2lp
    ])
    y_2lp = np.concatenate([np.full(n_per_corner_2lp, lab) for lab in labels_2lp])

    fig_2lp = plt.figure(figsize=(12, 9))
    ax_2lp = fig_2lp.add_subplot(111, projection="3d")

    ax_2lp.scatter(
        X_2lp[y_2lp == 0, 0], X_2lp[y_2lp == 0, 1], X_2lp[y_2lp == 0, 2],
        c="royalblue", s=70, edgecolor="k", alpha=0.85, label="Class 0",
        depthshade=True,
    )
    ax_2lp.scatter(
        X_2lp[y_2lp == 1, 0], X_2lp[y_2lp == 1, 1], X_2lp[y_2lp == 1, 2],
        c="crimson",  s=70, edgecolor="k", alpha=0.85, label="Class 1",
        depthshade=True,
    )

    # Draw the cube edges so the input region is unmistakable and
    # the lecturer has clear surfaces to draw separating planes on.
    edge_start_end = [
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
        ((1, 1, 1), (0, 1, 1)), ((1, 1, 1), (1, 0, 1)), ((1, 1, 1), (1, 1, 0)),
        ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)),
        ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)),
    ]
    for (s, e) in edge_start_end:
        ax_2lp.plot(
            [s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
            color="black", linewidth=1.2,
        )

    ax_2lp.set_xlim(-0.4, 1.4)
    ax_2lp.set_ylim(-0.4, 1.4)
    ax_2lp.set_zlim(-0.4, 1.4)
    ax_2lp.set_xlabel("x1")
    ax_2lp.set_ylabel("x2")
    ax_2lp.set_zlabel("x3")
    ax_2lp.set_title(
        "Classification capabilities of a 2LP — Gaussian clusters at the corners of the input cube",
        fontsize=13,
    )
    ax_2lp.legend(loc="upper left", fontsize=11)
    ax_2lp.view_init(elev=22, azim=35)
    fig_2lp.tight_layout()

    plt.close(fig_2lp)
    mo.as_html(fig_2lp)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Backpropagation algorithm

    - An efficient training algorithm for MLPs
    - [Fun blogpost on the topic of who invented Backpropagation.](https://people.idsia.ch/~juergen/who-invented-backpropagation.html)
    - Parallel to Stigler's law of eponymy
    - A string of important papers:
        - Reverse mode of automatic differentiation (Linnainmaa 1970)
        - Early work on training MLPs with gradient descent (Amari, 1967)
        - Preliminary application of backpropagation in the contex of neural network (Werbos, 1982)
        - Learning representations by back-propagating erros (Rumelhart, 1986)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Backpropagation algorithm
        """
    )
    return


@app.cell
def _(mo):
    # MLP diagram reused several slides later; referenced explicitly
    # so students can map the math back to the picture.
    mo.image(src="media/mlp.jpg", width="700px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Backpropagation algorithm - loss and optimization

    $$\displaystyle J = \sum_{i=1}^N E (i),$$

    $$\displaystyle E (i) = \sum_{m=1}^{N} \tfrac{1}{2}\, e_m(i)^2$$

    - Want: $\displaystyle\underset{\mathbf{w}}{\operatorname{min}}\ J(\mathbf{w})$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Backpropagation algorithm - gradients of the output layer

    $$\displaystyle \frac{\partial E (i)}{\partial \mathbf{w}_j^L} \;=\;$$

    ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Gradient descent reminder

    $$\displaystyle \mathbf{w}_j^L (t+1) \;=\; \mathbf{w}_j^L (t) \;-\; \gamma\, \frac{\partial}{\partial \mathbf{w}_j^L} J$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Backpropagation algorithm - gradients of the hidden layers

    $$\displaystyle \frac{\partial E (i)}{\partial \mathbf{w}_j^{L-1}} \;=\;$$

    ---

    - $z_j^{L-1}$ not present in $E (i)$, comes through $a_j^{L-1}$ and onward to the output.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Gradients of the hidden layers - chain rule

    $$\displaystyle \frac{\partial E (i)}{\partial a_j^{L-1} (i) } \;=\; \delta_j^{L-1}(i) \;=\;$$

    ---

    $$\displaystyle \frac{\partial a_j^{L} (i)}{\partial a_j^{L-1} (i) } \;=\;$$

    ---

    $$\displaystyle \delta_j^{L-1}(i) \;=\;$$

    ---

    - Error at output $(L)$ propagated to $(L-1)$ to compute gradients of weights.
    - Repeat to compute gradients for the entire network!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Backpropagation recipe

    - Initialize parameters and "freeze" them.
    - The forward pass: compute all
    - Backward pass: compute
    - Update all parameters
    - Repeat!
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
        ### Top tips for implementing and understanding backpropagation

        - Keep it simple in the begining!
        - Start with a fixed architecture.
            - Do not have the same number of neurons in a layer.
        - Do all calculations by hand before you start programming.
        - When you start programming, cross reference your implementation with your hand-calculations.
        - There is only one loop, the outer loop of the number of epochs.
            - Matrix multiplication!
                """
            ),
            mo.image(src="media/mlp.jpg", width="500px"),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
