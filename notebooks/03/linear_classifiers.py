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
# Marimo version of the linear classifiers lecture.
# Same content as notebooks/03/linear_classifiers.ipynb, but authored as
# a reactive Marimo app so each matplotlib figure stays tied to the data
# generation that produced it (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/03/linear_classifiers.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `x1`) can be redefined
# in any number of cells, but Marimo requires each name to be owned by
# exactly one cell. Below, every cell-local dataset and figure is given a
# unique suffix (`_intro`, `_three`, `_wh`, `_p212`, `_wine`) so the
# slides stay independent and the data each figure displays is
# reproducible.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/linear_classifiers.slides.json",
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
    # Linear classifiers
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Up until now, our classifiers have been designed based on probability density functions.
    - In some cases, these classifiers were equivalent to linear discriminant functions.
    - Now, we will design linear discriminant functions, *regardless of the underlying distribution*.
    - Linear classifiers are very useful:
        - "First thing you try".
        - Often an underlying part of more complex algorithms.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Two well-separated Gaussian clusters — the running example used
    # throughout the lecture to motivate linear discriminant functions.
    n_intro = 7

    mu1_intro = np.array([1, 1])
    mu2_intro = np.array([2.5, 2.5])
    sigma_intro = np.array([[0.1, 0.0], [0.0, 0.1]])

    x1_intro = np.random.multivariate_normal(mu1_intro, sigma_intro, n_intro)
    x2_intro = np.random.multivariate_normal(mu2_intro, sigma_intro, n_intro)

    fig_intro, ax_intro = plt.subplots(figsize=(5, 5))
    ax_intro.scatter(
        x1_intro[:, 0], x1_intro[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_intro.scatter(
        x2_intro[:, 0], x2_intro[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2",
    )
    ax_intro.set_xlim(0, np.max(x2_intro[:, 0]) + 1.5)
    ax_intro.set_ylim(0, np.max(x2_intro[:, 1]) + 1.5)
    ax_intro.set_xlabel("x1")
    ax_intro.set_ylabel("x2")
    ax_intro.set_xticks([0, 1, 2, 3, 4])
    ax_intro.set_yticks([0, 1, 2, 3, 4])
    ax_intro.legend()

    mo.as_html(fig_intro)
    plt.close(fig_intro)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### A high level view of linear classifiers

    - Assume we have a training set: $\{\mathbf{x}_i, y_i\}$, where $\mathbf{x} \in \mathbb{R}^d$
    - In terms of discriminant functions:
        - $\mathbf{x}$ belongs to class 1 if $g(\mathbf{x})>0$.
        - How to design $g$?
    - Can use:
        - $g(\mathbf{x}) = \mathbf{w}^T\mathbf{x}$.
        - $\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_d \\ w_0 \end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \\ 1 \end{bmatrix}$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Mean squared error (MSE) classifier

    - Idea:
        - $g(\mathbf{x}_i)=\mathbf{w}^T\mathbf{x}_i \approx y_i$
    - How can we find weights that gives the desired output?
    - We do this by designing a *cost function* that would encourage this behavior:
        - $J(\mathbf{w})=\frac{1}{2}\mathbb{E}_{\mathbf{x}} \left[ (y-\mathbf{w}^T\mathbf{x}) \right]$
    - $\widehat{\mathbf{w}} = \underset{\mathbf{w}}{\operatorname{argmin}}\ J(\mathbf{w})$
    - Want:
        - $\frac{\partial}{\partial \mathbf{w}} J(\mathbf{w}) = 0$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Calculating weights based on MSE loss

    - Now, we derive an expression for calculating the weights of a linear
      classifier with a MSE loss.
    - Recall:
        - $\mathbb{E}[g(\mathbf{x})] = \int g(\mathbf{x})p(\mathbf{x})d\mathbf{x}$
    - Hence:
        - $J(\mathbf{w}) = \frac{1}{2} \int \left[ (y-\mathbf{w}^T\mathbf{x}) \right]^2 p(\mathbf{x})d\mathbf{x}$
    - Using the law of total probablity we get:
        - $J(\mathbf{w}) = \frac{1}{2}p(w_1) \int \left[ (***-\mathbf{w}^T\mathbf{x}) \right]^2 p(\mathbf{x}|w_1)d\mathbf{x} + \frac{1}{2}p(w_2) \int \left[ (***-\mathbf{w}^T\mathbf{x}) \right]^2 p(\mathbf{x}|w_2)d\mathbf{x}$
    - Remember the chain rule ->
    - Followed by a vector derivative:
        - $\frac{\partial}{\partial \mathbf{w}} (1-\mathbf{w}^T\mathbf{x}) = \begin{bmatrix} \hspace{3.0cm} \\ \\ \vdots \\ \\ \end{bmatrix} = -\mathbf{x}$
    - Thus:
        - $\frac{\partial}{\partial \mathbf{w}} J(\mathbf{w}) = -\mathbb{E}[\mathbf{x}(y-\mathbf{w}^T\mathbf{x})]=\mathbf{0}$
    - Finally:
        - $\widehat{\mathbf{w}} = \mathbf{R}^{-1}_{\mathbf{x}}\mathbb{E}[\mathbf{x}y]$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Understanding the MSE weights

    - Note
        - Second order moment (variance, assuming mean zero).
        - Assumes knwoledge about probability density function!
    - Example under assumption normally distributed data:
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Three-class example: introduces a third Gaussian cluster so the
    # MSE / LS solution has to handle multi-class targets (here
    # {0, 1, 2}) rather than the binary case shown earlier.
    n_three = 7

    mu1_three = np.array([1, 1])
    mu2_three = np.array([2.5, 2.5])
    mu3_three = np.array([0.1, 2.5])
    sigma_three = np.array([[0.1, 0.0], [0.0, 0.1]])

    x1_three = np.random.multivariate_normal(mu1_three, sigma_three, n_three)
    x2_three = np.random.multivariate_normal(mu2_three, sigma_three, n_three)
    x3_three = np.random.multivariate_normal(mu3_three, sigma_three, n_three)

    fig_three, ax_three = plt.subplots(figsize=(5, 5))
    ax_three.scatter(
        x1_three[:, 0], x1_three[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_three.scatter(
        x2_three[:, 0], x2_three[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2",
    )
    ax_three.scatter(
        x3_three[:, 0], x3_three[:, 1],
        s=120, facecolors="none", edgecolors="red", linewidth=3.0, label="Class 3",
    )
    ax_three.legend()

    mo.as_html(fig_three)
    plt.close(fig_three)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interactive: MSE decision boundary in 2D

    - Pick a setting below to see how the MSE classifier partitions the
      two classes in 2D.
    - Toggle the **decision boundary** to overlay the linear separator
      that the MSE solution (with targets $y \in \{0, 1\}$) places at
      $\mathbf{w}^T\mathbf{x}=0.5$.
    - The last setting replaces class 1 with a *uniform* distribution in
      a square to break the Gaussian assumption. The MSE boundary is no
      longer Bayes-optimal once the normal-data assumption fails.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Four scenarios. The first three keep the Gaussian assumption and
    # illustrate how mean separation and per-class variance change the
    # MSE/LS solution. The last one replaces class 1 with a uniform
    # distribution in a square to show what happens when the normal
    # assumption breaks down — the MSE boundary becomes biased toward
    # the uniform cluster and stops matching the Bayes-optimal split.
    scenarios_lc = {
        "μ₁=(1,1), μ₂=(3,3), σ=0.4 (well-separated, equal σ)": (
            "normal", np.array([1.0, 1.0]), 0.4,
            "normal", np.array([3.0, 3.0]), 0.4,
        ),
        "μ₁=(1,1), μ₂=(3,3), σ₁=0.3, σ₂=0.8 (equal means, different σ)": (
            "normal", np.array([1.0, 1.0]), 0.3,
            "normal", np.array([3.0, 3.0]), 0.8,
        ),
        "μ₁=(1,1), μ₂=(2,2), σ=0.4 (close means)": (
            "normal", np.array([1.0, 1.0]), 0.4,
            "normal", np.array([2.0, 2.0]), 0.4,
        ),
        "Class 1 uniform, Class 2 Gaussian (non-Gaussian)": (
            "uniform", np.array([1.0, 1.0]), 0.5,
            "normal", np.array([3.0, 3.0]), 0.4,
        ),
    }
    return (scenarios_lc,)


@app.cell
def _(mo, scenarios_lc):
    # Widgets must live in their own cell — Marimo forbids reading a
    # widget's `.value` in the same cell that created it.
    tabs_lc = mo.ui.tabs(
        scenarios_lc,
        value="μ₁=(1,1), μ₂=(3,3), σ=0.4 (well-separated, equal σ)",
    )
    switch_boundary = mo.ui.switch(
        value=True, label="Show decision boundary",
    )
    mo.vstack(
        [
            tabs_lc,
            mo.hstack(
                [switch_boundary, mo.md("  Toggle the decision boundary above.")],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return switch_boundary, tabs_lc


@app.cell
def _(mo, np, plt, scenarios_lc, switch_boundary, tabs_lc):
    # Read the active scenario from the tabs widget.
    (c1_type, c1_loc, c1_scale,
     c2_type, c2_loc, c2_scale) = scenarios_lc[tabs_lc.value]

    n_lc = 80

    def sample_lc(kind, loc, scale, n):
        if kind == "normal":
            cov = (scale ** 2) * np.eye(2)
            return np.random.multivariate_normal(loc, cov, n)
        # uniform inside a square of half-side `scale` around `loc`
        return loc + scale * (2 * np.random.rand(n, 2) - 1)

    x1_lc = sample_lc(c1_type, c1_loc, c1_scale, n_lc)
    x2_lc = sample_lc(c2_type, c2_loc, c2_scale, n_lc)

    # MSE/LS solution: targets y = 0 for class 1, y = 1 for class 2,
    # augmented with a column of ones for the bias term. Solve the
    # normal equations and read off the decision boundary at 0.5.
    X_lc = np.vstack([x1_lc, x2_lc])
    y_lc = np.concatenate([np.zeros(n_lc), np.ones(n_lc)])
    X_aug_lc = np.hstack([X_lc, np.ones((2 * n_lc, 1))])
    w_lc = np.linalg.solve(X_aug_lc.T @ X_aug_lc, X_aug_lc.T @ y_lc)
    w0_lc, w1_lc, b_lc = w_lc

    fig_lc, ax_lc = plt.subplots(figsize=(7, 6))
    ax_lc.scatter(
        x1_lc[:, 0], x1_lc[:, 1],
        s=40, facecolors="none", edgecolors="black", linewidth=1.5,
        label="Class 1 (target 0)",
    )
    ax_lc.scatter(
        x2_lc[:, 0], x2_lc[:, 1],
        s=40, c="royalblue", alpha=0.5,
        label="Class 2 (target 1)",
    )

    if switch_boundary.value and abs(w1_lc) > 1e-9:
        # w0*x + w1*y + b = 0.5  ->  y = (0.5 - w0*x - b) / w1
        x_min, x_max = -0.5, 4.5
        xs = np.array([x_min, x_max])
        ys = (0.5 - w0_lc * xs - b_lc) / w1_lc
        ax_lc.plot(
            xs, ys, "r--", linewidth=2.5,
            label=f"MSE boundary  w=[{w0_lc:+.2f}, {w1_lc:+.2f}, {b_lc:+.2f}]",
        )
    elif switch_boundary.value and abs(w1_lc) <= 1e-9:
        # Vertical boundary case.
        ax_lc.axvline(
            x=(0.5 - b_lc) / w0_lc, color="red", linestyle="--", linewidth=2.5,
            label=f"MSE boundary  w=[{w0_lc:+.2f}, {w1_lc:+.2f}, {b_lc:+.2f}]",
        )

    ax_lc.set_xlim(-0.5, 4.5)
    ax_lc.set_ylim(-0.5, 4.5)
    ax_lc.set_xlabel("x1")
    ax_lc.set_ylabel("x2")
    ax_lc.set_title(f"Setting: {tabs_lc.value}")
    ax_lc.legend(loc="lower right", fontsize=9)
    ax_lc.grid(True, alpha=0.3)

    plt.close(fig_lc)
    mo.vstack(
        [
            mo.md(f"**MSE weights:** {w_lc.round(3)}"),
            mo.as_html(fig_lc),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Least sum of error squares (LS) classifier

    - Drawback of MSE classifier:
        - Assumes knowledge about the underlying distribution -> why not use Bayes classifier?
    - How can we design a linear classifier without relying on knowledge about the distribution of the data?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Least sum of error squres cost function

    - Consider the following loss function:
    - $J(\mathbf{w}) = \sum_{i=1}^{N} \left(y-\mathbf{w}^T\mathbf{x}_i\right)^2$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Differentiating and setting to zero

    - $\frac{\partial}{\partial \mathbf{w}} J(\mathbf{w}) = \mathbf{0} \rightarrow \sum_{i=1}^{N} -\mathbf{x}_i\left(y-\mathbf{w}^T\mathbf{x}_i\right)=0$
    - Collecting terms and rearranging:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Putting it all together

    - Assume data is stored columnwise.
        - $\mathbf{X}$=
        - $\mathbf{y}$=
    - Then, we get:
        - $(\mathbf{X}^T\mathbf{X})\mathbf{w}=\mathbf{X}^T \mathbf{y}$
        - $\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Understanding the LS classifier

    - Basically, MSE in practice. Optimal in the case of normal data.
    - Connected to Bayes classifier under normal and equal variance assumption.
    - Goes way back:
        - One of the earliest known uses is by Gauss for predicting motion of celestial bodies.
        - Astronomor Piazzi discovered Ceres. He made some observation before Ceres was obscured by the sun.
        - When would it return?
        - Could rely on equations for planetary motion.
        - Gauss used a least sum of error squared approach using the data collected by Piazzi.
    - Least sum of error squared approaches are used all the time.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Limitations of MSE and LS classifier

    - Batch learning
    - Solves equation in one step
        - Correlation matrix and inverse of correlation matrix.
    - What could be potential problems?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Widow-Hoff algorithm (1960)

    - Idea: iterative gradient descent on MSE loss.
    - $\widehat{\mathbf{w}}(k) = \widehat{\mathbf{w}}(k-1)+\rho_k \frac{\partial}{\partial \widehat{\mathbf{w}}(k-1)} J(\widehat{\mathbf{w}}(k-1))$
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Two toy cost landscapes over a single scalar weight: a strictly
    # convex quadratic (J_1) and a slightly bumpy sin + quadratic (J_2)
    # to motivate why iterative gradient descent needs care. The
    # original notebook labels both with J_1's formula; we use the
    # accurate expression for the right panel.
    w_cost = np.linspace(-10, 10, 100)

    def J_1(w): return w**2 + 2*w + 1
    def J_2(w): return np.sin(w) + 0.1 * w**2

    J_values_1 = J_1(w_cost)
    J_values_2 = J_2(w_cost)

    fig_cost, (ax_cost_1, ax_cost_2) = plt.subplots(1, 2, figsize=(10, 4))
    ax_cost_1.plot(w_cost, J_values_1, color="blue")
    ax_cost_1.set_xlabel("w")
    ax_cost_1.set_ylabel("J(w)")
    ax_cost_1.set_title("J(w) = w^2 + 2w + 1")
    ax_cost_2.plot(w_cost, J_values_2, color="blue")
    ax_cost_2.set_xlabel("w")
    ax_cost_2.set_title("J(w) = sin(w) + 0.1 w^2")

    mo.as_html(fig_cost)
    plt.close(fig_cost)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Derivatie of MSE loss

    - Need to calculate $$\frac{\partial}{\partial \widehat{\mathbf{w}}(k-1)} J(\widehat{\mathbf{w}}(k-1))$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example of Widrow-Hoff update
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Same two-class setup as the intro, but with tighter clusters and
    # a smaller inter-class distance — used for the in-class
    # Widrow-Hoff / Perceptron update trace.
    n_wh = 7

    mu1_wh = np.array([1, 1])
    mu2_wh = np.array([2.0, 2.0])
    sigma_wh = np.array([[0.1, 0.0], [0.0, 0.1]])

    x1_wh = np.random.multivariate_normal(mu1_wh, sigma_wh, n_wh)
    x2_wh = np.random.multivariate_normal(mu2_wh, sigma_wh, n_wh)

    fig_wh, ax_wh = plt.subplots(figsize=(5, 5))
    ax_wh.scatter(
        x1_wh[:, 0], x1_wh[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_wh.scatter(
        x2_wh[:, 0], x2_wh[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2",
    )
    ax_wh.legend()

    mo.as_html(fig_wh)
    plt.close(fig_wh)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How do we stop?

    - Check value of cost function $$J(\widehat{\mathbf{w}}(k)) \simeq J(\widehat{\mathbf{w}}(k-1))$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Testing

    - $\widehat{\mathbf{w}}$ is now fixed!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Perceptron

    - Proposed by Frank Rosenblatt in the 1950s
    - A model of neurons in the brain.
    - How to interpret the components in the linear classifiers we have seen so far?
        - $\mathbf{x}$: sensory input
        - $w_1, \cdots , w_d$: synapses with different strenghts
        - f: a neuron fires or not.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The online/iterative Perceptron algorithm

    - $\widehat{\mathbf{w}}(k) = \widehat{\mathbf{w}}(k-1)+\rho_k f(\mathbf{x}_k)\mathbf{x}_k$
    - If $\text{sign}(y_k) \neq \text{sign}(f(\mathbf{x}_k)$
        - missclassified.
    - Hence; do nothing if correctly classified and update if missclassified
    - When do we stop?
    - Can update in a batch-wise manner.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example

    - Show code example in linearly separable data.
        """
    )
    return


if __name__ == "__main__":
    app.run()
