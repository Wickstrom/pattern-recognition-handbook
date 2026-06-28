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
        """
    )
    return


@app.cell
def _(mo):
    # Lives in its own cell so the rendering cell below can read
    # show_boundary_lc.value without violating Marimo's "no reading
    # a UIElement in the cell that created it" rule.
    show_boundary_lc = mo.ui.switch(value=False, label="Show decision boundary")
    return (show_boundary_lc,)


@app.cell
def _(mo, np, plt, show_boundary_lc):
    # Three scenarios to keep the slide focused. The first two stress
    # the MSE solution under Gaussian assumptions (close means with
    # equal vs. different variances). The third deliberately violates
    # the Gaussian assumption by giving class 1 a *moon* shape and
    # class 2 a Gaussian placed well away from it, so the MSE
    # straight-line boundary is no longer Bayes-optimal.
    scenarios_lc = {
        "Option 1": (
            "normal", np.array([2.0, 2.0]), 0.6,
            "normal", np.array([3.5, 3.5]), 0.6,
        ),
        "Option 2": (
            "normal", np.array([2.0, 2.0]), 0.4,
            "normal", np.array([3.5, 3.5]), 0.9,
        ),
        "Option 3": (
            "moon", None, None,
            "normal", np.array([4.2, 1.5]), 0.5,
        ),
    }

    np.random.seed(0)
    n_lc = 80

    def sample_lc(kind, loc, scale, n):
        if kind == "normal":
            cov = (scale ** 2) * np.eye(2)
            return np.random.multivariate_normal(loc, cov, n)
        # Half-moon: upper 180-degree arc centered at (2.5, 2.5),
        # radius 1.4, plus a touch of thickness noise so the band has
        # a visible width rather than a hairline.
        theta = np.linspace(0, np.pi, n)
        moon_radius = 1.4
        moon_cx, moon_cy = 2.5, 2.5
        x = moon_cx + moon_radius * np.cos(theta)
        y = moon_cy + moon_radius * np.sin(theta)
        samples = np.column_stack([x, y])
        samples += np.random.normal(0, 0.1, samples.shape)
        return samples

    def make_fig_lc(title, c1_type, c1_loc, c1_scale, c2_type, c2_loc, c2_scale, highlight=False, show_boundary=False):
        x1 = sample_lc(c1_type, c1_loc, c1_scale, n_lc)
        x2 = sample_lc(c2_type, c2_loc, c2_scale, n_lc)

        # MSE/LS solution: targets y = 0 for class 1, y = 1 for class 2,
        # augmented with a column of ones for the bias term. Solve the
        # normal equations and read off the decision boundary at 0.5.
        X = np.vstack([x1, x2])
        y = np.concatenate([np.zeros(n_lc), np.ones(n_lc)])
        X_aug = np.hstack([X, np.ones((2 * n_lc, 1))])
        w = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
        w0, w1, b = w

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            x1[:, 0], x1[:, 1],
            s=40, facecolors="none", edgecolors="black", linewidth=1.5,
            label="Class 1 (target 0)",
        )
        ax.scatter(
            x2[:, 0], x2[:, 1],
            s=40, c="royalblue", alpha=0.5,
            label="Class 2 (target 1)",
        )

        if show_boundary:
            boundary_lw = 3.5 if highlight else 2.5
            if abs(w1) > 1e-9:
                # w0*x + w1*y + b = 0.5  ->  y = (0.5 - w0*x - b) / w1
                xs = np.array([0.5, 5.0])
                ys = (0.5 - w0 * xs - b) / w1
                ax.plot(
                    xs, ys, "r--", linewidth=boundary_lw,
                    label=f"MSE boundary  w=[{w0:+.2f}, {w1:+.2f}, {b:+.2f}]",
                )
            else:
                # Vertical boundary case.
                ax.axvline(
                    x=(0.5 - b) / w0, color="red", linestyle="--", linewidth=boundary_lw,
                    label=f"MSE boundary  w=[{w0:+.2f}, {w1:+.2f}, {b:+.2f}]",
                )

        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(0.5, 5.0)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if highlight:
            ax.set_title(title, fontweight="bold", color="darkred")
        else:
            ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        return fig, w

    tabs_lc = {}
    titles_lc = list(scenarios_lc.keys())
    for idx_lc, (title, params) in enumerate(scenarios_lc.items()):
        highlight_lc = idx_lc == len(scenarios_lc) - 1
        fig_lc, w_lc = make_fig_lc(title, *params, highlight=highlight_lc, show_boundary=show_boundary_lc.value)
        tabs_lc[title] = mo.as_html(fig_lc)
        plt.close(fig_lc)

    mo.vstack([show_boundary_lc, mo.ui.tabs(tabs_lc)])
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

    - $\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$
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

    - Need to calculate $\frac{\partial}{\partial \widehat{\mathbf{w}}(k-1)} J(\widehat{\mathbf{w}}(k-1))$
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
    # Widrow-Hoff / Perceptron update trace. Targets are encoded as
    # +1 / -1 so the Widrow-Hoff gradient step
    # w ← w + ρ (y − wᵀx) x and the perceptron step
    # w ← w + ρ y x fall out naturally.
    n_wh = 7

    mu1_wh = np.array([1, 1])
    mu2_wh = np.array([2.0, 2.0])
    sigma_wh = np.array([[0.1, 0.0], [0.0, 0.1]])

    x1_wh = np.random.multivariate_normal(mu1_wh, sigma_wh, n_wh)
    x2_wh = np.random.multivariate_normal(mu2_wh, sigma_wh, n_wh)
    X_wh = np.vstack([x1_wh, x2_wh])
    y_wh = np.concatenate([np.ones(n_wh), -np.ones(n_wh)])
    X_aug_wh = np.hstack([X_wh, np.ones((2 * n_wh, 1))])

    fig_wh, ax_wh = plt.subplots(figsize=(5, 5))
    ax_wh.scatter(
        x1_wh[:, 0], x1_wh[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1 (+1)",
    )
    ax_wh.scatter(
        x2_wh[:, 0], x2_wh[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2 (−1)",
    )
    ax_wh.legend()

    mo.as_html(fig_wh)
    plt.close(fig_wh)
    return X_aug_wh, X_wh, n_wh, x1_wh, x2_wh, y_wh


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interactive: Widrow-Hoff update

    - Start from *random* initial weights and click **Next update** to
      step through four Widrow-Hoff gradient updates:
      $\mathbf{w}(k+1) = \mathbf{w}(k) + \rho (y - \mathbf{w}^T\mathbf{x})\mathbf{x}$.
    - The point consumed by the most recent update is highlighted in
      red so you can see which sample drove the new boundary.
        """
    )
    return


@app.cell
def _(X_aug_wh, X_wh, n_wh, np, y_wh):
    # First interactive run: random initial weights drawn from
    # N(0, 0.5²) with a fixed seed so the trajectory is reproducible
    # across renders. The trace is pre-computed once for the four
    # sequential updates; the button picks which step to display.
    # We alternate class 1 / class 2 across the four steps so each
    # step makes a visible move (a pure class-1 sweep would bias the
    # boundary off-screen).
    rng_wh_a = np.random.default_rng(42)
    w_wh_a_init = rng_wh_a.normal(0, 0.5, size=3)
    rho_wh_a = 0.08

    n_steps_wh_a = 4
    trace_wh_a = [w_wh_a_init.copy()]
    update_idx_wh_a = []
    for _step in range(n_steps_wh_a):
        # Even step → class 1 point, odd step → class 2 point.
        _idx = (_step // 2) + (_step % 2) * n_wh
        _x_i = X_aug_wh[_idx]
        _y_i = y_wh[_idx]
        _w_new = trace_wh_a[-1] + rho_wh_a * (_y_i - np.dot(trace_wh_a[-1], _x_i)) * _x_i
        trace_wh_a.append(_w_new)
        update_idx_wh_a.append(_idx)

    return rho_wh_a, trace_wh_a, update_idx_wh_a, n_steps_wh_a


@app.cell
def _(mo, n_steps_wh_a):
    # One radio button per time step so the user can jump forward and
    # backward through the trajectory (rather than only stepping forward).
    step_btn_a = mo.ui.radio(
        options={f"Step {k}": k for k in range(n_steps_wh_a + 1)},
        value=0,
        label=f"Step (0..{n_steps_wh_a})",
    )
    return (step_btn_a,)


@app.cell
def _(
    X_wh,
    mo,
    n_steps_wh_a,
    np,
    plt,
    rho_wh_a,
    step_btn_a,
    trace_wh_a,
    update_idx_wh_a,
    x1_wh,
    x2_wh,
):
    _step_a = int(step_btn_a.value) if step_btn_a.value is not None else 0
    _w_a = trace_wh_a[_step_a]
    # The sample processed at the current step is update_idx_wh_a[k-1]
    # for step k ≥ 1; at step 0 no update has happened yet so we
    # preview the first sample that will be processed.
    _idx_cur_a = update_idx_wh_a[_step_a - 1] if _step_a >= 1 else update_idx_wh_a[0]

    fig_wh_a, ax_wh_a = plt.subplots(figsize=(5, 5))
    ax_wh_a.scatter(
        x1_wh[:, 0], x1_wh[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0,
        label="Class 1 (+1)",
    )
    ax_wh_a.scatter(
        x2_wh[:, 0], x2_wh[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0,
        label="Class 2 (−1)",
    )
    # Green ring on the sample being processed at this step.
    _x_h_a = X_wh[_idx_cur_a]
    ax_wh_a.scatter(
        [_x_h_a[0]], [_x_h_a[1]], s=450, facecolors="none",
        edgecolors="#2ca02c", linewidth=3.5, label="Current sample",
    )
    if abs(_w_a[1]) > 1e-9:
        # w0*x + w1*y + b = 0  ->  y = -(w0*x + b) / w1
        _xs_a = np.array([0.0, 3.0])
        _ys_a = -(_w_a[0] * _xs_a + _w_a[2]) / _w_a[1]
        ax_wh_a.plot(
            _xs_a, _ys_a, "r--", linewidth=2.5,
            label=f"Boundary  w=[{_w_a[0]:+.2f}, {_w_a[1]:+.2f}, {_w_a[2]:+.2f}]",
        )
    ax_wh_a.set_xlim(0.0, 3.0)
    ax_wh_a.set_ylim(0.0, 3.0)
    ax_wh_a.set_xlabel("x1")
    ax_wh_a.set_ylabel("x2")
    ax_wh_a.set_title(f"Widrow-Hoff, start A — step {_step_a}/{n_steps_wh_a}")
    ax_wh_a.legend(loc="upper right", fontsize=8)
    ax_wh_a.grid(True, alpha=0.3)
    plt.close(fig_wh_a)

    mo.vstack([
        step_btn_a,
        mo.md(
            f"**Step {_step_a} / {n_steps_wh_a}** &nbsp;—&nbsp; "
            f"weights: `{_w_a.round(3)}`"
        ),
        mo.as_html(fig_wh_a),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interactive: Widrow-Hoff update (different start)

    - Same four updates, same data — only the random initialization
      changes. Compare the trajectory to the previous slide.
        """
    )
    return


@app.cell
def _(X_aug_wh, X_wh, n_wh, np, y_wh):
    # Second interactive run with a different seed so the four
    # updates follow a noticeably different trajectory. Same
    # alternating order and learning rate as start A for a fair
    # side-by-side comparison.
    rng_wh_b = np.random.default_rng(123)
    w_wh_b_init = rng_wh_b.normal(0, 0.5, size=3)
    rho_wh_b = 0.08

    n_steps_wh_b = 4
    trace_wh_b = [w_wh_b_init.copy()]
    update_idx_wh_b = []
    for _step in range(n_steps_wh_b):
        _idx = (_step // 2) + (_step % 2) * n_wh
        _x_i = X_aug_wh[_idx]
        _y_i = y_wh[_idx]
        _w_new = trace_wh_b[-1] + rho_wh_b * (_y_i - np.dot(trace_wh_b[-1], _x_i)) * _x_i
        trace_wh_b.append(_w_new)
        update_idx_wh_b.append(_idx)

    return rho_wh_b, trace_wh_b, update_idx_wh_b, n_steps_wh_b


@app.cell
def _(mo, n_steps_wh_b):
    # One radio button per time step so the user can jump forward and
    # backward through the trajectory.
    step_btn_b = mo.ui.radio(
        options={f"Step {k}": k for k in range(n_steps_wh_b + 1)},
        value=0,
        label=f"Step (0..{n_steps_wh_b})",
    )
    return (step_btn_b,)


@app.cell
def _(
    X_wh,
    mo,
    n_steps_wh_b,
    np,
    plt,
    rho_wh_b,
    step_btn_b,
    trace_wh_b,
    update_idx_wh_b,
    x1_wh,
    x2_wh,
):
    _step_b = int(step_btn_b.value) if step_btn_b.value is not None else 0
    _w_b = trace_wh_b[_step_b]
    _idx_cur_b = update_idx_wh_b[_step_b - 1] if _step_b >= 1 else update_idx_wh_b[0]

    fig_wh_b, ax_wh_b = plt.subplots(figsize=(5, 5))
    ax_wh_b.scatter(
        x1_wh[:, 0], x1_wh[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0,
        label="Class 1 (+1)",
    )
    ax_wh_b.scatter(
        x2_wh[:, 0], x2_wh[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0,
        label="Class 2 (−1)",
    )
    # Green ring on the sample being processed at this step.
    _x_h_b = X_wh[_idx_cur_b]
    ax_wh_b.scatter(
        [_x_h_b[0]], [_x_h_b[1]], s=450, facecolors="none",
        edgecolors="#2ca02c", linewidth=3.5, label="Current sample",
    )
    if abs(_w_b[1]) > 1e-9:
        _xs_b = np.array([0.0, 3.0])
        _ys_b = -(_w_b[0] * _xs_b + _w_b[2]) / _w_b[1]
        ax_wh_b.plot(
            _xs_b, _ys_b, "r--", linewidth=2.5,
            label=f"Boundary  w=[{_w_b[0]:+.2f}, {_w_b[1]:+.2f}, {_w_b[2]:+.2f}]",
        )
    ax_wh_b.set_xlim(0.0, 3.0)
    ax_wh_b.set_ylim(0.0, 3.0)
    ax_wh_b.set_xlabel("x1")
    ax_wh_b.set_ylabel("x2")
    ax_wh_b.set_title(f"Widrow-Hoff, start B — step {_step_b}/{n_steps_wh_b}")
    ax_wh_b.legend(loc="upper right", fontsize=8)
    ax_wh_b.grid(True, alpha=0.3)
    plt.close(fig_wh_b)

    mo.vstack([
        step_btn_b,
        mo.md(
            f"**Step {_step_b} / {n_steps_wh_b}** &nbsp;—&nbsp; "
            f"weights: `{_w_b.round(3)}`"
        ),
        mo.as_html(fig_wh_b),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How do we stop?

    - Check value of cost function $J(\widehat{\mathbf{w}}(k)) \simeq J(\widehat{\mathbf{w}}(k-1))$
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

    - Same data and the same random initialization as the first
      Widrow-Hoff slide, but now we apply the **perceptron** rule:
      $\mathbf{w}(k+1) = \mathbf{w}(k) + \rho\, y_k\, \mathbf{x}_k$
      only when $\text{sign}(\mathbf{w}^T\mathbf{x}_k) \neq y_k$.
    - Click **Next update** to step through four perceptron updates
      and compare the trajectory to the Widrow-Hoff slide — the
      perceptron leaves the weights alone whenever a point is already
      correctly classified, so the boundary moves in larger, sparser
      jumps.
        """
    )
    return


@app.cell
def _(X_aug_wh, X_wh, n_wh, np, y_wh):
    # Same data and the same random initialization as the first
    # Widrow-Hoff slide; only the update rule differs. Same
    # class 1 / class 2 alternation so each step targets a sample
    # we can highlight.
    rng_p = np.random.default_rng(42)
    w_p_init = rng_p.normal(0, 0.5, size=3)
    rho_p = 0.05

    n_steps_p = 4
    trace_p = [w_p_init.copy()]
    update_idx_p = []
    for _step in range(n_steps_p):
        # Same alternating index pattern as the Widrow-Hoff runs.
        _idx = (_step // 2) + (_step % 2) * n_wh
        _x_i = X_aug_wh[_idx]
        _y_i = y_wh[_idx]
        _w_prev = trace_p[-1]
        _y_pred = np.sign(np.dot(_w_prev, _x_i))
        # Perceptron rule: update only when misclassified.
        if _y_pred != _y_i:
            _w_new = _w_prev + rho_p * _y_i * _x_i
        else:
            _w_new = _w_prev
        trace_p.append(_w_new)
        update_idx_p.append(_idx)

    return rho_p, trace_p, update_idx_p, n_steps_p


@app.cell
def _(mo, n_steps_p):
    # One radio button per time step so the user can jump forward and
    # backward through the trajectory.
    step_btn_p = mo.ui.radio(
        options={f"Step {k}": k for k in range(n_steps_p + 1)},
        value=0,
        label=f"Step (0..{n_steps_p})",
    )
    return (step_btn_p,)


@app.cell
def _(
    X_wh,
    mo,
    n_steps_p,
    np,
    plt,
    rho_p,
    step_btn_p,
    trace_p,
    update_idx_p,
    x1_wh,
    x2_wh,
):
    _step_p = int(step_btn_p.value) if step_btn_p.value is not None else 0
    _w_p = trace_p[_step_p]
    _idx_cur_p = update_idx_p[_step_p - 1] if _step_p >= 1 else update_idx_p[0]

    fig_p, ax_p = plt.subplots(figsize=(5, 5))
    ax_p.scatter(
        x1_wh[:, 0], x1_wh[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0,
        label="Class 1 (+1)",
    )
    ax_p.scatter(
        x2_wh[:, 0], x2_wh[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0,
        label="Class 2 (−1)",
    )
    # Green ring on the sample being processed at this step.
    _x_h_p = X_wh[_idx_cur_p]
    ax_p.scatter(
        [_x_h_p[0]], [_x_h_p[1]], s=450, facecolors="none",
        edgecolors="#2ca02c", linewidth=3.5, label="Current sample",
    )
    if abs(_w_p[1]) > 1e-9:
        # w0*x + w1*y + b = 0  ->  y = -(w0*x + b) / w1
        _xs_p = np.array([0.0, 3.0])
        _ys_p = -(_w_p[0] * _xs_p + _w_p[2]) / _w_p[1]
        ax_p.plot(
            _xs_p, _ys_p, "r--", linewidth=2.5,
            label=f"Boundary  w=[{_w_p[0]:+.2f}, {_w_p[1]:+.2f}, {_w_p[2]:+.2f}]",
        )
    ax_p.set_xlim(0.0, 3.0)
    ax_p.set_ylim(0.0, 3.0)
    ax_p.set_xlabel("x1")
    ax_p.set_ylabel("x2")
    ax_p.set_title(f"Perceptron — step {_step_p}/{n_steps_p}")
    ax_p.legend(loc="upper right", fontsize=8)
    ax_p.grid(True, alpha=0.3)
    plt.close(fig_p)

    mo.vstack([
        step_btn_p,
        mo.md(
            f"**Step {_step_p} / {n_steps_p}** &nbsp;—&nbsp; "
            f"weights: `{_w_p.round(3)}`"
        ),
        mo.as_html(fig_p),
    ])
    return


if __name__ == "__main__":
    app.run()
