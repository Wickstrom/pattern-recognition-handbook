# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-learn",
# ]
# ///
#
# Marimo version of the density estimation lecture.
# Same content as notebooks/02/density_estimation.ipynb, but authored as
# a reactive Marimo app so each matplotlib figure stays tied to the data
# generation that produced it (one slide per concept, matching the
# Bayes_decision_theory notebook's structure).
# Run locally with `marimo edit notebooks/02/density_estimation.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `normal_data`) can be
# redefined in any number of cells, but Marimo requires each name to be
# owned by exactly one cell. Below, every cell-local figure is given a
# unique name (`normal_data_overview`, `normal_data_likelihood`, …) so the
# slides stay independent and the data each figure displays is
# reproducible.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/density_estimation.slides.json",
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
def _(mo):
    mo.md(
        r"""
    # Density estimation

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Last week we looked a Bayes decision theory
    - Nice theoretically, but assumes knowledge about probablity density distribution.
    - This is rarely available, so how do we proceed?
    - Estimate densities from data!

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Overview

    - In this lecture we will cover:
        - Parametric density estimation
            - Maximum likelihood
        - Non-parametric density estimation
            - Histogram approach
            - Parzen window approach

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Parametric density estimation

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What do we mean by parametric?

    - Assume that the *form* of the probablity density function is known.
    - Estimate *parameters* from data

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.stats import norm, uniform, beta, gamma

    # Seed once so the same samples back every figure in this notebook —
    # otherwise the slides re-roll data on each render and the figures
    # change between class sessions.
    np.random.seed(0)
    mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 37</div>""")
    return beta, gamma, norm, np, plt, uniform


@app.cell
def _(mo, np, plt):
    # Pre-compute one histogram per distribution (same pattern as the
    # parametric estimation example below). We deliberately do *not*
    # overlay the true PDF — that would give the answer away. The
    # distributions are, in order of difficulty:
    #   1. Normal(0, 1)     — the textbook bell curve
    #   2. Uniform(-2, 2)   — easy: it's flat
    #   3. Gamma(2, 2)      — skewed unimodal, positive support
    #   4. Exponential(1)   — one-sided decay; not symmetric, no lower tail
    #   5. Beta(0.5, 0.5)   — U-shape on [0, 1]; tempting to call uniform
    rng_q = np.random.default_rng(2)
    n_samples = 1000

    dists = [
        ("Distribution 1", "Normal(0, 1)",      lambda: rng_q.normal(0, 1, n_samples)),
        ("Distribution 2", "Uniform(-2, 2)",    lambda: rng_q.uniform(-2, 2, n_samples)),
        ("Distribution 3", "Gamma(2, 2)",       lambda: rng_q.gamma(2, 2, n_samples)),
        ("Distribution 4", "Exponential(1)",    lambda: rng_q.exponential(1, n_samples)),
        ("Distribution 5", "Beta(0.5, 0.5)",    lambda: rng_q.beta(0.5, 0.5, n_samples)),
    ]

    figs_dist = {}
    for _tab_label_q, _dist_name_q, _sampler in dists:
        _samples = _sampler()
        _fig_dist, _ax_dist = plt.subplots(figsize=(10, 5))
        _ax_dist.hist(
            _samples, bins=40, density=True,
            color="#c44e52", alpha=0.75, edgecolor="black",
        )
        _ax_dist.set_xlabel("x")
        _ax_dist.set_ylabel("density")
        _ax_dist.set_title("What distribution is this?")
        _fig_dist.tight_layout()
        figs_dist[_tab_label_q] = _fig_dist
        plt.close(_fig_dist)

    mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 37</div>""")
    return dists, figs_dist


@app.cell
def _(mo):
    # Lives in its own cell so the tabs cell below can read
    # show_name_q.value without violating Marimo's "no reading a
    # UIElement in the cell that created it" rule.
    show_name_q = mo.ui.switch(value=False, label="Show distribution names")
    mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 37</div>""")
    return (show_name_q,)


@app.cell
def _(dists, figs_dist, mo, show_name_q):
    # Bundle the pre-computed figures into tabs. The label above each
    # figure toggles between the bare "Distribution N" and the
    # "Distribution N: *Family*" reveal depending on the switch.
    tabs_dist = {}
    for _tab_label_q, _dist_name_q, _sampler in dists:
        _fig_dist = figs_dist[_tab_label_q]
        _label_md = (
            f"**{_tab_label_q}**: *{_dist_name_q}*"
            if show_name_q.value
            else f"**{_tab_label_q}**"
        )
        tabs_dist[_tab_label_q] = mo.vstack(
            [
                mo.md(_label_md),
                mo.as_html(_fig_dist),
            ],
            gap=1,
        )

    mo.vstack(
        [
            mo.vstack([show_name_q, mo.ui.tabs(tabs_dist)]),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Recall

    - Obtain *realizations* from probability density function:
        - $p(\mathbf{x}_1, \boldsymbol{\theta}) \rightarrow \mathbf{x}_1$
        - $p(\mathbf{x}_5, \boldsymbol{\theta}) \rightarrow \mathbf{x}_5$
    - Joint:
        - $p(\mathbf{x}_1, \mathbf{x}_5, \boldsymbol{\theta}) \rightarrow \mathbf{x}_1$ and $\mathbf{x}_5$
    - If independent:
        - $p(\mathbf{x}_1, \mathbf{x}_5, \boldsymbol{\theta}) = p(\mathbf{x}_1, \boldsymbol{\theta})p(\mathbf{x}_5, \boldsymbol{\theta})$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Likelihood function

    - Let $\{\mathbf{x}_1, \cdots, \mathbf{x}_N\}$ and assume independent and identically distributed random variables (i.i.d.).
    - Let the set of parameters for a particular distribution form a parameter vector $\boldsymbol{\theta}$.
    - The likelihood function of $\boldsymbol{\theta}$ with respect to the random variable $X$ can be expressed as:
        - $p(X; \boldsymbol{\theta})=\prod\limits_{i=1}^N p(\mathbf{x}_i; \boldsymbol{\theta})$
    - Idea:
        - Find $\boldsymbol{\theta}$ that makes $X$ most probable.
    - Therefore:
        - $\hat{\boldsymbol{\theta}}_{ML} = \underset{\boldsymbol{\theta}}{\operatorname{argmax}}\ p(X; \boldsymbol{\theta})$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Log-likelihood

    - In most cases, we work with the log-likelihood instead of the likelihood itself.
        - Simplifies the mathematics.
        - Can be more numerically stable if we have the product of many small probabilities.
    - Log-likelihood:
        - $L(\boldsymbol{\theta}) = \log \prod\limits_{i=1}^N p(\mathbf{x}_i; \boldsymbol{\theta})$
        - Remember: $\log(ab) = \log(a)+\log(b)$
    - Want:
        - $\frac{\partial}{\partial \boldsymbol{\theta}} L(\boldsymbol{\theta}) = \mathbf{0}$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with univariate normal

    - Let us consider the case of a univeriate normal distribution
    - Parameters:
        - $\boldsymbol{\theta} = (\theta_{1}, \theta_{2})^T = (\mu, \sigma^2)^T$.
    - First: log-likelihood
        - $\log \prod\limits_{i=1}^N p(\mathbf{x}_i; \boldsymbol{\theta}) =$
    - Then: derivative
        - $\frac{\partial}{\partial \boldsymbol{\theta}} \log p(\mathbf{x}_i; \boldsymbol{\theta}) =$
    - Lastly: setting to zero
        - $\sum_{k=1}^N = \mathbf{0}$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with univariate normal continued

    - Putting it all together gives us:
        - For $\theta_1$:
            - $\hat{\theta}_1 = \hat{\mu} = \frac{1}{N}\sum\limits_{k=1}^N x_k$
        - For $\theta_2$:
            - $\hat{\theta}_2 = \hat{\sigma}^2 = \frac{1}{N}\sum\limits_{k=1}^N (x_k-\mu)^2$
    - Note: In many cases, we will use these estimators for mean and covariances
      regardless.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 37</div>
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    # Interactive demo for the maximum-likelihood derivation above.
    # Pre-compute one figure per (μ, σ) preset and bundle into a single
    # tabs widget so the controls and the visualization live in one cell
    # (one slide in slides view — same pattern as the Bayes lecture at
    # notebooks/01/bayes_decision_theory.py and the linear classifier
    # tabs at notebooks/03/linear_classifiers.py). Students click a preset
    # to swap the Gaussian overlay; the data is fixed across all presets
    # so the change in fit is what they see.
    data_param = np.array(
        [-1.2, -0.8, -0.3, 0.1, 0.2, 0.5, 0.9, 1.4, 1.8, 2.1]
    )
    presets_param = {
        "μ=0.0, σ=0.8": (0.0, 0.8),
        "μ=0.5, σ=0.8": (0.5, 0.8),
        "μ=0.9, σ=0.8": (0.9, 0.8),
        "μ=0.9, σ=0.4": (0.9, 0.4),
        "μ=0.9, σ=1.5": (0.9, 1.5),
    }

    tabs_param = {}
    for label, (mu_p, sigma_p) in presets_param.items():
        log_lik_p = float(
            np.sum(norm.logpdf(data_param, mu_p, sigma_p))
        )
        x_grid_p = np.linspace(
            data_param.min() - 1, data_param.max() + 1, 200
        )
        pdf_p = norm.pdf(x_grid_p, mu_p, sigma_p)

        fig_p, ax_p = plt.subplots(figsize=(10, 5))
        ax_p.hist(
            data_param, bins=8, density=True, alpha=0.4,
            color="gray", edgecolor="black", label="data",
        )
        ax_p.plot(
            x_grid_p, pdf_p, "r-", linewidth=2,
            label="candidate density",
        )
        ax_p.set_xlabel("x")
        ax_p.set_ylabel("density")
        ax_p.legend(loc="upper right")
        ax_p.set_title(
            f"log-likelihood = {log_lik_p:.3f}  —  {label}"
        )
        fig_p.tight_layout()

        tabs_param[label] = mo.vstack(
            [
                mo.md(
                    f"**Setting:** {label} &nbsp;&nbsp; "
                    f"**log-likelihood:** {log_lik_p:.3f}"
                ),
                mo.as_html(fig_p),
            ],
            gap=1,
        )
        plt.close(fig_p)

    mo.vstack(
        [
            mo.ui.tabs(tabs_param),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Non-parametric density estimation

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Overview

    - Real-world data rarely follow exact distributions.
    - So what do we do when we cannot find a parametric distribution that matches or data?
    - Consider the following example

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 37</div>
        """
    )
    return


@app.cell
def _():
    import base64
    import io

    # Load the breast-cancer features we visualize from data embedded
    # directly in the notebook. Fetching the dataset from UCI at runtime
    # intermittently fails in the WASM/Pyodide environment, and shipping
    # a sidecar file like `media/breast_cancer_subset.npz` does not work
    # either — Pyodide's virtual filesystem does not have access to files
    # placed next to the HTML page, so `np.load(...)` raises
    # FileNotFoundError and `X_bc` ends up undefined.
    _data_bytes = base64.b64decode(
        "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"
    )
    _loaded = np.load(io.BytesIO(_data_bytes))
    X_bc = _loaded["features"]
    y_bc = _loaded["labels"]
    X_1_name = "perimeter"
    X_2_name = "area"
    mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 37</div>""")
    return X_1_name, X_2_name, X_bc, y_bc


@app.cell
def _(X_1_name, X_2_name, X_bc, mo, plt, y_bc):
    fig_bc, ax_bc = plt.subplots(figsize=(6.5, 6.5))
    scatter_bc = ax_bc.scatter(
        X_bc[:, 0], X_bc[:, 1], c=y_bc, cmap="coolwarm",
        edgecolor="black", alpha=0.7,
    )
    ax_bc.set_xlabel(X_1_name)
    ax_bc.set_ylabel(X_2_name)
    legend_bc = ax_bc.legend(
        *scatter_bc.legend_elements(),
        title="Class",
        loc="upper right",
    )

    plt.close(fig_bc)
    mo.vstack(
        [
            mo.as_html(fig_bc),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Histogram approach

    - Idea:
        - Divide input into bins.
        - For each bin; estimate probablity of a sample x being in a bin.
        - This can probability can be estimated using the *frequency ratio*
            - $P \approx k_N / N$, where $N$ is the total number of samples and $k_N$ is the number of samples within the bin.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 37</div>
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    # First histogram framing: just the true PDF, no samples. The next
    # cell layers a few samples onto the same curve to set up the
    # "where do these dots fit under the curve?" question.
    normal_data_pdf = np.random.normal(loc=0, scale=1, size=50)
    x_normal_pdf = np.linspace(normal_data_pdf.min() - 0.1, normal_data_pdf.max() + 0.1, 100)
    p_normal_pdf = norm.pdf(x_normal_pdf, loc=0, scale=1)

    fig_pdf, ax_pdf = plt.subplots(figsize=(10, 5))
    ax_pdf.plot(x_normal_pdf, p_normal_pdf, "k", linewidth=2, label="PDF")
    ax_pdf.axis("off")

    plt.close(fig_pdf)
    mo.vstack(
        [
            mo.as_html(fig_pdf),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo, norm, np, plt):
    normal_data_samples = np.random.normal(loc=0, scale=1, size=50)
    normal_data_example_samples = np.random.normal(loc=0, scale=1, size=9)
    x_normal_samples = np.linspace(normal_data_samples.min() - 0.1, normal_data_samples.max() + 0.1, 100)
    p_normal_samples = norm.pdf(x_normal_samples, loc=0, scale=1)

    fig_samples, ax_samples = plt.subplots(figsize=(10, 5))
    ax_samples.plot(x_normal_samples, p_normal_samples, "k", linewidth=2, label="PDF")
    ax_samples.scatter(normal_data_example_samples, np.zeros_like(normal_data_example_samples), alpha=0.5)
    ax_samples.grid(True)

    plt.close(fig_samples)
    mo.vstack(
        [
            mo.as_html(fig_samples),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Alternative histogram approach

    - Place bin on each sample.
    - Count samples within bin.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 37</div>
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    normal_data_alt = np.random.normal(loc=0, scale=1, size=50)
    normal_data_example_alt = np.random.normal(loc=0, scale=1, size=9)
    x_normal_alt = np.linspace(normal_data_alt.min() - 0.1, normal_data_alt.max() + 0.1, 100)
    p_normal_alt = norm.pdf(x_normal_alt, loc=0, scale=1)

    fig_alt, ax_alt = plt.subplots(figsize=(10, 5))
    ax_alt.plot(x_normal_alt, p_normal_alt, "k", linewidth=2, label="PDF")
    ax_alt.scatter(normal_data_example_alt, np.zeros_like(normal_data_example_alt), alpha=0.5)
    ax_alt.grid(True)

    plt.close(fig_alt)
    mo.vstack(
        [
            mo.as_html(fig_alt),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">24 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Mathematical formulation of alternative approach

    - Define
        - $I(y) = \begin{cases} 1 & \text{if } |y| \leq 1/2 \\ 0 & \text{otherwise} \end{cases}$
    - Thus:
        - $I(\frac{x_i-x}{h}) = 1 \text{ if } \frac{|x_i-x|}{h} \leq 1/2$.
    - Therefore:
        - $p(x) \simeq \frac{1}{N h} \sum_{i=1}^N I(\frac{x_i-x}{h})$
    - Problem ->

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">25 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Parzen windows (Parzen, 1962)

    - Take the alternative histogram approach as a starting point.
    - Replace $I$ with a *smooth* function $\phi$ subject to:
        - $\phi(y) \geq 0$ and $\int \phi(y)dy=1$
    - This ensures an esimate that is a valid probablity density function.
    - A typical choice is $N(0, 1)$, which results in:

    $$p(x) = \frac{1}{Nh} \sum_{i=1}^N \phi\!\left(\frac{x_i - x}{h}\right)$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">26 / 37</div>
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    normal_data_parzen = np.random.normal(loc=0, scale=1, size=50)
    normal_data_example_parzen = np.random.normal(loc=0, scale=1, size=9)
    x_normal_parzen = np.linspace(normal_data_parzen.min() - 0.1, normal_data_parzen.max() + 0.1, 100)
    p_normal_parzen = norm.pdf(x_normal_parzen, loc=0, scale=1)

    fig_parzen, ax_parzen = plt.subplots(figsize=(10, 5))
    ax_parzen.plot(x_normal_parzen, p_normal_parzen, "k", linewidth=2, label="PDF")
    ax_parzen.scatter(normal_data_example_parzen, np.zeros_like(normal_data_example_parzen), alpha=0.5)
    ax_parzen.grid(True)

    plt.close(fig_parzen)
    mo.vstack(
        [
            mo.as_html(fig_parzen),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">27 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What to expect from Parzen windows?

    - $E[\hat{p}(x)]=\frac{1}{Nh}\sum_{i=1}^{N}E\left[\phi\left(\frac{x_i-x}{h}\right)\right]$
    - Have
        - $E[g(y)]=\int g(y)p(y)dy$
    - $E[\hat{p}(x)] = \frac{1}{Nh}\sum_{i=1}^{N}\int \phi\left(\frac{x_i-x}{h}\right) p(x_i)\,dx_i$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">28 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What to expect from Parzen windows? — Convergence

    - Note:
        - $\lim_{h \to 0} \frac{1}{h}\phi\left(\frac{x'-x}{h}\right) = \delta(x'-x)$
    - Thus:
        - $E[\hat{p}(x)]=p(x)$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">29 / 37</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Parzen windows in practice

    - Assume we have a fixed dataset with $N$ samples.
    - Then:
        - Bias low for "small" $h$.
        - Variance low for "large" $h$.
    - Remarks:
        - Plug $\hat{p}(\mathbf{x}|w_i)$ into Bayes!
        - Need more data for higher dimensions to be accurate.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">30 / 37</div>
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    # Bimodal mixture: two well-separated Gaussians with their true PDFs
    # drawn on top of the (jittered) samples. The next cell fits a
    # Gaussian KDE to the same data to show how Parzen-style smoothing
    # reconstructs the shape.
    data_dist_1 = np.random.normal(loc=8, scale=2.5, size=100)
    data_dist_2 = np.random.normal(loc=0, scale=1.5, size=250)

    x_dist_1 = np.linspace(data_dist_1.min() - 0.1, data_dist_1.max() + 0.1, 100)
    p_dist_1 = norm.pdf(x_dist_1, loc=6, scale=2.5)

    x_dist_2 = np.linspace(data_dist_2.min() - 0.1, data_dist_2.max() + 0.1, 100)
    p_dist_2 = norm.pdf(x_dist_2, loc=0, scale=1.5)

    fig_mix, ax_mix = plt.subplots(figsize=(10, 5))
    ax_mix.plot(x_dist_1, p_dist_1, "k", linewidth=2, label="PDF")
    ax_mix.plot(x_dist_2, p_dist_2, "k", linewidth=2, label="PDF")
    ax_mix.scatter(data_dist_1, np.zeros_like(data_dist_1) - 0.1, alpha=0.5)
    ax_mix.scatter(data_dist_2, np.zeros_like(data_dist_2) - 0.1, alpha=0.5)

    plt.close(fig_mix)
    mo.vstack(
        [
            mo.as_html(fig_mix),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">31 / 37</div>"""),
        ]
    )
    return data_dist_1, data_dist_2


@app.cell
def _(mo):
    from sklearn.neighbors import KernelDensity
    mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">32 / 37</div>""")
    return (KernelDensity,)


@app.cell
def _(KernelDensity, data_dist_1, data_dist_2, mo, np, plt):
    combined = np.concatenate([data_dist_1, data_dist_2])[:, np.newaxis]
    kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(combined)

    x_approx = np.linspace(combined.min() - 0.1, combined.max() + 0.1, 500)[:, np.newaxis]
    p_approx = np.exp(kde.score_samples(x_approx))

    fig_kde, ax_kde = plt.subplots(figsize=(10, 5))
    ax_kde.plot(x_approx, p_approx, "k", linewidth=2, label="PDF")
    ax_kde.scatter(combined, np.zeros_like(combined), alpha=0.5)
    ax_kde.grid(True)

    plt.close(fig_kde)
    mo.vstack(
        [
            mo.as_html(fig_kde),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">33 / 37</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Parzen + Bayes: how does $h$ shape the decision boundary?

    - Now plug the Parzen estimate $\hat{p}(\mathbf{x}|w_i)$ into the Bayes rule:
        - Decide $w_1$ when $\hat{p}(\mathbf{x}|w_1) > \hat{p}(\mathbf{x}|w_2)$.
    - The two classes below form the classic **two-moons** shape — each
      is a curved arc with noise, clearly not a Gaussian blob. A
      *parametric* Gaussian fit per class would draw a near-straight
      decision line through the overlap and misclassify the moons;
      Parzen windows with the right bandwidth recover the curved
      boundary.
    - Click the tabs to see how the kernel width $h$ trades bias vs.
      variance in the boundary.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">34 / 37</div>
        """
    )
    return


@app.cell
def _(KernelDensity, np):
    # Two interleaving "moons" — the canonical non-Gaussian 2D example.
    # Each class is a noisy arc, so the true decision boundary is a
    # non-linear curve that no parametric Gaussian can capture.
    # Parzen-window estimation, on the other hand, can recover it as
    # long as the bandwidth is roughly the noise scale.
    rng_pb = np.random.default_rng(3)
    n_per_class = 200
    noise = 0.12

    theta = np.linspace(0, np.pi, n_per_class)
    # Outer moon: arc opening downward
    X0_pb = np.column_stack([np.cos(theta), np.sin(theta)])
    # Inner moon: arc opening upward, offset right and down
    X1_pb = np.column_stack([
        1 - np.cos(theta),
        -np.sin(theta) - 0.5,
    ])
    X0_pb += rng_pb.normal(0, noise, X0_pb.shape)
    X1_pb += rng_pb.normal(0, noise, X1_pb.shape)

    h_presets_pb = {
        "h = 0.01": 0.01,
        "h = 0.15": 0.15,
        "h = 0.50": 0.50,
        "h = 1.50": 1.50,
    }

    grid_min_pb, grid_max_pb, n_grid_pb = -2.0, 2.5, 160
    xx_pb, yy_pb = np.meshgrid(
        np.linspace(grid_min_pb, grid_max_pb, n_grid_pb),
        np.linspace(grid_min_pb, grid_max_pb, n_grid_pb),
    )
    grid_pts_pb = np.column_stack([xx_pb.ravel(), yy_pb.ravel()])

    # Pre-compute the log-density difference and binarized decision
    # for each bandwidth preset so the rendering cell below only has
    # to draw (and re-draw on switch toggle), not refit the KDEs.
    tab_data_pb = {}
    for _tab_label_pb, _h_val_pb in h_presets_pb.items():
        _kde0 = KernelDensity(kernel="gaussian", bandwidth=_h_val_pb).fit(X0_pb)
        _kde1 = KernelDensity(kernel="gaussian", bandwidth=_h_val_pb).fit(X1_pb)
        _log_diff = (
            _kde0.score_samples(grid_pts_pb)
            - _kde1.score_samples(grid_pts_pb)
        ).reshape(xx_pb.shape)
        _decision = (_log_diff > 0).astype(float)
        tab_data_pb[_tab_label_pb] = (_log_diff, _decision)

    mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">35 / 37</div>""")
    return X0_pb, X1_pb, grid_max_pb, grid_min_pb, h_presets_pb, tab_data_pb, xx_pb, yy_pb


@app.cell
def _(mo):
    # Lives in its own cell so the tabs cell below can read
    # show_boundary_pb.value without violating Marimo's "no reading
    # a UIElement in the cell that created it" rule.
    show_boundary_pb = mo.ui.switch(value=False, label="Show decision boundary")
    mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">36 / 37</div>""")
    return (show_boundary_pb,)


@app.cell
def _(
    X0_pb,
    X1_pb,
    grid_max_pb,
    grid_min_pb,
    h_presets_pb,
    mo,
    plt,
    show_boundary_pb,
    tab_data_pb,
    xx_pb,
    yy_pb,
):
    tabs_pb = {}
    for tab_label_pb, _h_val_pb in h_presets_pb.items():
        log_diff, decision = tab_data_pb[tab_label_pb]

        fig_pb, ax_pb = plt.subplots(figsize=(6.5, 6.5))
        if show_boundary_pb.value:
            ax_pb.contourf(
                xx_pb, yy_pb, decision,
                levels=[-0.5, 0.5, 1.5],
                colors=["#f4cccc", "#cfe2f3"], alpha=0.4,
            )
            ax_pb.contour(
                xx_pb, yy_pb, log_diff, levels=[0],
                colors="black", linewidths=2,
            )
        ax_pb.scatter(
            X0_pb[:, 0], X0_pb[:, 1], c="#c44e52", edgecolor="k",
            s=18, alpha=0.7, label="Class 0",
        )
        ax_pb.scatter(
            X1_pb[:, 0], X1_pb[:, 1], c="#4c72b0", edgecolor="k",
            s=18, alpha=0.7, label="Class 1",
        )
        ax_pb.set_xlim(grid_min_pb, grid_max_pb)
        ax_pb.set_ylim(grid_min_pb, grid_max_pb)
        ax_pb.set_xlabel("x₁")
        ax_pb.set_ylabel("x₂")
        ax_pb.set_aspect("equal")
        ax_pb.set_title(f"Parzen–Bayes decision boundary  ({tab_label_pb})")
        ax_pb.legend(loc="upper right")
        fig_pb.tight_layout()

        tabs_pb[tab_label_pb] = mo.vstack(
            [
                mo.md(f"**Bandwidth:** {tab_label_pb}"),
                mo.as_html(fig_pb),
            ],
            gap=1,
        )
        plt.close(fig_pb)

    mo.vstack(
        [
            mo.vstack([show_boundary_pb, mo.ui.tabs(tabs_pb)]),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">37 / 37</div>"""),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
