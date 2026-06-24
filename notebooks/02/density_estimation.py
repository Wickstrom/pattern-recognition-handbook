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
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Parametric density estimation
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
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.stats import norm, uniform, beta

    # Seed once so the same samples back every figure in this notebook —
    # otherwise the slides re-roll data on each render and the figures
    # change between class sessions.
    np.random.seed(0)
    return beta, norm, np, plt, uniform


@app.cell
def _(beta, mo, norm, np, plt, uniform):
    # Overview: a Normal, Uniform, and Beta PDF side-by-side, sampled from
    # each distribution to show the shape the PDF is meant to describe.
    normal_data_overview = np.random.normal(loc=0, scale=1, size=1000)
    uniform_data_overview = np.random.uniform(low=-1, high=1, size=1000)
    beta_data_overview = np.random.beta(a=2, b=5, size=1000)

    x_normal_overview = np.linspace(normal_data_overview.min() - 0.1, normal_data_overview.max() + 0.1, 100)
    p_normal_overview = norm.pdf(x_normal_overview, loc=0, scale=1)

    x_uniform_overview = np.linspace(uniform_data_overview.min() - 0.1, uniform_data_overview.max() + 0.1, 100)
    p_uniform_overview = uniform.pdf(x_uniform_overview, loc=-1, scale=2)

    x_beta_overview = np.linspace(beta_data_overview.min() - 0.1, uniform_data_overview.max() + 0.1, 100)
    p_beta_overview = beta.pdf(x_beta_overview, a=2, b=5)

    fig_dists, axes_dists = plt.subplots(1, 3, figsize=(12, 3))
    axes_dists[0].plot(x_normal_overview, p_normal_overview, "k", linewidth=2, label="PDF")
    axes_dists[0].set_title("Normal distribution")
    axes_dists[1].plot(x_uniform_overview, p_uniform_overview, "k", linewidth=2, label="PDF")
    axes_dists[1].set_title("Uniform distribution")
    axes_dists[2].plot(x_beta_overview, p_beta_overview, "k", linewidth=2, label="PDF")
    axes_dists[2].set_title("Beta distribution")
    fig_dists.tight_layout()

    mo.as_html(fig_dists)
    plt.close(fig_dists)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - Recall:
        - Obtain *realizations* from probability density function:
            - $p(\mathbf{x}_1, \boldsymbol{\theta}) \rightarrow \mathbf{x}_1$
            - $p(\mathbf{x}_5, \boldsymbol{\theta}) \rightarrow \mathbf{x}_5$
        - Joint:
            - $p(\mathbf{x}_1, \mathbf{x}_5, \boldsymbol{\theta}) \rightarrow \mathbf{x}_1$ and $\mathbf{x}_5$
        - If independent:
            - $p(\mathbf{x}_1, \mathbf{x}_5, \boldsymbol{\theta}) = p(\mathbf{x}_1, \boldsymbol{\theta})p(\mathbf{x}_5, \boldsymbol{\theta})$
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
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    # Likelihood intuition: same data overlaid with a "good" fit (matching
    # parameters) and a "bad" fit (mean offset by 2). The good fit puts
    # most of its mass where the samples are.
    normal_data_lik = np.random.normal(loc=0, scale=1, size=50)

    x_normal_good = np.linspace(normal_data_lik.min() - 0.1, normal_data_lik.max() + 0.1, 100)
    p_normal_good = norm.pdf(x_normal_good, loc=0, scale=1)

    x_normal_bad = np.linspace(normal_data_lik.min() - 0.1, normal_data_lik.max() + 3.0, 100)
    p_normal_bad = norm.pdf(x_normal_bad, loc=2, scale=1)

    fig_lik, axes_lik = plt.subplots(1, 2, figsize=(6, 3))
    axes_lik[0].scatter(normal_data_lik, np.zeros_like(normal_data_lik), alpha=0.5)
    axes_lik[0].plot(x_normal_good, p_normal_good, "k", linewidth=2, label="PDF")
    axes_lik[1].scatter(normal_data_lik, np.zeros_like(normal_data_lik), alpha=0.5)
    axes_lik[1].plot(x_normal_bad, p_normal_bad, "k", linewidth=2, label="PDF")
    fig_lik.tight_layout()

    mo.as_html(fig_lik)
    plt.close(fig_lik)
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
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Non-parametric density estimation
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
    - Consider the example below
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
    X_1_name = "perimeter"
    X_2_name = "area"
    return X_1_name, X_2_name, X_bc


@app.cell
def _(X_1_name, X_2_name, X_bc, mo, plt):
    fig_bc, ax_bc = plt.subplots(figsize=(4, 4))
    ax_bc.scatter(X_bc[:, 0], X_bc[:, 1])
    ax_bc.set_xlabel(X_1_name)
    ax_bc.set_ylabel(X_2_name)

    mo.as_html(fig_bc)
    plt.close(fig_bc)
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
            - $P \approx k_N / N$, where $N$=total number of samples and $k_N$ = number of samples within bin.
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

    fig_pdf, ax_pdf = plt.subplots(figsize=(8, 4))
    ax_pdf.plot(x_normal_pdf, p_normal_pdf, "k", linewidth=2, label="PDF")
    ax_pdf.axis("off")

    mo.as_html(fig_pdf)
    plt.close(fig_pdf)
    return


@app.cell
def _(mo, norm, np, plt):
    normal_data_samples = np.random.normal(loc=0, scale=1, size=50)
    normal_data_example_samples = np.random.normal(loc=0, scale=1, size=9)
    x_normal_samples = np.linspace(normal_data_samples.min() - 0.1, normal_data_samples.max() + 0.1, 100)
    p_normal_samples = norm.pdf(x_normal_samples, loc=0, scale=1)

    fig_samples, ax_samples = plt.subplots(figsize=(8, 4))
    ax_samples.plot(x_normal_samples, p_normal_samples, "k", linewidth=2, label="PDF")
    ax_samples.scatter(normal_data_example_samples, np.zeros_like(normal_data_example_samples), alpha=0.5)
    ax_samples.grid(True)

    mo.as_html(fig_samples)
    plt.close(fig_samples)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Alternative histogram approach

    - Place bin on each sample.
    - Count samples within bin.
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    normal_data_alt = np.random.normal(loc=0, scale=1, size=50)
    normal_data_example_alt = np.random.normal(loc=0, scale=1, size=9)
    x_normal_alt = np.linspace(normal_data_alt.min() - 0.1, normal_data_alt.max() + 0.1, 100)
    p_normal_alt = norm.pdf(x_normal_alt, loc=0, scale=1)

    fig_alt, ax_alt = plt.subplots(figsize=(8, 4))
    ax_alt.plot(x_normal_alt, p_normal_alt, "k", linewidth=2, label="PDF")
    ax_alt.scatter(normal_data_example_alt, np.zeros_like(normal_data_example_alt), alpha=0.5)
    ax_alt.grid(True)

    mo.as_html(fig_alt)
    plt.close(fig_alt)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Mathematical formulation of alternative approach

    - Define
        - $I(y) = 1 \text{ if } |y| \leq 1/2$
    - Thus:
        - $I(\frac{x_i-x}{h}) = 1 \text{ if } \frac{|x_i-x|}{h} \leq 1/2$.
    - Therefore:
        - $p(x) \simeq \frac{1}{N h} \sum_{i=1}^N I(\frac{x_i-x}{h})$
    - Problem ->
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
        """
    )
    return


@app.cell
def _(mo, norm, np, plt):
    normal_data_parzen = np.random.normal(loc=0, scale=1, size=50)
    normal_data_example_parzen = np.random.normal(loc=0, scale=1, size=9)
    x_normal_parzen = np.linspace(normal_data_parzen.min() - 0.1, normal_data_parzen.max() + 0.1, 100)
    p_normal_parzen = norm.pdf(x_normal_parzen, loc=0, scale=1)

    fig_parzen, ax_parzen = plt.subplots(figsize=(8, 4))
    ax_parzen.plot(x_normal_parzen, p_normal_parzen, "k", linewidth=2, label="PDF")
    ax_parzen.scatter(normal_data_example_parzen, np.zeros_like(normal_data_example_parzen), alpha=0.5)
    ax_parzen.grid(True)

    mo.as_html(fig_parzen)
    plt.close(fig_parzen)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What to expect from Parzen windows?

    - $E[\hat{p}(x)]=\frac{1}{Nh}\sum_{i=1}^{N}E\left[\phi\left(\frac{x_i-x}{h}\right)\right]$
    - Have
        - $E[g(y)]=\int g(y)p(y)dy$
    - $E[\hat{p}(x)] = $
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What to expect from Parzen windows?

    - Note:
        - $\lim_{h \to 0} \frac{1}{h}\phi\left(\frac{x'-x}{h}\right) = \delta(x'-x)$
    - Thus:
        - $E[\hat{p}(x)]=p(x)$
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

    fig_mix, ax_mix = plt.subplots(figsize=(8, 3))
    ax_mix.plot(x_dist_1, p_dist_1, "k", linewidth=2, label="PDF")
    ax_mix.plot(x_dist_2, p_dist_2, "k", linewidth=2, label="PDF")
    ax_mix.scatter(data_dist_1, np.zeros_like(data_dist_1) - 0.1, alpha=0.5)
    ax_mix.scatter(data_dist_2, np.zeros_like(data_dist_2) - 0.1, alpha=0.5)

    mo.as_html(fig_mix)
    plt.close(fig_mix)
    return data_dist_1, data_dist_2


@app.cell
def _():
    from sklearn.neighbors import KernelDensity
    return (KernelDensity,)


@app.cell
def _(KernelDensity, data_dist_1, data_dist_2, mo, np, plt):
    combined = np.concatenate([data_dist_1, data_dist_2])[:, np.newaxis]
    kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(combined)

    x_approx = np.linspace(combined.min() - 0.1, combined.max() + 0.1, 500)[:, np.newaxis]
    p_approx = np.exp(kde.score_samples(x_approx))

    fig_kde, ax_kde = plt.subplots(figsize=(8, 4))
    ax_kde.plot(x_approx, p_approx, "k", linewidth=2, label="PDF")
    ax_kde.scatter(combined, np.zeros_like(combined), alpha=0.5)
    ax_kde.grid(True)

    mo.as_html(fig_kde)
    plt.close(fig_kde)
    return


if __name__ == "__main__":
    app.run()
