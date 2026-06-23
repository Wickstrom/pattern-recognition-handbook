# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-learn",
#     "ucimlrepo",
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
    mo.md(r"""---""")
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
    - Real-world data rarely follow exact distributions.
    - So what do we do when we cannot find a parametric distribution that matches or data?
    - Consider the example below
        """
    )
    return


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo
    return (fetch_ucirepo,)


@app.cell
def _(fetch_ucirepo):
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    X_bc = breast_cancer_wisconsin_diagnostic.data.features
    X_1_name = "perimeter"
    X_2_name = "area"
    return X_1_name, X_2_name, X_bc


@app.cell
def _(X_1_name, X_2_name, X_bc, mo, plt):
    fig_bc, ax_bc = plt.subplots(figsize=(4, 4))
    ax_bc.scatter(X_bc.iloc[:, 4], X_bc.iloc[:, 5])
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

    - $E[\hat{p}(x)]=\frac{1}{Nh}\sum_{i=1^N}E[\phi(\frac{x_i-x}{h})]$
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
        - $\lim_{h \to 0} \frac{1}{h}\phi(\frac{x'-x}{h}) = \delta(x'-x)$
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
    data = np.concatenate([data_dist_1, data_dist_2])[:, np.newaxis]
    kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(data)

    x_approx = np.linspace(data.min() - 0.1, data.max() + 0.1, 500)[:, np.newaxis]
    p_approx = np.exp(kde.score_samples(x_approx))

    fig_kde, ax_kde = plt.subplots(figsize=(8, 4))
    ax_kde.plot(x_approx, p_approx, "k", linewidth=2, label="PDF")
    ax_kde.scatter(data, np.zeros_like(data), alpha=0.5)
    ax_kde.grid(True)

    mo.as_html(fig_kde)
    plt.close(fig_kde)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Programming exercises

    Below are programming exercises assocaited with this lecture. These cell blocks are starting points that loads the data and prepares the problem such that you can get going with the implementation. There are also theoretical exercsies, but due to copyright we cannot shared them here. They will be made available in a private repository connected to the course.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Problem 2.36 from the book

    - The cell below generates data from a uniform distribution. Use the Parzen
      window method to estimate the probability density function. Consider the
      following settings:
        - Varying number of samples (32, 256, 5000).
        - Use Gaussian kernel for the Parzen window with smoothing parameters
          of 0.05 and 0.2.
        """
    )
    return


@app.cell
def _(mo, np, plt, uniform):
    # Problem 2.36 starter: uniform samples on [0, 2] with the true PDF
    # overlaid. Students vary the sample count and the kernel bandwidth.
    number_of_samples = 32
    uniform_data_p36 = np.random.uniform(low=0, high=2, size=number_of_samples)

    x_uniform_p36 = np.linspace(-1, 3, 100)
    p_uniform_p36 = uniform.pdf(x_uniform_p36, loc=0, scale=2)

    fig_uni, ax_uni = plt.subplots(figsize=(8, 4))
    ax_uni.plot(x_uniform_p36, p_uniform_p36, "k", linewidth=2, label="PDF")
    ax_uni.set_xlabel("Value")
    ax_uni.set_title("Uniform distribution")
    ax_uni.legend()
    ax_uni.grid(True)

    mo.as_html(fig_uni)
    plt.close(fig_uni)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Problem 2.12 from the book

    - Below is code that generates the data associated with problem 2.12
      from the book. Your task is to:
        - Design a Bayesian classifier.
        - Design a Bayesian classifier using the following risk parameters.
          How does this change the decision boundary?
            - $\lambda_{12}$: 1.0
            - $\lambda_{21}$: 0.5
        - Experiment with changing the mean and variance of each class.
          What do you observe?
    - However, this time you will estimate the probability density functions.
      Use both a parametric and non-parametric approach. How do they compare
      to Bayes classifier?
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Problem 2.12 starter: two well-overlapping Gaussians with the same
    # covariance. The Bayes-optimal boundary is the perpendicular
    # bisector of the line between the means, but density-estimation
    # classifiers won't recover that exactly — that's the point of the
    # exercise.
    number_of_samples_in_each_class = 100

    mu1 = np.array([1, 1])
    mu2 = np.array([1.5, 1.5])
    sigma = np.array([[0.2, 0.0], [0.0, 0.2]])

    x_train_1 = np.random.multivariate_normal(mu1, sigma, number_of_samples_in_each_class)
    x_train_2 = np.random.multivariate_normal(mu2, sigma, number_of_samples_in_each_class)
    x_train = np.concatenate((x_train_1, x_train_2))
    y_train = np.concatenate(
        (np.ones(number_of_samples_in_each_class), np.zeros(number_of_samples_in_each_class))
    )

    fig_p212, ax_p212 = plt.subplots(figsize=(8, 8))
    ax_p212.scatter(
        x_train_1[:, 0], x_train_1[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_p212.scatter(
        x_train_2[:, 0], x_train_2[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2",
    )
    ax_p212.set_xlabel("x1")
    ax_p212.set_ylabel("x2")
    ax_p212.legend()

    mo.as_html(fig_p212)
    plt.close(fig_p212)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Classification using Bayes classifiers and density estimation

    You will now get your first experince working with real-world data. Below is a code block that loads the Iris dataset, one of the earliest known datasets used for evaluating classification methods and a classic introductory dataset in pattern recognition and machine learning. [See here for more information about the dataset](https://archive.ics.uci.edu/dataset/53/iris).

    The data is 4-dimensional with 3 classes. Here, we have simplifed the problem a bit and extracted 2 features such that the dataset can be easily visualized. Use the density estimation approach you deem the most suitable and combine it with a Bayesian classifier to tackle the problem of Iris plant classification.
        """
    )
    return


@app.cell
def _(fetch_ucirepo, mo, np, plt):
    # Iris starter: 2 of the 4 features (sepal length, sepal width),
    # all 3 classes. Students pick a density estimator and combine it
    # with a Bayes classifier to draw a decision boundary.
    iris = fetch_ucirepo(id=53)

    X_iris = iris.data.features.iloc[:, :2]
    feature_1_name = "sepal length (cm)"
    feature_2_name = "sepal width (cm)"
    y_iris = np.zeros(150)
    y_iris[50:100] = 1
    y_iris[100:150] = 2
    y_names = np.unique(iris.data.targets)

    fig_iris, ax_iris = plt.subplots(figsize=(6, 6))
    for class_i, class_name in enumerate(y_names):
        ax_iris.scatter(
            X_iris.iloc[np.where(y_iris == class_i)[0], 0],
            X_iris.iloc[np.where(y_iris == class_i)[0], 1],
            label=class_name,
        )
    ax_iris.set_xlabel(feature_1_name)
    ax_iris.set_ylabel(feature_2_name)
    ax_iris.legend()

    mo.as_html(fig_iris)
    plt.close(fig_iris)
    return


if __name__ == "__main__":
    app.run()
