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
# Mixture modelling lecture — Gaussian mixture models with the
# Expectation-Maximisation (EM) algorithm.
#
# NOTE (under development): this lecture was carved out of the former
# "Density estimation II" notebook. It now ships as its own last lecture.
# The content below mirrors the original mixture-model slides; expect
# further polish as the course evolves.
#
# Run locally with `marimo edit notebooks/15/gaussian_mixture_models.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/gaussian_mixture_models.slides.json",
)


@app.cell
def _():
    import os
    from pathlib import Path

    # Make `mo.image(src="media/foo.png")` resolve regardless of which
    # directory `marimo edit` is launched from. We chdir to the notebook's
    # own directory so the relative `media/` path works the same way it
    # does for the deployed WASM build. In WASM, `__file__` may not be set
    # and the chdir becomes a no-op.
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
    # Gaussian mixture models

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    > **Under development.** This lecture was moved out of the density
    > estimation module and is now the final lecture of the course. The
    > notes below are a starting point and will be expanded.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Mixture models

    - Take a look at the following data:

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 20</div>
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Generate data samples from three normal distributions and scatter
    # them. Same data as the former "Density estimation II" notebook so
    # the two-D Gaussian blobs line up with the EM demo later.
    np.random.seed(42)
    rng_mix = np.random.default_rng(42)

    mean1_mix, cov1_mix = [2, 3], [[1.5, 0.5], [0.5, 1]]
    mean2_mix, cov2_mix = [6, 5], [[1, -0.3], [-0.3, 1.2]]
    mean3_mix, cov3_mix = [4, 1], [[0.8, 0.2], [0.2, 0.5]]

    data1_mix = rng_mix.multivariate_normal(mean1_mix, cov1_mix, 300)
    data2_mix = rng_mix.multivariate_normal(mean2_mix, cov2_mix, 300)
    data3_mix = rng_mix.multivariate_normal(mean3_mix, cov3_mix, 300)

    data_mix = np.vstack((data1_mix, data2_mix, data3_mix))

    fig_mix, ax_mix = plt.subplots(figsize=(7, 5))
    ax_mix.scatter(data_mix[:, 0], data_mix[:, 1])

    plt.close(fig_mix)
    mo.vstack(
        [
            mo.as_html(fig_mix),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 20</div>"""),
        ]
    )
    return data_mix


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How can we handle this "mix" of distributions?

    - Model the unknown density $p(\mathbf{x})$ via a linear combination of density functions:
    - $p(\mathbf{x})=\sum_{j=1}^{J}p(\mathbf{x}|j)P_j$
    - where
        - $\sum_{j=1}^{J}P_j=1$ and $\int p(\mathbf{x}|j) d\mathbf{x}=1$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What are the implications of this mix of models?

    - Assumes that each sample $\mathbf{x}$ may be "drawn" from any of the $J$ distributions.
    - What is the problem here?

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How to perform the modeling?

    - First step:
        - Assume a parametric form $p(\mathbf{x}|j;\boldsymbol{\theta})$
    - Need to find the unknown parameters $\theta$ and $P_j$ for $j=1,\cdots,J$.
    - What would be your first idea?

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How to perform the modeling?

    - First idea:
        - Maximize the likelihood function -> $\prod_k p(\mathbf{x}_k|j;\boldsymbol{\theta}, P_1, P_2, \cdots , P_J)$
    - Problem:
        - Unknown parameters enter the maximization task in a nonlinear fashion.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Expectation Maximisation (EM) - algorithm

    - The EM algorithm is one of the most widely used methods for local maximum likelihood estimate of parameters.
    - Ideally suited to handle incomplete data settings.
    - Let $\mathbf{y}\in Y \subseteq \mathcal{R}^m$ denote the complete data samples.
        - Probability density function $p_y(\mathbf{y};\boldsymbol{\theta})$, with unknown parameters $\boldsymbol{\theta}$.
    - Problem:
        - The $y\text{'s}$ cannot be directly observed.
        - Instead, we observe: $\mathbf{x}=g(\mathbf{y})\in X_{ob} \subseteq \mathcal{R}^l$, where $l<m$.
        - Probability density function of $\mathbf{x}$ denoted $p_x(\mathbf{x};\boldsymbol{\theta})$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Expectation Maximisation (EM) - algorithm

    - We have a **many-to-one-mapping**. ->
    - Let $Y(\mathbf{x})\subseteq Y$, that is, the subset of all the $y\text{'s}$ corresponding to a specific $\mathbf{x}$.
    - Then, the probability density of the incomplete data is:
        - $p_x(\mathbf{x};\boldsymbol{\theta})=\int_{Y(\mathbf{x})}p_y(\mathbf{y};\boldsymbol{\theta})d\mathbf{y}$.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Expectation Maximisation (EM) - algorithm

    - Maximum likelihood estimate of $\theta$ is found by:
        - $\hat{\theta}_m : \sum_k \frac{\partial \ln(p_y(y_k; \theta))}{\partial \theta} = \mathbf{0}$
    - Problem:
        - We do not have the $y\text{'s}$.
    - EM-algorithm maximises the expectations of the likelihood function, **conditioned** on the observed samples and the current estimate of $\theta$.
    - Consists of two iterative steps:
        - E-step -> expectation step.
        - M-step -> maximization step

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The E and M step

    - **E-step**:
        - Calculate $Q(\theta; \theta^{(t)}) = \mathbb{E}\left\{ \sum_k \ln p_y(y_k; \Theta) | X, \theta^{(t)} \right\}$
    - **M-step**:
        - Update $\theta^{(t+1)}$ by maximising $Q(\theta; \theta^{(t)})$
        - i.e. $\theta^{(t+1)} : \frac{\partial Q(\theta; \theta^{(t)})}{\partial \theta} = 0$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Practical considerations

    - Need a starting point for our unknown parameters.
        - Many smart ways to find this starting point, but still a challenging aspect.
    - When do we stop iterating? ->

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Application to mixture modelling

    - Consider a case with complete data.
        - Joint events $(\mathbf{x}_k), j_k$.
    - Applying Bayes rule gives:
        - $p_y(\mathbf{x}, j_k;\theta)=p_y(\mathbf{x}| j_k;\theta)P_{jk}$.
    - Assuming independence between samples, the log-likelihood is:
        - $L(\theta)=\sum_{k=1}^{N} \ln(p_y(\mathbf{x}| j_k;\theta)P_{jk})$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Application to mixture modelling

    - Let $\mathbf{P} = \begin{bmatrix} P_1, P_2, \ldots, P_J \end{bmatrix}^T$ and the complete parameters:
    $$
    \boldsymbol{\Theta} = \begin{bmatrix} \boldsymbol{\theta}^T, \mathbf{P}^T \end{bmatrix}^T
    $$

    - Hence the E-step:
    $$
    Q(\boldsymbol{\Theta}; \boldsymbol{\Theta}^{(t)}) = \mathbb{E} \left\{ \sum_{k=1}^N \ln \left( p(\mathbf{x}_k | j_k; \boldsymbol{\Theta}) P_{j_k} \right) \right\}
    $$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Application to mixture modelling - the Gaussian case

    - Eg., for Gaussian with $\Sigma_j = \sigma_j^2 I$:
    - $P_j$, $\mu_j$ and $\sigma_j$ are unknown.

    - **E-step:**
    $$
    p(x_k | j_k; \Theta) = \frac{1}{(2\pi \sigma_j^2)^{1/2}} \exp\left( -\frac{\|x_k - \mu_j\|^2}{2\sigma_j^2} \right)
    $$

    - Thus,
    $$
    Q(\Theta, \Theta^{(t)}) = \sum_{k=1}^N \sum_{j=1}^J P(j | x_k, \Theta^{(t)}) \left( -\frac{1}{2} \ln \sigma_j - \frac{1}{2\sigma_j^2} \|x_k - \mu_j\|^2 + \ln P_j \right)
    $$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Application to mixture modelling - the Gaussian case

    - **M-step:**
    $$
    \mu_j^{(t+1)} = \frac{\sum_{k=1}^N P(j | \mathbf{x}_k; \Theta^{(t)}) \, \mathbf{x}_k}{\sum_{k=1}^N P(j | \mathbf{x}_k; \Theta^{(t)})}
    $$

    $$
    \sigma_j^2 (t+1) = \frac{\sum_{k=1}^N P(j | \mathbf{x}_k; \Theta^{(t)}) \, \|\mathbf{x}_k - \mu_j^{(t+1)}\|^2}{\sum_{k=1}^N P(j | \mathbf{x}_k; \Theta^{(t)})}
    $$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Application to mixture modelling - the Gaussian case

    $$
    P_j(t+1) = \frac{1}{N} \sum_{k=1}^N P(j | \mathbf{x}_k, \Theta^{(t)})
    $$

    Where
    $$
    P(j | \mathbf{x}_k, \Theta^{(t)}) = \frac{p(\mathbf{x}_k | j; \Theta^{(t)}) P_j(t)}{p(\mathbf{x}_k; \Theta^{(t)})}
    $$

    and
    $$
    p(\mathbf{x}_k; \Theta^{(t)}) = \sum_{j=1}^J p(\mathbf{x}_k | j; \Theta^{(t)}) P_j(t)
    $$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 20</div>
        """
    )
    return


@app.cell
def _(data_mix, mo, np, plt):
    from scipy.stats import multivariate_normal

    # Run EM for 10 iterations on the three-Gaussian data generated in the
    # "Mixture models" slide and plot the assignments at selected iterations.
    np.random.seed(0)
    K_em = 3
    N_total_em = data_mix.shape[0]
    means_em = np.random.rand(K_em, 2) * 10
    covs_em = [np.eye(2) for _ in range(K_em)]
    weights_em = np.ones(K_em) / K_em

    def e_step(data, means, covs, weights):
        resp = np.zeros((data.shape[0], K_em))
        for k in range(K_em):
            resp[:, k] = weights[k] * multivariate_normal.pdf(data, mean=means[k], cov=covs[k])
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def m_step(data, resp):
        Nk = resp.sum(axis=0)
        means = np.dot(resp.T, data) / Nk[:, None]
        covs = []
        for k in range(K_em):
            diff = data - means[k]
            cov = np.dot(resp[:, k] * diff.T, diff) / Nk[k]
            covs.append(cov)
        weights = Nk / data.shape[0]
        return means, covs, weights

    iterations_to_plot = [1, 2, 5, 10]
    fig_em, axs_em = plt.subplots(1, 4, figsize=(20, 4))
    means_run = means_em
    covs_run = covs_em
    weights_run = weights_em
    for i in range(1, 11):
        resp_run = e_step(data_mix, means_run, covs_run, weights_run)
        means_run, covs_run, weights_run = m_step(data_mix, resp_run)
        if i in iterations_to_plot:
            idx = iterations_to_plot.index(i)
            axs_em[idx].scatter(
                data_mix[:, 0], data_mix[:, 1],
                c=resp_run.argmax(axis=1), cmap="viridis", s=10,
            )
            for k in range(K_em):
                axs_em[idx].scatter(
                    means_run[k, 0], means_run[k, 1],
                    c="red", marker="x", s=100,
                )
            axs_em[idx].set_title(f"Iteration {i}")
    fig_em.suptitle("EM Algorithm: Expectation and Maximization Steps")

    plt.close(fig_em)
    mo.vstack(
        [
            mo.as_html(fig_em),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 20</div>"""),
        ]
    )
    return


@app.cell
def _(mo, np, plt):
    from ucimlrepo import fetch_ucirepo
    from sklearn.mixture import GaussianMixture

    # Fetch the Iris dataset and show how the number of Gaussian components
    # changes the fitted means on the first two features.
    iris = fetch_ucirepo(id=53)
    X_iris = iris.data.features.iloc[:, :2]
    X_1_name = "sepal length (cm)"
    X_2_name = "sepal width (cm)"

    number_of_components = [2, 3, 4, 5, 6]
    fig_gm, axs_gm = plt.subplots(1, 5, figsize=(12, 3))
    for counter, n_components_i in enumerate(number_of_components):
        gm = GaussianMixture(n_components=n_components_i, random_state=0).fit(X_iris)
        axs_gm[counter].scatter(
            X_iris.iloc[:, 0], X_iris.iloc[:, 1],
            c="none", edgecolor="k", s=50,
        )
        axs_gm[counter].scatter(
            gm.means_[:, 0], gm.means_[:, 1], s=100, color="blue",
        )
        axs_gm[counter].set_xlabel(X_1_name)
        axs_gm[counter].set_ylabel(X_2_name)
    fig_gm.tight_layout()

    plt.close(fig_gm)
    mo.vstack(
        [
            mo.as_html(fig_gm),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 20</div>"""),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
