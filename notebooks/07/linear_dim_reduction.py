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
# Marimo version of the Data transformation and dimensionality reduction
# I lecture (FDA + PCA).
# Same content as notebooks/07/linear_dim_reduction.ipynb, but authored
# as a reactive Marimo app (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/07/linear_dim_reduction.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/linear_dim_reduction.slides.json",
    css_file="../_shared/math.css",
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
    # Data transformation and dimensionality reduction (DTDR) I
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - So far, the course has been problem oriented (mainly classification).
    - However, an essential part of pattern recognition is data analysis.
    - In this lecture, we will look at methods that compress data into a more compact representation through linear transformations.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Analyzing data

    - Real data often live in a high-dimensional space, but the information
      they carry may be concentrated on a much smaller subspace.
    - Example below: two features that are almost perfectly correlated.
        - Little is lost by describing the cloud with a single direction.
        """
    )
    return


@app.cell
def _(np, plt):
    # Motivating example: two strongly correlated features. The scatter
    # stretches along a single direction, so the cloud is essentially
    # one-dimensional even though it lives in 2-D.
    rng_corr = np.random.default_rng(0)
    cov_corr = np.array([[1.0, 0.95], [0.95, 1.0]])
    X_corr = rng_corr.multivariate_normal([0.0, 0.0], cov_corr, size=400)

    fig_corr = plt.figure(figsize=(10, 5))
    gs_corr = fig_corr.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
    )
    ax_corr = fig_corr.add_subplot(gs_corr[1, 0])
    ax_corr_top = fig_corr.add_subplot(gs_corr[0, 0], sharex=ax_corr)
    ax_corr_right = fig_corr.add_subplot(gs_corr[1, 1], sharey=ax_corr)

    ax_corr.scatter(X_corr[:, 0], X_corr[:, 1], s=12, alpha=0.5)
    ax_corr_top.hist(X_corr[:, 0], bins=30, color="tab:blue")
    ax_corr_right.hist(
        X_corr[:, 1], bins=30, orientation="horizontal", color="tab:blue"
    )
    ax_corr.set_xlabel("$x_1$")
    ax_corr.set_ylabel("$x_2$")
    ax_corr_top.tick_params(labelbottom=False)
    ax_corr_right.tick_params(labelleft=False)
    ax_corr_top.set_title(r"Two highly correlated features ($\rho = 0.95$)")

    plt.close(fig_corr)
    fig_corr
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Problems

    - The data dimensionality may be very large.
        - Computationally demanding.
        - Curse of dimensionality
    - Some parts of the data may not be discriminative / be redundant.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remedies

    - May pick only parts of the data to use.
    - May transform the data!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Approaches to DTDR

    - To better discriminate between classes (supervised).
    - To remove redundancy (unsupervised).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Fisher discriminant analysis (FDA)

    - Transform (project) to 1D: $z = \mathbf{w}^T \mathbf{x}$
    - Start with the two class case and $P(w_1)=P(w_2)$
    - Fisher discriminant ratio (FDR) — large is good:

    $$
    \mathrm{FDR} = \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2}
    $$

    where $\mu_i$ and $\sigma_i^2$ are the mean and variance of the
    projected class-$i$ data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Scatter matrices

    - Within class: $\boldsymbol{S}_w = \sum\limits_{i=1}^{M} P(w_i) \boldsymbol{\Sigma}_i$

    - where $\boldsymbol{\Sigma}_i = \mathbb{E}\left[(\mathbf{x} - \boldsymbol{\mu}_i)(\mathbf{x} - \boldsymbol{\mu}_i)^T\right]$

    ---

    - Between class: $\boldsymbol{S}_B = \sum\limits_{i=1}^{M} P(w_i) (\boldsymbol{\mu}_i - \boldsymbol{\mu})(\boldsymbol{\mu}_i - \boldsymbol{\mu})^T$

    - where $\boldsymbol{\mu} = \sum\limits_{i=1}^{M} P(w_i) \boldsymbol{\mu}_i$ is the global mean.

    - Both are $d \times d$ matrices: $\boldsymbol{S}_w$ measures spread *inside* classes, $\boldsymbol{S}_B$ spread *between* class means.
        """
    )
    return


@app.cell
def _(np, plt):
    # Two 2-D Gaussian classes, reused by the Fisher-projection figure
    # below. Equal priors so the within/between scatter reduces to the
    # simple averages used in the slides. Seeded for reproducibility.
    from matplotlib.patches import Ellipse

    rng_fda = np.random.default_rng(0)
    mu_fda_1 = np.array([-1.5, 0.5])
    mu_fda_2 = np.array([1.5, -0.5])
    cov_fda_1 = np.array([[1.0, 0.6], [0.6, 1.0]])
    cov_fda_2 = np.array([[0.8, -0.4], [-0.4, 0.9]])
    X_fda_1 = rng_fda.multivariate_normal(mu_fda_1, cov_fda_1, size=200)
    X_fda_2 = rng_fda.multivariate_normal(mu_fda_2, cov_fda_2, size=200)

    fig_scat, ax_scat = plt.subplots(figsize=(7, 6))
    ax_scat.scatter(
        X_fda_1[:, 0], X_fda_1[:, 1], s=12, alpha=0.4, color="tab:blue", label="$w_1$"
    )
    ax_scat.scatter(
        X_fda_2[:, 0], X_fda_2[:, 1], s=12, alpha=0.4, color="tab:orange", label="$w_2$"
    )
    # 2-sigma covariance ellipses: the "within-class" scatter S_w.
    for mu_i, cov_i, col_i in [
        (mu_fda_1, cov_fda_1, "tab:blue"),
        (mu_fda_2, cov_fda_2, "tab:orange"),
    ]:
        w_i, v_i = np.linalg.eigh(cov_i)
        order_i = np.argsort(w_i)[::-1]
        w_i, v_i = w_i[order_i], v_i[:, order_i]
        ang_i = np.degrees(np.arctan2(v_i[1, 0], v_i[0, 0]))
        ax_scat.add_patch(
            Ellipse(
                mu_i,
                *(4 * np.sqrt(w_i)),
                angle=ang_i,
                fill=False,
                edgecolor=col_i,
                lw=2,
                ls="--",
            )
        )
        ax_scat.plot(*mu_i, marker="X", color=col_i, ms=14, mec="k")
    # Global mean = average of class means for equal priors.
    mu_fda = 0.5 * (mu_fda_1 + mu_fda_2)
    ax_scat.plot(*mu_fda, marker="*", color="k", ms=18, label=r"global mean $\mu$")
    ax_scat.set_xlabel("$x_1$")
    ax_scat.set_ylabel("$x_2$")
    ax_scat.set_aspect("equal")
    ax_scat.legend()
    ax_scat.set_title("Within-class scatter $S_w$ and class means")

    plt.close(fig_scat)
    fig_scat
    return (
        X_fda_1,
        X_fda_2,
        cov_fda_1,
        cov_fda_2,
        mu_fda,
        mu_fda_1,
        mu_fda_2,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remark

    - Class separability in $\mathbf{x}$ can be measured by e.g.

    $$\frac{\operatorname{trace}(\boldsymbol{S}_w)}{\operatorname{trace}(\boldsymbol{S}_B)}$$

    - Here **small is good**: little within-class spread relative to the between-class spread.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Fisher discriminant analysis

    - Remember; want to learn a transformation into 1D $z = \mathbf{w}^T \mathbf{x} \;\Rightarrow\; \mu_z = \mathbf{w}^T \boldsymbol{\mu}$

    - In the projected space, with equal priors $P(w_1)=P(w_2)=\tfrac{1}{2}$:

    $$S_B = (\mu_1 - \mu_2)^2, \qquad \mu_i = \mathbf{w}^T \boldsymbol{\mu}_i$$

    $$\sigma_i^2 = \mathbb{E}\left[(z - \mu_i)^2\right] = \mathbf{w}^T \boldsymbol{\Sigma}_i \mathbf{w}$$

    $$S_w = \tfrac{1}{2}\sigma_1^2 + \tfrac{1}{2}\sigma_2^2$$

    - Hence: Fisher discriminant ratio (FDR) — to be **maximized**:

    $$\mathrm{FDR}(\mathbf{w}) = \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks

    - Don't need $P(w_1) = P(w_2)$, but easy to solve.

    - If $P(w_1) \neq P(w_2)$: $\mathbf{w}$ is the leading eigenvector of $\boldsymbol{S}_w^{-1} \boldsymbol{S}_B$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### FDR

    - Want:

    $$\arg\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{S}_B \mathbf{w}}{\mathbf{w}^T \boldsymbol{S}_w \mathbf{w}}$$

    - At the solution (generalized eigenvalue problem):
      $\boldsymbol{S}_B \mathbf{w} = \lambda \boldsymbol{S}_w \mathbf{w}$
      $\;\Leftrightarrow\;$ $\boldsymbol{S}_w^{-1} \boldsymbol{S}_B \mathbf{w} = \lambda \mathbf{w}$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Solving the Fisher discriminant ratio

    I. If $P(w_1) = P(w_2)$:
    $$
    \lambda \boldsymbol{S}_w \mathbf{w} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \mathbf{w}
    $$
    $\implies$ $\mathbf{w} \propto \boldsymbol{S}_w^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$

    ---

    II. If $P(w_1) \neq P(w_2)$:
    $$
    \mathbf{w} = \text{leading eigenvector of } \boldsymbol{S}_w^{-1} \boldsymbol{S}_B
    $$
        """
    )
    return


@app.cell
def _(mo):
    theta_fda = mo.ui.slider(
        0, 180, value=30, step=1, label="Projection angle (degrees)"
    )
    theta_fda
    return (theta_fda,)


@app.cell
def _(
    X_fda_1,
    X_fda_2,
    cov_fda_1,
    cov_fda_2,
    mu_fda,
    mu_fda_1,
    mu_fda_2,
    np,
    plt,
    theta_fda,
):
    # Slide the projection direction and watch the FDR peak at the
    # Fisher-optimal direction w ∝ S_w^{-1}(mu_1 - mu_2).
    theta_rad = np.radians(theta_fda.value)
    w_fda = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    S_w_fda = 0.5 * (cov_fda_1 + cov_fda_2)
    w_opt_fda = np.linalg.solve(S_w_fda, mu_fda_1 - mu_fda_2)
    w_opt_fda = w_opt_fda / np.linalg.norm(w_opt_fda)
    if w_opt_fda[0] < 0:
        w_opt_fda = -w_opt_fda

    z_fda_1 = X_fda_1 @ w_fda
    z_fda_2 = X_fda_2 @ w_fda
    fdr_cur = (z_fda_1.mean() - z_fda_2.mean()) ** 2 / (
        z_fda_1.var() + z_fda_2.var()
    )

    fig_fda, (ax_fda_2d, ax_fda_1d) = plt.subplots(1, 2, figsize=(12, 5))
    ax_fda_2d.scatter(
        X_fda_1[:, 0], X_fda_1[:, 1], s=10, alpha=0.35, color="tab:blue"
    )
    ax_fda_2d.scatter(
        X_fda_2[:, 0], X_fda_2[:, 1], s=10, alpha=0.35, color="tab:orange"
    )
    for vec_fda, col_fda, lab_fda in [
        (w_fda * 3, "k", "$w$"),
        (w_opt_fda * 3, "tab:green", r"$w_{\mathrm{opt}}$"),
    ]:
        ax_fda_2d.annotate(
            "",
            xy=mu_fda + vec_fda,
            xytext=mu_fda,
            arrowprops=dict(arrowstyle="->", lw=2.5, color=col_fda),
        )
        ax_fda_2d.text(*(mu_fda + vec_fda * 1.08), lab_fda, color=col_fda, fontsize=13)
    ax_fda_2d.set_xlabel("$x_1$")
    ax_fda_2d.set_ylabel("$x_2$")
    ax_fda_2d.set_aspect("equal")
    ax_fda_2d.set_title("Slider direction $w$ vs Fisher optimum $w_{\\mathrm{opt}}$")

    ax_fda_1d.hist(z_fda_1, bins=30, alpha=0.6, color="tab:blue")
    ax_fda_1d.hist(z_fda_2, bins=30, alpha=0.6, color="tab:orange")
    ax_fda_1d.set_xlabel("projected coordinate $z = \\mathbf{w}^T \\mathbf{x}$")
    ax_fda_1d.set_ylabel("count")
    ax_fda_1d.set_title(f"FDR = {fdr_cur:.2f}")

    plt.close(fig_fda)
    fig_fda
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remark

    - Generalized to $\mathbf{z} = \mathbf{W}^T \mathbf{x} \in \mathbb{R}^k$ where $k \leq d$ and $\mathbf{W} \in \mathbb{R}^{d \times k}$.
        - More complex (pages 291-297 in book).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Principal Component Analysis (PCA)

    - First: $\mathbf{z} = \mathbf{A} \mathbf{x}$ such that $\mathbf{z} \in \mathbb{R}^d$, $\mathbf{x} \in \mathbb{R}^d$, and $\mathbf{A} \in \mathbb{R}^{d \times d}$
    - Want: $\boldsymbol{\Sigma}_z$ diagonal!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### A closer look at the covariance matrix

    - Have:

    $$
    \boldsymbol{\Sigma}_z = \mathbb{E}[(\mathbf{z} - \boldsymbol{\mu}_z)(\mathbf{z} - \boldsymbol{\mu}_z)^T]
    $$

    $$
    = \mathbb{E}[(\mathbf{A}\mathbf{x} - \mathbf{A}\boldsymbol{\mu}_x)(\mathbf{A}\mathbf{x} - \mathbf{A}\boldsymbol{\mu}_x)^T]
    = \mathbf{A} \, \boldsymbol{\Sigma}_x \mathbf{A}^T
    $$

    ---

    - $\boldsymbol{\Sigma}_x$: symmetric and positive semi-definite $\implies$ orthogonal eigenvectors and non-negative eigenvalues.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Eigendecomposition of the covariance matrix

    - Let $\mathbf{E} = [\mathbf{e}_1, \ldots, \mathbf{e}_d]$

    $$
    \boldsymbol{\Sigma}_x \mathbf{E} = \mathbf{E} \boldsymbol{\Lambda}
    $$

    where
    $$
    \boldsymbol{\Lambda} = \begin{bmatrix}
    \lambda_1 & 0 & \cdots & 0 \\
    0 & \lambda_2 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & \lambda_d
    \end{bmatrix}
    $$
    (diagonal matrix of eigenvalues)

    ---

    - With $\mathbf{A} = \mathbf{E}^T$ (orthonormal), the covariance of the
      transformed data is diagonal:

    $$
    \boldsymbol{\Sigma}_z = \mathbf{E}^T \boldsymbol{\Sigma}_x \mathbf{E} = \boldsymbol{\Lambda}
    $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interpreting the eigenvalues and eigenvectors

    - $\boldsymbol{\Sigma}_z = \boldsymbol{\Lambda}$: the transformed features
      are uncorrelated.
    - The variance of $z_i$ equals $\lambda_i$.
    - The eigenvectors $\mathbf{e}_i$ are the directions of maximal variance in
      the original space.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Variance maximally preserved

    - First: $\sum_{i=1}^d \text{Var}(z_i) = \operatorname{trace}(\boldsymbol{\Sigma}_z) = \sum_{i=1}^d \lambda_i$

    - Thus: Let $\mathbf{A} = \mathbf{E}^T = [\mathbf{e}_1, \ldots, \mathbf{e}_d]^T$

    - Remark: assume $\mathbb{E}[\mathbf{x}] = 0$ (center the data first).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example: projecting onto the eigenvectors

    - The data cloud below is **unlabeled** — PCA only sees the point cloud.
    - Use the dropdown to project onto $e_1$ (most variance) or $e_2$ (least).
    - Compare the spread of the 1-D projections with $\sqrt{\lambda_i}$.
        """
    )
    return


@app.cell
def _():
    # Unsupervised PCA illustration: two overlapping 2-D blobs (the labels
    # used to *generate* the data are not used anywhere downstream — the
    # eigendecomposition only sees the unlabeled point cloud). Seeded so
    # the figure is stable across re-renders.
    rng_pca = np.random.default_rng(0)
    X_pca = np.vstack(
        [
            rng_pca.normal(loc=[-1.0, 0.3], scale=[0.55, 0.35], size=(150, 2)),
            rng_pca.normal(loc=[0.6, -0.4], scale=[0.65, 0.55], size=(150, 2)),
        ]
    )
    mu_pca = X_pca.mean(axis=0)
    Xc_pca = X_pca - mu_pca
    cov_pca = np.cov(Xc_pca, rowvar=False)
    w_pca, V_pca = np.linalg.eigh(cov_pca)
    # np.linalg.eigh returns ascending order; flip so e_1 is the leading one.
    order_pca = np.argsort(w_pca)[::-1]
    w_pca = w_pca[order_pca]
    V_pca = V_pca[:, order_pca]
    return V_pca, w_pca, X_pca, Xc_pca, mu_pca


@app.cell
def _(mo):
    # Pick which eigenvector to project onto. The dict's value is the index
    # into V_pca, so `pca_dir.value` is an int ready to use.
    pca_dir = mo.ui.dropdown(
        options={
            "$e_1$ (most variance)": 0,
            "$e_2$ (least variance)": 1,
        },
        value="$e_1$ (most variance)",
        label="Project along",
    )
    pca_dir
    return (pca_dir,)


@app.cell
def _(V_pca, X_pca, Xc_pca, mu_pca, np, pca_dir, plt, w_pca):
    i_pca = int(pca_dir.value)
    v_pca = V_pca[:, i_pca]
    proj_pca = Xc_pca @ v_pca  # 1-D projections of every centred point

    fig_pca, (ax_pca_2d, ax_pca_1d) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: 2-D scatter, both eigenvectors drawn from the mean, and
    # thin projection lines from each point to its image on the chosen
    # eigenvector.
    ax_pca_2d.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.35, s=12, color="gray")
    scale_pca = 2.5 * np.sqrt(w_pca.max())
    for j in range(2):
        ev = V_pca[:, j] * scale_pca
        chosen = j == i_pca
        ax_pca_2d.annotate(
            "",
            xy=mu_pca + ev,
            xytext=mu_pca,
            arrowprops=dict(
                arrowstyle="->",
                lw=2.5,
                color="tab:red" if chosen else "tab:blue",
                alpha=1.0 if chosen else 0.35,
            ),
        )
        ax_pca_2d.text(
            *(mu_pca + ev * 1.15),
            f"$e_{{{j + 1}}}$",
            fontsize=15,
            ha="center",
            color="tab:red" if chosen else "tab:blue",
            alpha=1.0 if chosen else 0.5,
        )
    x_recon_pca = mu_pca + proj_pca[:, None] * v_pca
    for k_pca in range(0, len(X_pca), 8):
        ax_pca_2d.plot(
            [X_pca[k_pca, 0], x_recon_pca[k_pca, 0]],
            [X_pca[k_pca, 1], x_recon_pca[k_pca, 1]],
            color="tab:red",
            alpha=0.18,
            lw=0.6,
        )
    ax_pca_2d.set_aspect("equal")
    ax_pca_2d.set_title(
        f"Project along $e_{{{i_pca + 1}}}$  "
        f"($\\lambda_{{{i_pca + 1}}} = {w_pca[i_pca]:.2f}$, "
        f"{100 * w_pca[i_pca] / w_pca.sum():.0f}\\% of total variance)"
    )

    # Right panel: histogram of 1-D projections. The wider this is, the
    # more variance is preserved along that direction.
    ax_pca_1d.hist(proj_pca, bins=30, color="tab:red", alpha=0.75, edgecolor="white")
    ax_pca_1d.set_xlabel(f"projection onto $e_{{{i_pca + 1}}}$")
    ax_pca_1d.set_ylabel("count")
    ax_pca_1d.set_title(
        f"1-D spread: $\\sigma = {np.std(proj_pca):.2f}$ "
        f"$\\;(\\sqrt{{\\lambda_{{{i_pca + 1}}}}})$"
    )

    plt.close(fig_pca)
    fig_pca
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## PCA is reconstruction / compression

    - Let $\mathbf{z} \in \mathbb{R}^d = [z(0), z(1), \ldots, z(d-1)]^T$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### PCA is reconstruction / compression

    - Have $\mathbf{x} = \mathbf{A}^T \mathbf{z}$ (with $\mathbf{A} = \mathbf{E}^T$)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### PCA is reconstruction / compression

    - **MSE:** $\mathbb{E}\left[\|\mathbf{x} - \hat{\mathbf{x}}\|^2\right]$

    For $\mathbf{z} \in \mathbb{R}^k$: $\hat{\mathbf{x}} = \sum_{i=0}^{k-1} z(i) \mathbf{e}_i$

    If $z(i) = 0$ for $i \geq k$:
    $$
    \mathbb{E}\left[\|\mathbf{x} - \hat{\mathbf{x}}\|^2\right] = \sum_{i=k}^{d-1} \lambda_i
    $$

    $\implies$ **MSE is minimized**.

    ---

    - **Compression:** Store/save $\mathbf{z} \in \mathbb{R}^k$ instead of $\mathbf{x}$ (e.g. images).
    - **Reconstruct:** $\hat{\mathbf{x}}$ using $\mathbf{z}$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example: compressing handwritten digits

    - Each $8 \times 8$ digit image is a point in $\mathbb{R}^{64}$.
    - Drag the slider to keep the first $k$ principal components and
      reconstruct.
    - The reconstruction error should follow $\sum_{i \geq k} \lambda_i$.
        """
    )
    return


@app.cell
def _(mo):
    k_rec = mo.ui.slider(
        1, 64, value=10, step=1, label="Principal components kept, $k$"
    )
    k_rec
    return (k_rec,)


@app.cell
def _(k_rec, np, plt):
    # PCA reconstruction on the digits dataset (bundled with scikit-learn,
    # so no network access). The empirical MSE is compared with the
    # theoretical tail sum of the eigenvalues from the slides.
    from sklearn.datasets import load_digits

    dgt_rec = load_digits()
    X_rec = dgt_rec.data.astype(float)
    mu_rec = X_rec.mean(axis=0)
    Xc_rec = X_rec - mu_rec
    # PCA via SVD: principal directions are the right singular vectors.
    _, s_rec, Vt_rec = np.linalg.svd(Xc_rec, full_matrices=False)
    lam_rec = s_rec**2 / (len(X_rec) - 1)  # eigenvalues of the covariance

    k_rec_v = int(k_rec.value)
    z_rec = Xc_rec @ Vt_rec[:k_rec_v].T
    Xhat_rec = z_rec @ Vt_rec[:k_rec_v] + mu_rec
    mse_cur = ((X_rec - Xhat_rec) ** 2).sum(axis=1).mean()

    # Empirical and theoretical reconstruction error as a function of k.
    # Keeping k components leaves eigenvalues i = k, ..., d-1, so the
    # theoretical error is the tail sum lam_rec[k:].
    ks_rec = np.arange(1, len(lam_rec) + 1)
    mse_curve = np.array(
        [
            ((Xc_rec - (Xc_rec @ Vt_rec[:k].T) @ Vt_rec[:k]) ** 2).sum(axis=1).mean()
            for k in ks_rec
        ]
    )
    theory_curve = np.array([lam_rec[k:].sum() for k in ks_rec])

    fig_rec = plt.figure(figsize=(12, 4.5))
    gs_rec = fig_rec.add_gridspec(1, 3, width_ratios=(1, 1, 1.7), wspace=0.3)
    ax_orig = fig_rec.add_subplot(gs_rec[0, 0])
    ax_recon = fig_rec.add_subplot(gs_rec[0, 1])
    ax_err = fig_rec.add_subplot(gs_rec[0, 2])

    ax_orig.imshow(X_rec[0].reshape(8, 8), cmap="gray_r")
    ax_orig.set_title("Original")
    ax_orig.axis("off")
    ax_recon.imshow(Xhat_rec[0].reshape(8, 8), cmap="gray_r")
    ax_recon.set_title(f"Reconstruction, $k={k_rec_v}$")
    ax_recon.axis("off")

    ax_err.plot(ks_rec, mse_curve, label="empirical MSE")
    ax_err.plot(ks_rec, theory_curve, "--", label=r"$\sum_{i \geq k} \lambda_i$")
    ax_err.axvline(k_rec_v, color="k", ls=":", lw=1)
    ax_err.scatter([k_rec_v], [mse_cur], color="tab:red", zorder=3)
    ax_err.set_xlabel("components kept, $k$")
    ax_err.set_ylabel("reconstruction MSE")
    ax_err.legend()
    ax_err.set_title(f"MSE at $k={k_rec_v}$: {mse_cur:.0f}")

    plt.close(fig_rec)
    fig_rec
    return


if __name__ == "__main__":
    app.run()
