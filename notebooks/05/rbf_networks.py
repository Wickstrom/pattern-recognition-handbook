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
# Marimo version of the RBF Networks and Prototypical Learning lecture.
# Same content as notebooks/05/rbf_networks.ipynb, but authored as a
# reactive Marimo app so each matplotlib figure stays tied to the data
# generation that produced it (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/05/rbf_networks.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `X`) can be redefined
# in any number of cells, but Marimo requires each name to be owned by
# exactly one cell. Below, the cell-local dataset and figure use the
# `_xor` suffix so they stay independent of any future cells that
# might want similar names.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="full",
    layout_file="layouts/rbf_networks.slides.json",
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
    # RBF Networks and Prototypical Learning
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - We have learned how neural networks transform data into new representation to solve non-linear problems.
    - We will now look at non-linear classification from the perspective of function approximation.
    - Key idea:
        - Transform data in terms of a preselected class of interpolation functions.
    - We will also see how RBF networks relate to the field of prototypical learning.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Function approximation

    - Recall the XOR example:
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 24</div>
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # XOR dataset: four small Gaussian blobs at the corners of the
    # unit square with labels {0, 1, 1, 0}. Used here as a small,
    # already-familiar running example to motivate the function-
    # approximation framing of RBF networks.
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

    fig_xor, (ax_xor_1, ax_xor_2) = plt.subplots(1, 2, figsize=(8, 4))
    ax_xor_1.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], color="blue", label="Class 0")
    ax_xor_1.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], color="red", label="Class 1")
    ax_xor_1.set_xlabel("x1")
    ax_xor_1.set_ylabel("x2")
    ax_xor_2.set_ylim(-0.5, 1.5)
    ax_xor_2.set_xlim(-0.5, 1.5)
    ax_xor_2.set_xlabel("x1")
    ax_xor_2.set_ylabel("x2")

    mo.vstack(
        [
            mo.as_html(fig_xor),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 24</div>"""),
        ]
    )
    return X_xor, y_xor


@app.cell
def _(mo):
    mo.md(
        r"""
    ### General idea

    - Let (non-linear) $f_i: \mathbb{R}^d \rightarrow \mathbb{R}, i=1,\cdots,k$
    - Look at $\mathbf{x} \in \mathbb{R}^d$. (Usually $k>l$)
    - Draw example:
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Network

    - The function approximation setup described above can be thought of as a network:
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Radial basis function (RBF) networks

    - General idea:

    $$f(\|\mathbf{x} - \mathbf{c}_i\|)$$

    - RBF network:

    $$f(\mathbf{x}) = \exp\!\left(-\frac{1}{2\sigma_i^2}\|\mathbf{x} - \mathbf{c}_i\|^2\right)$$

    - Activation at node $f_i$ given by distance to $\mathbf{c}_i$.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Estimating the weights

    - Any linear method will do!
    - Let $\mathbf{y}$ be the desired outputs. Construct matrix $\mathbf{G}$:

    $$\mathbf{G} = \begin{bmatrix} \phi(\mathbf{x}_1, \mathbf{c}_1) & \cdots & \phi(\mathbf{x}_1, \mathbf{c}_k) \\ \vdots & & \vdots \\ \phi(\mathbf{x}_N, \mathbf{c}_1) & \cdots & \phi(\mathbf{x}_N, \mathbf{c}_k) \end{bmatrix}_{N \times k}$$

    - So:

    $$\mathbf{g} = \mathbf{G}\mathbf{w} = \begin{bmatrix} g(\mathbf{x}_1) \\ \vdots \\ g(\mathbf{x}_N) \end{bmatrix}$$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Estimating the weights

    - Choose $\mathbf{w}$ such that $J = \|\mathbf{y} - \mathbf{G}\mathbf{w}\|^2$ is minimized.

    $$\frac{\partial J}{\partial \mathbf{w}} = 0 \implies \boxed{\mathbf{w} = (\mathbf{G}^\top \mathbf{G})^{-1} \mathbf{G}^\top \mathbf{y}}$$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How to choose the centers?

    - **Fixed centres:** Centres selected randomly from the training set.
       - Simple, but not optimum!
    - **Training of the centres:** Let $\sigma_i^2$, $\mathbf{c}_i$, $w_i$, $i=1,\ldots,k$, be free parameters learned from the training set.

    $$\frac{\partial J}{\partial \mathbf{c}_i} = 0 \implies \mathbf{c}_i^{(\text{new})} = \mathbf{c}_i^{(\text{old})} - \mu \frac{\partial J}{\partial \mathbf{c}_i}$$
    - Choose centres according to how the data are distributed in space.
       - The next two slides show an interactive example — first with centres fixed on a grid, then with centres learned by gradient descent.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 24</div>
        """
    )
    return


@app.cell
def _(np):
    # Shared dataset for the function-approximation example on the next
    # three slides: samples of a cosine target plus Gaussian noise. Seeded
    # so figures are reproducible, and generated independently of the UI
    # elements below so that moving a slider never re-samples the data.
    np.random.seed(11)
    x_rbf = np.sort(np.random.uniform(0.0, 1.0, 200))
    y_rbf = 0.3 + 0.2 * np.cos(2 * np.pi * x_rbf) + 0.03 * np.random.randn(200)
    x_grid_rbf = np.linspace(-0.05, 1.05, 400)
    f_grid_rbf = 0.3 + 0.2 * np.cos(2 * np.pi * x_grid_rbf)
    return f_grid_rbf, x_grid_rbf, x_rbf, y_rbf


@app.cell
def _(mo):
    # UI elements for the two interactive demos. They live in this cell
    # (which produces no slide output) so the demo cells below can read
    # their `.value` — Marimo forbids reading a UIElement's value in the
    # cell that created it. Same pattern as density_estimation.py.
    nc_fix = mo.ui.slider(2, 50, value=10, step=1, label="Number of centres")
    sig_fix = mo.ui.slider(0.01, 1.0, value=0.1, label="Kernel width σ")
    nc_learn = mo.ui.slider(2, 30, value=4, step=1, label="Number of centres")
    sig_learn = mo.ui.slider(0.05, 0.5, value=0.15, label="Kernel width σ")
    return nc_fix, nc_learn, sig_fix, sig_learn


@app.cell
def _(f_grid_rbf, mo, plt, x_grid_rbf, x_rbf, y_rbf):
    # Static overview figure: the target function and the noisy training
    # data shared by both interactive demos.
    fig_data, ax_data = plt.subplots(figsize=(11, 5))
    ax_data.scatter(x_rbf, y_rbf, s=15, color="cornflowerblue", alpha=0.6, label="Data")
    ax_data.plot(x_grid_rbf, f_grid_rbf, color="black", linestyle="--", linewidth=2, label="True function")
    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    ax_data.set_ylim(-0.08, 0.68)
    ax_data.legend(loc="upper right")
    fig_data.tight_layout()
    plt.close(fig_data)

    mo.vstack(
        [
            mo.md(
                r"""
    ### Example: function approximation with an RBF network

    - Data: $y = f(x) + \varepsilon$, where $f(x) = 0.3 + 0.2 \cos(2\pi x)$ and $x \in [0, 1]$, $N = 200$ samples.
    - Model: an RBF network in one dimension,

    $$g(x) = \sum_{i=1}^{k} w_i \exp\left(-\frac{1}{2\sigma^2}(x - c_i)^2\right),$$

    - The weights $\mathbf{w}$ are solved by least squares. First, the centres are fixed on a uniform grid; then they are learned by gradient descent.
            """
            ),
            mo.as_html(fig_data),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 24</div>"""),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo, nc_fix, np, plt, sig_fix, x_grid_rbf, x_rbf, y_rbf):
    # Reactive demo A: fixed centres on a uniform grid. Re-fits the
    # network whenever one of the sliders moves.
    nc_a = nc_fix.value
    sigma_a = sig_fix.value

    c_a = np.linspace(0.0, 1.0, nc_a)
    G_a = np.exp(-((x_rbf[:, None] - c_a[None, :]) ** 2) / (2 * sigma_a**2))
    w_a = np.linalg.lstsq(G_a, y_rbf, rcond=None)[0]
    g_grid_a = np.exp(-((x_grid_rbf[:, None] - c_a[None, :]) ** 2) / (2 * sigma_a**2)) @ w_a
    mse_a = np.mean((G_a @ w_a - y_rbf) ** 2)

    fig_a, ax_a = plt.subplots(figsize=(11, 5))
    ax_a.scatter(x_rbf, y_rbf, s=15, color="cornflowerblue", alpha=0.6, label="Data")
    ax_a.plot(x_grid_rbf, g_grid_a, color="darkred", linewidth=2.5, label="RBF network")
    ax_a.scatter(c_a, np.zeros(nc_a), marker="x", color="black", s=70, linewidths=2, label="Centres", zorder=3)
    ax_a.set_xlabel("x")
    ax_a.set_ylabel("y")
    ax_a.set_ylim(-0.08, 0.68)
    ax_a.set_title(rf"{nc_a} centres, $\sigma$ = {sigma_a:.2f} — training MSE: {mse_a:.4f}")
    ax_a.legend(loc="upper right")
    fig_a.tight_layout()
    plt.close(fig_a)

    mo.vstack(
        [
            mo.md(
                r"""
    ### Interactive: fixed centres
            """
            ),
            mo.vstack([nc_fix, sig_fix]),
            mo.as_html(fig_a),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 24</div>"""),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo, nc_learn, np, plt, sig_learn, x_grid_rbf, x_rbf, y_rbf):
    # Reactive demo B: centres learned by alternating optimization —
    # closed-form least squares for the output weights, then a gradient
    # step on the centres (slide "How to choose the centers?"), repeated.
    # Seeded so the initialization is the same on every slider change.
    np.random.seed(5)
    nc_b = nc_learn.value
    sigma_b = sig_learn.value

    c_b = np.linspace(0.0, 1.0, nc_b) + 0.02 * np.random.randn(nc_b)
    c_init_b = c_b.copy()
    for _it_b in range(500):
        G_b = np.exp(-((x_rbf[:, None] - c_b[None, :]) ** 2) / (2 * sigma_b**2))
        w_b = np.linalg.lstsq(G_b, y_rbf, rcond=None)[0]
        resid_b = y_rbf - G_b @ w_b
        grad_b = -(2 * w_b / sigma_b**2) * ((resid_b[:, None] * (x_rbf[:, None] - c_b[None, :])) * G_b).sum(axis=0) / y_rbf.size
        c_b = np.clip(c_b - 0.05 * grad_b, -0.05, 1.05)

    G_b = np.exp(-((x_rbf[:, None] - c_b[None, :]) ** 2) / (2 * sigma_b**2))
    w_b = np.linalg.lstsq(G_b, y_rbf, rcond=None)[0]
    g_grid_b = np.exp(-((x_grid_rbf[:, None] - c_b[None, :]) ** 2) / (2 * sigma_b**2)) @ w_b
    mse_b = np.mean((G_b @ w_b - y_rbf) ** 2)
    c_grid_b = np.linspace(0.0, 1.0, nc_b)
    G_grid_b = np.exp(-((x_rbf[:, None] - c_grid_b[None, :]) ** 2) / (2 * sigma_b**2))
    w_grid_b = np.linalg.lstsq(G_grid_b, y_rbf, rcond=None)[0]
    mse_grid_b = np.mean((G_grid_b @ w_grid_b - y_rbf) ** 2)

    fig_b, ax_b = plt.subplots(figsize=(11, 5))
    ax_b.scatter(x_rbf, y_rbf, s=15, color="cornflowerblue", alpha=0.6, label="Data")
    ax_b.plot(x_grid_rbf, g_grid_b, color="darkred", linewidth=2.5, label="RBF network (learned centres)")
    ax_b.scatter(c_init_b, np.zeros(nc_b), marker="o", facecolors="none", edgecolors="gray", s=80, linewidths=1.5, label="Initial centres", zorder=3)
    ax_b.scatter(c_b, np.zeros(nc_b), marker="x", color="black", s=70, linewidths=2, label="Learned centres", zorder=4)
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("y")
    ax_b.set_ylim(-0.08, 0.68)
    ax_b.set_title(rf"{nc_b} centres, $\sigma$ = {sigma_b:.2f} — MSE: {mse_b:.4f} (fixed grid: {mse_grid_b:.4f})")
    ax_b.legend(loc="upper right")
    fig_b.tight_layout()
    plt.close(fig_b)

    mo.vstack(
        [
            mo.md(
                r"""
    ### Interactive: learned centres
            """
            ),
            mo.vstack([nc_learn, sig_learn]),
            mo.as_html(fig_b),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 24</div>"""),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Prototypical learning

    - RBF networks classify by similarity to reference points — can we make this idea explicit, and use it to build interpretable models?
    - Consider the following example:
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    # Uncertainty figure: motivates why prototypes / typical examples
    # help when a single decision boundary isn't enough to communicate
    # what the model has learned.
    mo.vstack(
        [
            mo.image(src="media/uncertaintyex.jpg", width="800px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What is a prototype?

    - A **prototype** is a representative example of the data: a point $\mathbf{p}_j \in \mathbb{R}^d$ that is "typical" for (a region of) the input space.
    - A prototype-based classifier assigns labels by similarity to a set of prototypes $\mathbf{p}_1, \ldots, \mathbf{p}_M$:

    $$\hat{y}(\mathbf{x}) = \arg\max_{j} \; w_j \, s(\mathbf{x}, \mathbf{p}_j), \qquad s(\mathbf{x}, \mathbf{p}) = \exp\left(-\frac{1}{2\sigma^2} \|\mathbf{x} - \mathbf{p}\|^2\right)$$

    - Prototypes can be defined in the input space or in a latent space!
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### RBF networks are prototype classifiers

    - Comparing with the RBF network,

    $$g(\mathbf{x}) = \sum_{i=1}^{k} w_i \exp\left(-\frac{1}{2\sigma_i^2}\|\mathbf{x} - \mathbf{c}_i\|^2\right),$$

    the centres $\mathbf{c}_i$ play the role of prototypes, the basis functions act as similarities, and the output weights $w_i$ score each prototype.
    - Two special cases:
        - Prototypes fixed to (a subset of) the training samples with $\sigma \rightarrow 0$: nearest-prototype classification $\rightarrow$ nearest neighbour classification.
        - Centres learned by gradient descent: prototypes learned from data.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Prototypes in latent space

    - Prototypes do not have to live in the input space.
    - An encoder maps $\mathbf{x}$ to a representation $\mathbf{z} = \mathrm{enc}(\mathbf{x})$, and the prototypes are defined in this latent space:

    $$\hat{y}(\mathbf{x}) = \arg\max_{j} \; w_j \, s(\mathrm{enc}(\mathbf{x}), \mathbf{p}_j)$$

    - Similarities in a learned representation space are often more meaningful than similarities between raw samples.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    # ProtoVAE figure: prototypes learned in the latent space of a
    # variational autoencoder.
    mo.vstack(
        [
            mo.image(src="media/protovae.png", width="850px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Why prototypes? Interpretability

    - Classifying by similarity to concrete examples makes the behaviour of the model easier to understand:
        - "This sample is classified as $\hat{y}$ because it resembles prototype $\mathbf{p}_j$."
    - Prototypes represent "typical" examples in the data.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 24</div>
        """
    )
    return


@app.cell
def _(np):
    # Data for the prototype browser on the next slide: kernel PCA on a
    # seeded subsample of the digits dataset (bundled with scikit-learn,
    # so no network access). Prototypes = the per-digit medoid in the 2-D
    # embedding, i.e. the sample closest to its class mean.
    from sklearn.datasets import load_digits
    from sklearn.decomposition import KernelPCA

    dgt_all = load_digits()
    rng_dgt = np.random.default_rng(3)
    sub_dgt = rng_dgt.choice(len(dgt_all.data), size=800, replace=False)
    dgt_X = dgt_all.data[sub_dgt]
    dgt_y = dgt_all.target[sub_dgt]
    dgt_Z = KernelPCA(n_components=2, kernel="rbf", gamma=0.001, eigen_solver="dense").fit_transform(dgt_X)
    dgt_proto = {}
    for d_dgt in range(10):
        m_dgt = np.flatnonzero(dgt_y == d_dgt)
        dgt_proto[d_dgt] = int(m_dgt[np.argmin(((dgt_Z[m_dgt] - dgt_Z[m_dgt].mean(0)) ** 2).sum(1))])
    return dgt_X, dgt_Z, dgt_proto, dgt_y


@app.cell
def _(mo):
    # Dropdowns for the prototype browser. They live in this output-free
    # cell so the slide cell below can read their `.value` — Marimo
    # forbids reading a UIElement's value in the cell that created it.
    dd_from = mo.ui.dropdown(options=[str(d) for d in range(10)], value="3", label="From prototype")
    dd_to = mo.ui.dropdown(options=[str(d) for d in range(10)], value="8", label="To prototype")
    return dd_from, dd_to


@app.cell
def _(dd_from, dd_to, dgt_X, dgt_Z, dgt_proto, dgt_y, mo, np, plt):
    # Reactive prototype browser: pick two digit prototypes and walk
    # through the actual samples encountered along the straight path
    # between them in the kernel PCA embedding.
    cls_p = int(dd_from.value)
    cls_q = int(dd_to.value)
    pa_p = dgt_Z[dgt_proto[cls_p]]
    pb_p = dgt_Z[dgt_proto[cls_q]]

    ts_p = np.linspace(0.0, 1.0, 25)
    zpath_p = (1 - ts_p)[:, None] * pa_p + ts_p[:, None] * pb_p
    nearest_p = ((dgt_Z[None, :, :] - zpath_p[:, None, :]) ** 2).sum(-1).argmin(1)
    seq_p, seen_p = [], set()
    for i_p in nearest_p:
        if i_p not in seen_p:
            seen_p.add(i_p)
            seq_p.append(int(i_p))
    if len(seq_p) > 10:
        keep_p = np.round(np.linspace(0, len(seq_p) - 1, 10)).astype(int)
        seq_p = [seq_p[i] for i in keep_p]

    n_img_p = len(seq_p)

    fig_map_p, ax_map_p = plt.subplots(figsize=(10, 7))
    for d_p in range(10):
        m_p = dgt_y == d_p
        ax_map_p.scatter(dgt_Z[m_p, 0], dgt_Z[m_p, 1], s=14, color=f"C{d_p}", alpha=0.55)
    ax_map_p.plot([pa_p[0], pb_p[0]], [pa_p[1], pb_p[1]], color="black", linestyle="--", linewidth=1.8, zorder=3)
    ax_map_p.scatter(dgt_Z[seq_p, 0], dgt_Z[seq_p, 1], facecolors="none", edgecolors="black", s=160, linewidths=1.4, zorder=4)
    for d_p in range(10):
        zp_p = dgt_Z[dgt_proto[d_p]]
        ax_map_p.scatter(zp_p[0], zp_p[1], marker="D", color="black", s=70, zorder=5)
        ax_map_p.text(zp_p[0], zp_p[1] + 0.015, str(d_p), fontsize=11, ha="center", zorder=6)
    ax_map_p.set_xlabel("kernel PCA component 1", fontsize=13)
    ax_map_p.set_ylabel("kernel PCA component 2", fontsize=13)
    ax_map_p.set_title(f"Latent space: {cls_p} → {cls_q}", fontsize=14)
    fig_map_p.tight_layout()
    plt.close(fig_map_p)

    fig_img_p, axes_img_p = plt.subplots(1, n_img_p, figsize=(n_img_p * 1.15, 1.6))
    if n_img_p == 1:
        axes_img_p = [axes_img_p]
    for j_p, i_p in enumerate(seq_p):
        axes_img_p[j_p].imshow(dgt_X[i_p].reshape(8, 8), cmap="gray_r", interpolation="nearest")
        axes_img_p[j_p].set_title(str(int(dgt_y[i_p])), fontsize=11)
        axes_img_p[j_p].axis("off")
    fig_img_p.tight_layout()
    plt.close(fig_img_p)

    mo.vstack(
        [
            mo.md(
                r"""
    ### Browsing the latent space between prototypes

    - Kernel PCA embedding of the digits dataset; the black diamonds are prototypes (the most typical sample of each digit in the embedding).
    - Pick two prototypes: the row shows the actual samples encountered when walking from one to the other.
            """
            ),
            mo.hstack([dd_from, dd_to]),
            mo.as_html(fig_map_p),
            mo.as_html(fig_img_p),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 24</div>"""),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Prototypes vs. centroids

    - A prototype can be an observed sample: a concrete case that can be inspected and compared with new samples.
    - A centroid (e.g., a class mean) is a synthesized point that need not correspond to any real sample.
    - **Archetypal analysis** (Cutler and Breiman, 1994) summarizes a class by a small set of extreme, "pure" examples — data points on the boundary of the class rather than its centre.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    # Archetype figure: an alternative way of summarising a class with
    # a single "typical" example rather than a centroid.
    mo.vstack(
        [
            mo.image(src="media/archetype.png", width="400px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Challenges with prototypical learning

    - Curse of dimensionality:
        - In high dimensions, pairwise distances concentrate, so the similarities $s(\mathbf{x}, \mathbf{p}_j)$ become less informative.
        - Covering the input space with prototypes may require exponentially many of them.
    - Computational cost:
        - Each prediction requires computing $M$ similarities, each of cost $\mathcal{O}(d)$, and the prototypes must be stored in memory.
    - Optimization:
        - Learning $\{\mathbf{c}_i, \sigma_i, w_i\}$ jointly is non-convex, so the result depends on initialization (recall: centres selected randomly from the training set).
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">24 / 24</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
