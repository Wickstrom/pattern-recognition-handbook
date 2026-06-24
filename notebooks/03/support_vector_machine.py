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
# Marimo version of the support vector machine lecture.
# Same content as notebooks/03/support_vector_machine.ipynb, but authored
# as a reactive Marimo app so each matplotlib figure stays tied to the
# data generation that produced it (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/03/support_vector_machine.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `X`, `y`) can be
# redefined in any number of cells, but Marimo requires each name to be
# owned by exactly one cell. Below, every cell-local dataset and figure
# is given a unique suffix (`_perceptron`, `_nonsep`, `_moons`, `_wine`)
# so the slides stay independent and the data each figure displays is
# reproducible.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/support_vector_machine.slides.json",
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
    # the value (3) matches the original notebook's opening seed so the
    # perceptron-comparison slide stays consistent.
    np.random.seed(3)
    return np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Support Vector Machine (SVM)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Last week we finished the lecture by learning about the Perceptron.
    - Iterative algorithm that will become very useful in the next lectures.
    - But, also some serious limitations:
        - No training error -> assumes linear separability.
        - Not unique $\mathbf{w}$.
    - Consider the following example:
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Perceptron-comparison example: two well-separated Gaussians with
    # the same covariance, but wider apart than the linear-classifiers
    # examples. Many hyperplanes separate this data, motivating the
    # need for SVM's margin-maximization argument.
    n_perceptron = 100

    mu1_perceptron = np.array([1, 1])
    mu2_perceptron = np.array([3.0, 3.0])
    sigma_perceptron = np.array([[0.2, 0.0], [0.0, 0.2]])

    x_train_1_perceptron = np.random.multivariate_normal(mu1_perceptron, sigma_perceptron, n_perceptron)
    x_train_2_perceptron = np.random.multivariate_normal(mu2_perceptron, sigma_perceptron, n_perceptron)

    fig_perceptron, ax_perceptron = plt.subplots(figsize=(8, 8))
    ax_perceptron.scatter(
        x_train_1_perceptron[:, 0], x_train_1_perceptron[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_perceptron.scatter(
        x_train_2_perceptron[:, 0], x_train_2_perceptron[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2",
    )
    ax_perceptron.legend()

    mo.as_html(fig_perceptron)
    plt.close(fig_perceptron)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Support Vector Machine (SVM)

    - Find $$g(\mathbf{x})=\mathbf{w}^T\mathbf{x}+w_0$$ such that the **margin** $m_{\text{svm}}$ is maximum (optimal).
        - Leaves room for variation between training data and test data.
        - **Generalizes** better.
        """
    )
    return


@app.cell
def _(mo, plt):
    # Empty coordinate frame used as the canvas for hand-drawing the
    # SVM decision boundary and the ±1 margins during the lecture.
    fig_axes, ax_axes = plt.subplots(figsize=(6, 4))
    ax_axes.axhline(0, color="black", linewidth=2)
    ax_axes.axvline(0, color="black", linewidth=2)
    ax_axes.set_xlim(-0.3, 5)
    ax_axes.set_ylim(-0.4, 5)
    ax_axes.set_xticks([])
    ax_axes.set_yticks([])

    mo.as_html(fig_axes)
    plt.close(fig_axes)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Support Vector Machine (SVM)

    - Note: $g(\mathbf{x'})$ not unique: $$\mathbf{w'}=a\mathbf{w}$$
    - Define: $$g(\mathbf{x'})= \pm 1 \hspace{0.1cm} \text{for $\mathbf{x}$ on margin.}$$
    - Hence: $$m_{\text{svm}}=\frac{2}{\| \mathbf{w} \|}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## SVM optimization problem

    - $$ \underset{\mathbf{w}, w_0}{\operatorname{min}}\ J(\mathbf{w}, w_0)\coloneqq \frac{1}{2}\| \mathbf{w} \|^2 $$
    - $$ \text{subject to} \hspace{0.2cm} y_i\left(\mathbf{w}^T \mathbf{x_i}+w_0\right) \geq 1$$
    - "A training algorithm for optimal margin classifiers" - (Boser, Guyon, Vapnik, 1992.)
    - "Support-vector networks" - (Cortes and Vapnik, 1995.)
    - How to optimize the objective function?
        - Lagrange multipliers!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Hint on Lagrangian optimization

    - $$ \underset{\boldsymbol{\theta}}{\operatorname{min}}\ J(\boldsymbol{\theta}) $$
    - $$ \text{subject to} \hspace{0.2cm} f(\boldsymbol{\theta})= \theta_1 \theta_2-3 \geq 0 $$
    - Primal problem $\boldsymbol{\theta}$.
    - Note: At $\boldsymbol{\theta}=\boldsymbol{\theta}_{*}\Rightarrow$ $$ \frac{\partial}{\partial \boldsymbol{\theta}} J(\boldsymbol{\theta}) = \lambda\frac{\partial}{\partial \boldsymbol{\theta}} f(\boldsymbol{\theta}) \text{, for } \lambda > 0$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Defining the Lagrange function

    - $$ L(\boldsymbol{\theta}, \lambda)=J(\boldsymbol{\theta})-\lambda f(\boldsymbol{\theta}) $$
    - Then: $$ \underset{\boldsymbol{\theta}}{\operatorname{min}}\ L(\boldsymbol{\theta}, \lambda)  \Rightarrow \frac{\partial}{\partial \boldsymbol{\theta}}J(\boldsymbol{\theta}) - \lambda \frac{\partial}{\partial \boldsymbol{\theta}}f(\boldsymbol{\theta}) = 0$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Illustrating the Lagrangian

    - $$ \underset{\boldsymbol{\theta}}{\operatorname{min}}\ J(\boldsymbol{\theta}) $$
    - $$ \text{subject to} \hspace{0.2cm} f_i(\boldsymbol{\theta}) \geq 0 \hspace{0.2cm} \text{for } i=1,\cdots,N $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Illustrating the Lagrangian

    - Note: Only one solution $\boldsymbol{\theta}$.
    - Due to $J(\boldsymbol{\theta})$ being convex and feasible region being a convex set.
    - Langrange function: $$ L(\boldsymbol{\theta}, \lambda)=J(\boldsymbol{\theta})-\sum_{i=1}^N\lambda_i f_i(\boldsymbol{\theta}) $$
    - Then:  $$ \frac{\partial}{\partial \boldsymbol{\theta}} L(\boldsymbol{\theta}, \lambda)= \frac{\partial}{\partial \boldsymbol{\theta}}J(\boldsymbol{\theta})-\sum_{i=1}^N\lambda_i \frac{\partial}{\partial \boldsymbol{\theta}}f_i(\boldsymbol{\theta})=\mathbf{0}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Illustrating the Lagrangian

    - Note: only one **active** constraint: $$f_1(\boldsymbol{\theta})=0$$
    - $\Rightarrow$ $$\lambda_1 \geq 0, \hspace{0.1cm} \lambda_2 = 0, \hspace{0.1cm} \lambda_3 = 0$$
    - Draw $\Rightarrow$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Karush-Kuhn-Tucker (KKT)

    - The SVM optimization objective is a nonlinear optimization task subject to a set of linear constraints.
    - Difficult to optimize, how do we konw if a solution is optimal?
    - The Karush-Kuhn-Tucker conditions is a well-known set of conditions for optimality in nonlinear optimization.
    - At $\boldsymbol{\theta}=\boldsymbol{\theta}_*$: $$\frac{\partial}{\partial \boldsymbol{\theta}} L(\boldsymbol{\theta}, \lambda) = \mathbf{0} \\ \lambda_i f_i(\boldsymbol{\theta}) = 0 \\ \lambda_i \geq 0$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Dual problem ($\lambda$)

    - Note: $$ \underset{\boldsymbol{\lambda} \geq 0}{\operatorname{max}}\ L(\boldsymbol{\theta}_*, \lambda) = J(\boldsymbol{\theta}_*)-\sum_{i=1}^N\lambda_i f_i(\boldsymbol{\theta}_*) $$
    - actually forces $\lambda_i f_i(\boldsymbol{\theta}_*)=0$ since $\lambda_i \geq 0$ and $f_i(\boldsymbol{\theta}_*) \geq 0$.
    - Hence: $$ \underset{\boldsymbol{\lambda} \geq 0}{\operatorname{max}}\ L(\boldsymbol{\theta}_*, \lambda) = J(\boldsymbol{\theta}_*)$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Primal versus dual problem

    - Putting it all together:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### SVM Lagrangian

    The Lagrangian for the SVM optimization problem is:

    $$
    L(\mathbf{w}, w_0, \lambda) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^N \lambda_i \left[ y_i (\mathbf{w}^T \mathbf{x}_i + w_0) - 1 \right]
    $$

    ---

    Taking the derivative with respect to $w_0$:

    $$
    \frac{\partial L}{\partial w_0} = -\sum_{i=1}^N \lambda_i y_i = 0
    $$

    ---

    Taking the derivative with respect to $\mathbf{w}$:

    $$
    \frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^N \lambda_i y_i \mathbf{x}_i = 0
    $$

    ---

    Constraint KKT:

    $$
    \sum_{i=1}^N \lambda_i y_i = 0
    $$

     and

    $$
    \lambda_i\left[ y_i (\mathbf{w}^T \mathbf{x}_i + w_0) - 1 \right]=0
    $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks

    - Let $y_i=1$ if $\lambda_i \neq 0$: $$\mathbf{w}^T\mathbf{x}_i+w_0=1$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Support vectors

    - Those $\mathbf{x_i}$ that are subject to $\lambda_i \neq 0$ are on the $\pm 1$ **margin**.
    - Only $\mathbf{x_i}$ in the set of support vectors (SV) are important.
    - Hence: $$\mathbf{w}=\sum_{\mathbf{x_i}\in\text{SV}}\lambda_i y_i \mathbf{x_i}$$
    - And for the bias: $$w_0$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Dual problem

    Remember: $$L(\mathbf{w}, w_0, \lambda) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^N \lambda_i \left[ y_i (\mathbf{w}^T \mathbf{x}_i + w_0) - 1 \right]$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks

    - Optimization is a science in itself (quadratic programming).
    - Sequential minimal optimization (SMO, Platt)
    - Dual problem: feature vectors enter loss in the form of an inner product.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Non-separable classes

    - Allow erros in training $\Rightarrow$
    - SVM objective: $$ \underset{\mathbf{w}, w_0}{\operatorname{min}}\ J(\mathbf{w}, w_0)\coloneqq \frac{1}{2}\| \mathbf{w} \|^2 $$
    - $$ \text{subject to} \hspace{0.2cm} y_i\left(\mathbf{w}^T \mathbf{x_i}+w_0\right) \geq 1$$
    - Dual formulation:
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Non-separable-classes example: two overlapping Gaussians whose
    # means are too close for any linear separator to get zero training
    # error — sets up the slack-variable / soft-margin motivation.
    n_nonsep = 100

    mu1_nonsep = np.array([1, 1])
    mu2_nonsep = np.array([2.3, 2.3])
    sigma_nonsep = np.array([[0.2, 0.0], [0.0, 0.2]])

    x_train_1_nonsep = np.random.multivariate_normal(mu1_nonsep, sigma_nonsep, n_nonsep)
    x_train_2_nonsep = np.random.multivariate_normal(mu2_nonsep, sigma_nonsep, n_nonsep)

    fig_nonsep, ax_nonsep = plt.subplots(figsize=(8, 8))
    ax_nonsep.scatter(
        x_train_1_nonsep[:, 0], x_train_1_nonsep[:, 1],
        s=120, facecolors="none", edgecolors="black", linewidth=3.0, label="Class 1",
    )
    ax_nonsep.scatter(
        x_train_2_nonsep[:, 0], x_train_2_nonsep[:, 1],
        s=120, facecolors="none", edgecolors="blue", linewidth=3.0, label="Class 2",
    )
    ax_nonsep.set_xlabel("x1")
    ax_nonsep.set_ylabel("x2")
    ax_nonsep.legend()

    mo.as_html(fig_nonsep)
    plt.close(fig_nonsep)
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn import svm
    return DecisionBoundaryDisplay, make_moons, svm


@app.cell
def _(DecisionBoundaryDisplay, make_moons, mo, plt, svm):
    # Kernel SVM on the two-moons dataset: the classes are not linearly
    # separable in 2-D input space, so we use a polynomial kernel.
    # Support vectors are highlighted with a red ring.
    X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)

    clf_moons = svm.SVC(kernel="poly", C=1.0, coef0=1, degree=4)
    clf_moons.fit(X_moons, y_moons)

    fig_moons, ax_moons = plt.subplots(figsize=(6, 6))
    DecisionBoundaryDisplay.from_estimator(
        clf_moons,
        X_moons,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax_moons,
        xlabel="x1",
        ylabel="x2",
    )
    ax_moons.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap=plt.cm.coolwarm,
                     s=20, edgecolors="k")
    ax_moons.scatter(
        clf_moons.support_vectors_[:, 0],
        clf_moons.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="Support Vectors",
    )
    ax_moons.legend()
    ax_moons.set_xticks(())
    ax_moons.set_yticks(())

    mo.as_html(fig_moons)
    plt.close(fig_moons)
    return


@app.cell
def _():
    from sklearn.datasets import load_wine
    return (load_wine,)


@app.cell
def _(load_wine, mo, np, plt):
    # Wine starter (intro to the SVM exercise): 2 of the 13 features
    # (alcohol, malic acid), all 3 classes. Targets are 0/1/2; legend
    # is shifted to 1/2/3 to match the original notebook's "Class N"
    # labeling.
    wine_data = load_wine()

    X_wine = wine_data.data[:, :2]
    feature_1_name_wine = "alcohol"
    feature_2_name_wine = "malic acid"
    y_wine = wine_data.target
    y_names_wine = np.unique(y_wine)
    colors_wine = ["black", "blue", "red"]

    fig_wine, ax_wine = plt.subplots(figsize=(5, 5))
    for class_value_wine, color_wine in zip(y_names_wine, colors_wine):
        ax_wine.scatter(
            X_wine[y_wine == class_value_wine, 0],
            X_wine[y_wine == class_value_wine, 1],
            s=120, facecolors="none",
            edgecolors=color_wine, linewidth=3.0,
            label=f"Class {class_value_wine+1}",
        )
    ax_wine.set_xlabel(feature_1_name_wine)
    ax_wine.set_ylabel(feature_2_name_wine)
    ax_wine.legend()

    mo.as_html(fig_wine)
    plt.close(fig_wine)
    return X_wine, feature_1_name_wine, feature_2_name_wine, y_wine


@app.cell
def _(
    DecisionBoundaryDisplay,
    X_wine,
    feature_1_name_wine,
    feature_2_name_wine,
    mo,
    plt,
    svm,
    y_wine,
):
    # Linear-kernel SVM on the 2-feature wine projection. Highlights
    # the support vectors with a red ring so students can see which
    # samples define the decision boundary.
    clf_wine = svm.SVC(kernel="linear", C=100.0)
    clf_wine.fit(X_wine, y_wine)

    print(clf_wine.support_vectors_)

    fig_wine_svm, ax_wine_svm = plt.subplots(figsize=(6, 6))
    DecisionBoundaryDisplay.from_estimator(
        clf_wine,
        X_wine,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax_wine_svm,
        xlabel=feature_1_name_wine,
        ylabel=feature_2_name_wine,
    )
    ax_wine_svm.scatter(X_wine[:, 0], X_wine[:, 1], c=y_wine, cmap=plt.cm.coolwarm,
                        s=20, edgecolors="k")
    ax_wine_svm.scatter(
        clf_wine.support_vectors_[:, 0],
        clf_wine.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="Support Vectors",
    )
    ax_wine_svm.legend()
    ax_wine_svm.set_xticks(())
    ax_wine_svm.set_yticks(())

    mo.as_html(fig_wine_svm)
    plt.close(fig_wine_svm)
    return


if __name__ == "__main__":
    app.run()
