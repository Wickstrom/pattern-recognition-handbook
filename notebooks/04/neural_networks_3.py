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
# Marimo version of the Neural Networks III lecture.
# Same content as notebooks/04/neural_networks_3.ipynb, but authored as
# a reactive Marimo app so each matplotlib figure stays tied to the data
# generation that produced it (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/04/neural_networks_3.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `x_inner`) can be
# redefined in any number of cells, but Marimo requires each name to be
# owned by exactly one cell. Below, the cell-local dataset and figure
# use the `_polar` suffix so they stay independent of any future
# cells that might want similar names.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks_3.slides.json",
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
    # Neural Networks III
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - You have now learned about the fundamentals of neural networks.
    - The goal of this lecture it to elevate your understanding of neural networks.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The core idea of neural networks - transform data into a new representation.

    - Neural networks belong to the field of research called representation learning.
    - Very useful in cases where the original representation is difficult to work with.
        - Images, text, graphs, etc.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Two noisy concentric circles in Cartesian space and the same
    # samples in polar coordinates. The polar view collapses each
    # circle onto a thin horizontal band of radii, making the two
    # classes trivially separable — a concrete illustration of
    # representation learning.
    n_samples_polar = 200

    r_inner_polar = 1
    theta_inner_polar = 2 * np.pi * np.random.rand(n_samples_polar // 2)
    x_inner_polar = r_inner_polar * np.cos(theta_inner_polar) + 0.2 * np.random.randn(n_samples_polar // 2)
    y_inner_polar = r_inner_polar * np.sin(theta_inner_polar) + 0.2 * np.random.randn(n_samples_polar // 2)
    r_inner_polar_calc = np.sqrt(x_inner_polar ** 2 + y_inner_polar ** 2)
    theta_inner_polar_calc = np.arctan2(y_inner_polar, x_inner_polar)

    r_outer_polar = 2.5
    theta_outer_polar = 2 * np.pi * np.random.rand(n_samples_polar // 2)
    x_outer_polar = r_outer_polar * np.cos(theta_outer_polar) + 0.2 * np.random.randn(n_samples_polar // 2)
    y_outer_polar = r_outer_polar * np.sin(theta_outer_polar) + 0.2 * np.random.randn(n_samples_polar // 2)
    r_outer_polar_calc = np.sqrt(x_outer_polar ** 2 + y_outer_polar ** 2)
    theta_outer_polar_calc = np.arctan2(y_outer_polar, x_outer_polar)

    fig_polar, (ax_polar_xy, ax_polar_rt) = plt.subplots(1, 2, figsize=(12, 4))
    ax_polar_xy.scatter(x_inner_polar, y_inner_polar, color="blue", edgecolor="k",
                        s=80, label="Class 1")
    ax_polar_xy.scatter(x_outer_polar, y_outer_polar, color="red", edgecolor="k",
                        s=80, label="Class 2")
    ax_polar_xy.set_xlabel("x")
    ax_polar_xy.set_ylabel("y")
    ax_polar_xy.set_aspect("equal")
    ax_polar_xy.legend()
    ax_polar_rt.scatter(r_inner_polar_calc, theta_inner_polar_calc, color="blue",
                        edgecolor="k", s=80, label="Class 1")
    ax_polar_rt.scatter(r_outer_polar_calc, theta_outer_polar_calc, color="red",
                        edgecolor="k", s=80, label="Class 2")
    ax_polar_rt.set_xlabel("r")
    ax_polar_rt.set_ylabel("θ")
    ax_polar_rt.legend()
    fig_polar.tight_layout()

    mo.as_html(fig_polar)
    plt.close(fig_polar)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The core idea of neural networks - transform data into a new representation

    - Show video
    - [Visualizing Neural Networks with the Grand Tour](https://distill.pub/2020/grand-tour/)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## From scalars and sum to vectors and matrix multiplication

    - Our derivation of Backpropagation two lectures back looked at scalar derivatives.
    - Very little linear algebra and vector calculus.
    - Simple operation, but leads to complicated computations
        - A lot of indices and sums.
    - We will now continue the representation perspective to derive a much cleaner version of Backpropagation.
        - Bonus: also aligns much more closely to how the code should look!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### A new perspective of neural networks - modules of computation

    - Draw example.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Forward pass with linear algebra

    - Before we looked at the output of one neuron: $$ v_j^l = \mathbf{w}_j^l \mathbf{y}^{l-1}$$
    - Note: augmented space!
    - Now we look at output of the whole layer: $$ \mathbf{v}^l = \mathbf{W}^l \mathbf{y}^{l-1}$$

    ---

    - Reminder: $$ \mathbf{y}^{l-1} = \begin{bmatrix} y_1^{l-1} \\ y_2^{l-1} \\ \vdots \\ y_{k_{l-1}}^{l-1} \\ 1 \end{bmatrix} $$

    ---

    - And : $$ \mathbf{W}^l = \begin{bmatrix} w_{1,1}^l & w_{1,2}^l & \cdots & w_{1,k_{l-1}}^l & w_{1,0}^l \\ w_{2,1}^l & w_{2,2}^l & \cdots & w_{2,k_{l-1}}^l & w_{2,0}^l \\ \vdots    & \vdots    & \ddots & \vdots          \\ w_{k_l,1}^l & w_{k_l,2}^l & \cdots & w_{k_l,k_{l-1}}^l & w_{k_l,0}^l\end{bmatrix} $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Forward pass with linear algebra

    - Usually want to process a batch of samples.
    - Can also be done efficiently!
    - Let a batch of inputs be represented as $$ \mathbf{Y}^{l-1} = \begin{bmatrix} y_{1,1}^{l-1} & y_{1,2}^{l-1} & \cdots & y_{1,k_{l-1}}^{l-1} & 1 \\ y_{2,1}^{l-1} & y_{2,2}^{l-1} & \cdots & y_{2,k_{l-1}}^{l-1} & 1 \\ \vdots        & \vdots        & \ddots & \vdots              \\ y_{N,1}^{l-1} & y_{N,2}^{l-1} & \cdots & y_{N,k_{l-1}}^{l-1} & 1 \end{bmatrix} $$

    ---

    - Then: $$ \mathbf{V}^l = \mathbf{Y}^{l-1} (\mathbf{W}^l)^T $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Backward pass with linear algebra and vector calculus

    - Previously, for the output layer: $$\frac{\partial}{\partial \mathbf{w}_j^L} E (i) = \frac{\partial}{\partial \mathbf{w}_j^L} v_j^L (i) \frac{\partial}{\partial v_j^L (i) }E (i)$$
    - Want: $$ \frac{\partial J}{\partial \mathbf{W}^l}$$
    - Difficult, ends up with a vector by matrix derivate.
    - Start simpler: $$\frac{\partial}{\partial \mathbf{w}_j^L} \mathbf{v}^L \frac{\partial}{\partial \mathbf{v}^L}\mathbf{e}^T\mathbf{e}\frac{1}{2}$$
    - Where we assume one-hot encoded labels and $$\mathbf{e}=(f(\mathbf{v}^L)-\mathbf{y})$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Finding $\delta$ for output layer

    - We have dealt with the following term before $$\frac{\partial}{\partial \mathbf{v}^L}\mathbf{e}^T\mathbf{e}\frac{1}{2}=\mathbf{e}\frac{\partial}{\partial \mathbf{v}^L}(f(\mathbf{v}^L)-\mathbf{y})$$
    - $\mathbf{y}$ does not depend on $\mathbf{v}^L$, and we keep the derivative of $f(\mathbf{v}^L)$ general.
    - We have a vector by vector derivative $\Rightarrow$ Jacobian: $$ \frac{\partial f(\mathbf{v}^L)}{\partial \mathbf{v}^L} = \begin{bmatrix} \frac{\partial}{\partial v^L_1}f(v_1^L)  &  \frac{\partial}{\partial v^L_1} f(v_2^L) & \cdots & \frac{\partial}{\partial v^L_1} f(v^L_{k_{L}})  \\ \frac{\partial}{\partial v_2^L} f(v_1^L)  &  \frac{\partial}{\partial v^L_2} f(v_2^L)  & \cdots & \frac{\partial}{\partial v_2^L} f(v_2^L)  \\ \vdots    & \vdots    & \ddots & \vdots          \\ \frac{\partial}{\partial v_{k_L}} f(v_1^L)  &  \frac{\partial}{\partial v_{k_L}^L} f(v_2^L)  & \cdots & \frac{\partial}{\partial v_{k_L}} f(v_2^L)  \end{bmatrix} $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Finding $\delta$ for output layer

    - Can be compatctly represented as $$\frac{\partial}{\partial \mathbf{v}^L}\mathbf{e}^T\mathbf{e}\frac{1}{2} = \mathbf{e} \odot f'(\mathbf{v}^L) = \boldsymbol{\delta}^L$$
    - $\odot$ is the Hadamard product or elementwise multiplication.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Backward pass with linear algebra and vector calculus

    - Now, we turn to: $$ \frac{\partial}{\partial \mathbf{w}_j^L} \mathbf{v}^L $$
    - Vector by vector derivatve $\Rightarrow$ Jacobian matrix: $$ \begin{bmatrix} \frac{\partial}{\partial w_{1,1}^L}v^L_1  &  \frac{\partial}{\partial w_{1,1}^L} v^L_2 & \cdots & \frac{\partial}{\partial w_{1, 1}^L} v^L_{k_{L-1}}  \\ \frac{\partial}{\partial w_{1,2}^L} v^L_1  &  \frac{\partial}{\partial w_{1,2}^L} v^L_2  & \cdots & \frac{\partial}{\partial w_{1, 2}^L} v^L_{k_{L-1}}  \\ \vdots    & \vdots    & \ddots & \vdots          \\ \frac{\partial}{\partial w_{1, k_l}^L} v_1^L  &  \frac{\partial}{\partial w_{1, k_l}^L} v^L_2  & \cdots & \frac{\partial}{\partial w_{1, k_l}^L} v^L_{k_{L-1}}  \end{bmatrix} $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Derivative of one element in Jacobian

    - Reminder: $$ v^L_1 = w_{1,1}*y^{L-1}_1 + w_{1,2}*y^{L-1}_2 + \cdots w_{1,k_{L-1}}*y^{L-1}_{k_{L-1}} + 1*w_{1,0}$$
    - Therefore: $$ \frac{\partial}{\partial w_{1,1}}v^L_1 = \frac{\partial}{\partial w_{1,1}} w_{1,1}*y^{L-1}_1 + \frac{\partial}{\partial w_{1,1}} w_{1,2}*y^{L-1}_2 + \cdots \frac{\partial}{\partial w_{1,1}} w_{1,k_{L-1}}*y^{L-1}_{k_{L-1}} + 1*w_{1,0}$$

    ---

    - Back to Jacobian: $$ \begin{bmatrix} y_1^{L-1} & 0 & \cdots & 0 \\ y_2^{L-1} & 0 & \cdots & 0 \\ \vdots    & \vdots & \ddots & \vdots \\ y_{k_l}^{L-1} & 0 & \cdots & 0 \end{bmatrix} $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Putting it all together

    - Combing both derivatives give: $$ \begin{bmatrix} y_1^{L-1} & 0 & \cdots & 0 \\ y_2^{L-1} & 0 & \cdots & 0 \\ \vdots    & \vdots & \ddots & \vdots \\ y_{k_l}^{L-1} & 0 & \cdots & 0 \end{bmatrix} \begin{bmatrix} \delta_1^{L} \\ \delta_2^{L} \\ \vdots \\ \delta_{k_{L}}^{L} \end{bmatrix} $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Putting it all together

    - Take a step back. Derivative of loss with respect to neuron 1 gave non-zero elements in column 1.
    - If we repeat process of derivative with resepct to neuron $j$, coulmn $j$ will be non-zero.
    - We can get all derivatives with one matrix operation:
    - $$ \frac{\partial J}{\partial \mathbf{W}^L} = (\mathbf{y}^{L-1})^T \boldsymbol{\delta}^L$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What have we gained?

    - A lot of work, but clean result with simple rule.
    - To find gradients of loss with respect to weights in layer $l$, look at $\delta$ from current layer and output from previous layer.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## A final touch on neural networks

    - Many nice results for linear classifiers.
        - Optimal in terms of e.g. maximum likelihood.
        - Guarantueed unique solution for SVM.
    - Hard to obtain similar results for neural networks due to their complexity.
    - However, work is ongoing!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Connecting SVMs and neural networks

    - The big question in neural network research:
        - Why do they generalize so well?
        - Seems to defy traditional statistics.
    - One of many directions to answer this question:
        - Generalization due to implicit bias in gradient descent algorithm.
    - [Interesting work by Soundry et al (2018)](https://www.jmlr.org/papers/volume19/18-188/18-188.pdf)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Connecting SVMs and neural networks

    - Setting:
        - Neural network with one layer.
        - Binary classification.
        - Separable classes (similar results have been shown for non-separable classes).
        - Also some assumptions on loss function.
        """
    )
    return


@app.cell
def _(mo):
    # Setting figure from Soudry et al. — the optimisation trajectory
    # in the (w₁, w₂) plane that motivates the gradient-descent /
    # hard-margin-SVM equivalence in the next slide.
    mo.image(src="media/soundry.png", width="700px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Connecting SVMs and neural networks

    - What happens when we let the number of iterations tend towards infinity?
    - Remarkably, we obtain the weights of the hard margin SVM!
    - They also show how the KKT conditions "naturally" fall out of this analysis.
        """
    )
    return


@app.cell
def _(mo):
    # Companion figure — same setting after many iterations; the
    # weight vector aligns with the max-margin SVM solution and the
    # KKT-supporting samples are highlighted on the data.
    mo.image(src="media/soundry2.png", width="700px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Wrapping up neural networks

    - Neural networks are powerful and versatile tools.
    - You have now learned the fundamentals. A lot more to be said.
        - However, this foundation often applies also for bigger and more modern neural networks.
    - This course is your chance to code a neural network from scratch, take that opportunity!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Programming exercises

    Below are programming exercises associated with this lecture. These cell blocks are starting points that load the data and prepare the problem such that you can get going with the implementation. There are also theoretical exercises, but due to copyright we cannot share them here. They will be made available in a private repository connected to the course.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Wine classification with neural networks.

    This problem is focused on multi-class classification of wine types based on 13 features. Below, we have simplified the problem by extracting 2 features. Take this as a starting point, but when you feel confident in your implementation, train your classifier on all 13 features. Does performance change?
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_wine
    return (load_wine,)


@app.cell
def _(load_wine, mo, np, plt):
    # Wine starter: 2 of the 13 features (alcohol, malic acid), all 3
    # classes. Targets are 0/1/2; legend is shifted to 1/2/3 to match
    # the original notebook's "Class N" labeling.
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
    return


if __name__ == "__main__":
    app.run()
