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
# Marimo version of the Neural Networks II lecture.
# Same content as notebooks/04/neural_networks_2.ipynb, but authored as
# a reactive Marimo app so each matplotlib figure stays tied to the data
# generation that produced it (one slide per concept, matching the
# density_estimation notebook's structure).
# Run locally with `marimo edit notebooks/04/neural_networks_2.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: in Jupyter the same name (e.g. `x`) can be redefined
# in any number of cells, but Marimo requires each name to be owned by
# exactly one cell. Below, every cell-local dataset and figure is given
# a unique suffix (`_step`, `_sigmoid_tanh`, `_moons`) so the slides
# stay independent and the data each figure displays is reproducible.

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks_2.slides.json",
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
    # Neural Networks II
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - We continue our work on neural networks.
    - This lecture focuses on some key components and challenges.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Activation functions

    - McCulloch - Pitts neuron and the Perceptron was motivated by neurons in the brain.
    - Utilised a step function to mimic a neuron "firing".
    - Not good for gradient descent $\Rightarrow$ replace with a differentiable approximation of the step function.
    - The "classic" activation function is the sigmoid activation function: $f_{\text{s}}(x) = \dfrac{1}{1+\exp(-x)}$
    - Nice derivative: $\frac{d f_{\text{s}}(x)}{dx}=f_{\text{s}}(x)(1-f_{\text{s}}(x))$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 20</div>
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Side-by-side: step activation, sigmoid activation, and the
    # sigmoid's derivative. Anchors the discussion of why a smooth
    # alternative to the step function is needed for gradient-based
    # training.
    x_step = np.linspace(-6, 6, 400)

    step_step = np.where(x_step >= 0, 1, 0)
    sigmoid_step = 1 / (1 + np.exp(-x_step))

    fig_step, (ax_step_step, ax_step_sig, ax_step_deriv) = plt.subplots(1, 3, figsize=(15, 4))
    ax_step_step.plot(x_step, step_step, color="blue")
    ax_step_step.set_title("Step Activation Function")
    ax_step_sig.plot(x_step, sigmoid_step, color="red")
    ax_step_sig.set_title("Sigmoid Activation Function")
    sigmoid_derivative_step = sigmoid_step * (1 - sigmoid_step)
    ax_step_deriv.plot(x_step, sigmoid_derivative_step, color="green")
    ax_step_deriv.set_title("Sigmoid Derivative")
    fig_step.tight_layout()

    plt.close(fig_step)
    mo.vstack(
        [
            mo.as_html(fig_step),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 20</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Backpropagation for MLP with one neuron in each layer

    - Consider MLP with 4 layers and 1 neuron in each layer.
    - Let us do Backpropagation with this network.

    <img src="media/simple_nn.png" width="1000px" />
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The vanishing gradient problem

    - As the number of layers increase, the number of $<1$ numbers get multiplied.
    - Gradients "vanish", training lower layers becomes difficult.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What about other activation functions?

    - Another classical choice:

    $$f_{\text{t}}(x) = \text{tanh}(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$

    - Also nice derivative:

    $$\frac{d f_{\text{t}}(x) }{dx}=1-f_{\text{t}}(x)^2$$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 20</div>
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Sigmoid and tanh plotted with their derivatives, side by side,
    # so students can compare magnitudes — tanh is centred around zero
    # which matters for the vanishing-gradient story later.
    x_st = np.linspace(-6, 6, 400)

    sigmoid_st = 1 / (1 + np.exp(-x_st))
    sigmoid_derivative_st = sigmoid_st * (1 - sigmoid_st)

    tanh_st = np.tanh(x_st)
    tanh_derivative_st = 1 - tanh_st ** 2

    fig_st, (ax_st_sig, ax_st_tanh) = plt.subplots(1, 2, figsize=(12, 4))
    ax_st_sig.plot(x_st, sigmoid_st, label="Sigmoid", color="red")
    ax_st_sig.plot(x_st, sigmoid_derivative_st, label="Sigmoid'", color="green", linestyle="--")
    ax_st_sig.set_title("Sigmoid Activation & Derivative")
    ax_st_sig.legend()
    ax_st_tanh.plot(x_st, tanh_st, label="Tanh", color="blue")
    ax_st_tanh.plot(x_st, tanh_derivative_st, label="Tanh'", color="orange", linestyle="--")
    ax_st_tanh.set_title("Tanh Activation & Derivative")
    ax_st_tanh.legend()

    plt.close(fig_st)
    mo.vstack(
        [
            mo.as_html(fig_st),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 20</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What about other activation functions?

    - Tanh has slightly better behavior than sigmoid, but still issues.
    - Modern activation functions handle this problem much better (covered in FYS3033).
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Gradient descent with momentum
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Momentum

    - Standard gradient descent can have slow convergence
        - Slow to train! Draw example $\Rightarrow$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 20</div>
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # Plain gradient descent on two loss landscapes. Left: a clean convex
    # bowl where plain SGD converges in a few steps. Right: a long flat
    # plateau where the gradient is tiny and roughly constant, so plain SGD
    # stalls — exactly why we introduce momentum on the next slide. Rendered
    # as a static figure so the slide displays reliably in the WASM export
    # (an interactive mo.state/button version rendered blank in the browser).
    def loss_smooth_mom(w):
        return (w - 3.0) ** 2

    def grad_smooth_mom(w):
        return 2.0 * (w - 3.0)

    def loss_plateau_mom(w):
        return np.where(
            w <= 1.0,
            -0.05 * w + 2.05,
            0.5 * (w - 3.0) ** 2,
        )

    def grad_plateau_mom(w):
        return np.where(w <= 1.0, -0.05, w - 3.0)

    n_steps_mom = 40
    gamma_mom = 0.3
    gamma_plateau_mom = 2.0
    w_init_mom = -5.0

    def _run_gd_mom(grad, gamma):
        w = w_init_mom
        traj = [w]
        for _ in range(n_steps_mom):
            w = w - gamma * grad(w)
            traj.append(w)
        return np.array(traj)

    traj1_mom = _run_gd_mom(grad_smooth_mom, gamma_mom)
    traj2_mom = _run_gd_mom(grad_plateau_mom, gamma_plateau_mom)

    w_grid_mom = np.linspace(-5.5, 5.5, 400)

    fig_mom, (ax_mom_s, ax_mom_p) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_mom_s.plot(w_grid_mom, loss_smooth_mom(w_grid_mom), color="steelblue")
    ax_mom_s.plot(
        traj1_mom, loss_smooth_mom(traj1_mom), "o-",
        color="orangered", markersize=4, label="SGD path",
    )
    ax_mom_s.scatter(
        [traj1_mom[-1]], [loss_smooth_mom(traj1_mom[-1])],
        s=80, color="orangered", zorder=5,
    )
    ax_mom_s.set_title(f"Smooth bowl  ({n_steps_mom} steps)")
    ax_mom_s.set_xlabel("w")
    ax_mom_s.set_ylabel("J(w)")
    ax_mom_s.legend()

    ax_mom_p.plot(w_grid_mom, loss_plateau_mom(w_grid_mom), color="steelblue")
    ax_mom_p.plot(
        traj2_mom, loss_plateau_mom(traj2_mom), "o-",
        color="orangered", markersize=4, label="SGD path",
    )
    ax_mom_p.scatter(
        [traj2_mom[-1]], [loss_plateau_mom(traj2_mom[-1])],
        s=80, color="orangered", zorder=5,
    )
    ax_mom_p.set_title(f"Long flat plateau  ({n_steps_mom} steps)")
    ax_mom_p.set_xlabel("w")
    ax_mom_p.set_ylabel("J(w)")
    ax_mom_p.legend()

    plt.close(fig_mom)
    mo.vstack(
        [
            mo.as_html(fig_mom),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 20</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Momentum

    - Idea: incorporate information from previous iteration. Keep the "momentum".
    - Reminder: $\mathbf{w}_j^l (t+1) = \mathbf{w}_j^l (t) - \gamma \frac{\partial}{\partial  \mathbf{w}_j^l} J$
    - Now, with momentum: $\Delta \mathbf{w}_j^l = \alpha \mathbf{w}_j^l (t) - \gamma \frac{\partial}{\partial  \mathbf{w}_j^l} J$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Momentum for T succesive iteration steps

    - Momentum effectively increases the learning constant.
    - Let $\frac{\partial}{\partial  \mathbf{w}_j^l} J = \mathbf{g}(t)$
    - $\Delta \mathbf{w}_j^l (T) = -\gamma \sum_{t=0}^{T-1} \alpha^t \mathbf{g}(T-t)+\alpha^T \Delta \mathbf{w}_j^l (0)$
    - Now, assume we are in a low curvature point of the loss function $\Rightarrow$ gradient approximately constant!
    - $\Delta \mathbf{w}_j^l (T) \approx -\gamma(1+\alpha+\alpha^2+ \cdots \alpha^{T-1})\mathbf{g}$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Beyond binary classification

    - So far, we have focused on binary classification (2 classes).
    - Fits well with sigmoid.
    - However, we often have more than 2 classes. In these cases, the sigmoid is less suitable.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The softmax function

    - The standard choice for multiclass classification is to use a softmax function in the output layer.
    - $\displaystyle \hat{y}_k = \dfrac{\exp(z_k^L)}{\sum_{k'} \exp(z_{k'}^L)}$
    - Guarantees that the output lies in the interval $[0, 1]$ and sums to 1.
    - Note: **one-hot encoding**
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Derivative of the softmax function

    - Need to know derivative of softmax function for Backpropagation and gradient descent.
    - Need to compute $\displaystyle \dfrac{\partial}{\partial z^L_m} \hat{y}_k = \dfrac{\partial}{\partial z^L_m} \dfrac{\exp(z_k^L)}{\sum_{k'} \exp(z_{k'}^L)}.$
    - Key is to realize that we have two cases: $m=k$ and $m\neq k$
    - For $m=k$: $\displaystyle \dfrac{\exp(z_k^L) \sum_{k'} \exp(z_{k'}^L)-\exp(z_k^L)\exp(z_m^L)}{\left(\sum_{k'} \exp(z_{k'}^L)\right)^2}$
    - For $m\neq k$: $\displaystyle \dfrac{-\exp(z_k^L)\exp(z_m^L)}{\left(\sum_{k'} \exp(z_{k'}^L)\right)^2}$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Cross-entropy loss

    - Can use a squared error like we have done in the past.
    - However, using the cross-entropy loss function is much more common: $J_{ce} -\sum_{k=1}^{k_L} y_k(i) \log (\hat{y}_k)$
    - Fits nicely with softmax, derivative of cross-entropy loss assuming softmax loss function and one-hot encoded labels.
    - Take derivative with respect to preactivation $z^L_m$. Key idea again, split sum into $m=k$ and $m\neq k$:
    - $\frac{\partial}{\partial z^L_m} J_{ce} = -\frac{\partial}{\partial z^L_m} y_k(i) \log (\hat{y}_k) -\frac{\partial}{\partial z^L_m} \sum_{k'\neq m}^{k_L}y_{k'}(i) \log (\hat{y}_{k'})$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Cross-entropy loss

    - Other benefits associated with the cross-entropy loss.
    - [See this book for a very nice and detailed explanation.](http://neuralnetworksanddeeplearning.com/chap3.html)
    - Can also be used for binary classification.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Weight initialization

    - Need a starting point for our parameters.
    - Many heuristics. Different choices for different activation functions. Will focus on sigmoid and tanh.
    - Key paper: Glorot and Bengio, Understanding the difficulty of training deep feedforward neural networks, 2010.
    - Avoid symmetry!
    - Xavier initialization:

    $$w_{jm}^l \sim \mathcal{U}\left(-\sqrt{\frac{6}{k_{l+1}+k_{l-1}}}, \sqrt{\frac{6}{k_{l+1}+k_{l-1}}}\right)$$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 20</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
