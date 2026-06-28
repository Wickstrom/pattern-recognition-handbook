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
    - Not good for gradient descent $\Rightarrow $replace with a differentiable approximation of the step function.
    - The "classic" activation function is the sigmoid activation function: $f_{\text{s}}(x) = \dfrac{1}{1+\exp(-x)}$
    - Nice derivative: $\frac{d f_{\text{s}}(x)}{dx}=f_{\text{s}}(x)(1-f_{\text{s}}(x))$
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

    mo.as_html(fig_step)
    plt.close(fig_step)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Backpropagation for MLP with one neuron in each layer

    - Consider MLP with 4 layers and 1 neuron in each layer.
    - Let us do Backpropagation with this network.
        """
    )
    return


@app.cell
def _(mo):
    mo.image(src="media/simple_nn.png", width="800px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The vanishing gradient problem

    - As the number of layers increase, the number of$1>$numbers get multiplied.
    - Gradients "vanish", training lower layers becomes difficult.
        """
    )
    return


@app.cell
def _(mo):
    mo.image(src="media/simple_nn.png", width="800px")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What about other activation functions?

    - Another classical choice: $f_{\text{t}}(x) = \text{tanh}(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$
    - Also nice derivative: $\frac{d f_{\text{t}}(x) }{dx}=1-f_{\text{t}}(x)^2$
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

    mo.as_html(fig_st)
    plt.close(fig_st)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What about other activation functions?

    - Tanh has slightly better behavior than sigmoid, but still issues.
    - Modern activation functions handle this problem much better (covered in FYS3033).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Gradient descent with momentum
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
        """
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
    - Now, assume we are in a low curvature point of the loss function $\Rightarrow $gradient approximately constant!
    - $\Delta \mathbf{w}_j^l (T) \approx -\gamma(1+\alpha+\alpha^2+ \cdots \alpha^{T-1})\mathbf{g}$
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
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The softmax function

    - The standard choice for multiclass classification is to use a softmax function in the output layer.
    - $\hat{y}_k = \frac{\exp(v_k^L)}{\sum_{k'} \exp(v_{k'}^L)}$
    - Guarantees that the output lies in the interval$[0, 1]$and sums to 1.
    - Note: **one-hot encoding**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Derivative of the softmax function

    - Need to know derivative of softmax function for Backpropagation and gradient descent.
    - Need to compute $\frac{\partial}{\partial v^L_m} \hat{y}_k =  \frac{\partial}{\partial v^L_m} \frac{\exp(v_k^L)}{\sum_{k'} \exp(v_{k'}^L)}.$
    - Key is to realize that we have two cases: $m=k $and $m\neq k$
    - For $m=k$: $\frac{\exp(v_k^L) \sum_{k'} \exp(v_{k'}^L)-\exp(v_k^L)\exp(v_m^L)}{(\sum_{k'} \exp(v_{k'}^L))^2}$
    - For $m\neq k$: $\frac{-\exp(v_k^L)\exp(v_m^L)}{(\sum_{k'} \exp(v_{k'}^L))^2}$
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
    - Take derivative with respect to preactivation $v^L_m$. Key idea again, split sum into $m=k $and $m\neq k$:
    - $\frac{\partial}{\partial v^L_m} J_{ce} = -\frac{\partial}{\partial v^L_m} y_k(i) \log (\hat{y}_k) -\frac{\partial}{\partial v^L_m} \sum_{k'\neq m}^{k_L}y_{k'}(i) \log (\hat{y}_{k'})$
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
    - Xavier initialization: $w_{jm}^l \sim \mathcal{U}\left(-\sqrt{\frac{6}{k_{l+1}+k_{l-1}}}, \sqrt{\frac{6}{k_{l+1}+k_{l-1}}}\right)$
        """
    )
    return


if __name__ == "__main__":
    app.run()
