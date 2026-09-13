# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo version of the Non-linear SVM / Kernel Methods lecture.
# Authored as a reactive Marimo app so the deck reads cleanly as a series
# of slides (one cell per concept, matching the density_estimation
# notebook's structure).
# Run locally with `marimo edit notebooks/06/kernel_methods.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/kernel_methods.slides.json",
    css_file="_shared/math.css",
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
    # Non-linear SVM
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - For the final part of non-linear classifiers, we will see how we can turn SVMs into non-linear classifiers.
    - Heavily depends on a field of research known as **kernel methods**.
    - A field of its own with lots of use cases throughout machine learning.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Remembering SVMs

    - Linear SVM: a hyperplane $g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$, trained by maximizing the margin.

    - **Dual formulation:** the weights are a combination of support vectors, $\mathbf{w} = \sum_{i \in SV} \lambda_i y_i \mathbf{x}_i$, and training becomes

    $$\max_{\lambda \geq 0} \; \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j \rangle$$

    - Subject to: $\sum_i \lambda_i y_i = 0$

    ---

    - **In testing:** $g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0 = \sum_{i \in SV} \lambda_i y_i \langle \mathbf{x}_i, \mathbf{x} \rangle + w_0$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with explicit mapping

    - One way to obtain a non-linear SVM: map the data into a feature space where it becomes linearly separable, and use a linear SVM there.

    - Let
    $$
    \mathbf{z} = \begin{bmatrix} x_1^2 \\ \sqrt{2} x_1 x_2 \\ x_2^2 \end{bmatrix}
    $$
    where
    $$
    \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
    $$

    ---

    - **Need:** $\mathbf{z}_i^T \mathbf{z}_j = x_{i1}^2 x_{j1}^2 + 2 x_{i1} x_{i2} x_{j1} x_{j2} + x_{i2}^2 x_{j2}^2 =$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with explicit mapping - training and testing

    - **Training:** insert $\mathbf{z}_i^T \mathbf{z}_j$ into the dual:

    $$\max_{\lambda \geq 0} \; \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j \, \mathbf{z}_i^T \mathbf{z}_j$$

    - **Key trick:** we never need $\mathbf{z}$ itself — only its inner products. Write

    $$\mathbf{z}_i^T \mathbf{z}_j = \underline{\hspace{3cm}}$$

    - **Testing:** $g(\mathbf{x}) = \sum_{i \in SV} \lambda_i y_i \; \underline{\hspace{3cm}} \; + w_0$

    - Can always find inner-product kernel $K(\mathbf{x}_i, \mathbf{x}_j)$!
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Mercer's theorem

    - A symmetric function $K(\mathbf{x}_i, \mathbf{x}_j)$ is a valid inner product
    $K(\mathbf{x}_i, \mathbf{x}_j) = \boldsymbol{\phi}(\mathbf{x}_i)^T \boldsymbol{\phi}(\mathbf{x}_j)$
    for some feature map $\boldsymbol{\phi}$ if and only if it is positive semi-definite.

    - $K(\mathbf{x}_i,\mathbf{x}_j)=$

    - **Reference:** J. Mercer, "Functions of positive and negative type, and their connection with the theory of integral equations," *Philosophical Transactions of the Royal Society A*, vol. 209, pp. 415–446, 1909.
        - See also B. Schölkopf and A. J. Smola, *Learning with Kernels*, MIT Press, 2002.

    - [Nice open access article on kernel methods for those who want to learn more.](https://arxiv.org/pdf/math/0701907)
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Kernels

    - **Polynomials:**  $K(\mathbf{x}_i, \mathbf{x}_j) = \left(\mathbf{x}_i^T \mathbf{x}_j + 1\right)^q \text{, where } q>0$

    - **RBF:**  $K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{1}{2\sigma^2} \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$

    - **Tanh:**  $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh\left(\beta\, \mathbf{x}_i^T \mathbf{x}_j + \gamma\right)$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Non-linear SVM

    - Choose $K(\mathbf{x}_i, \mathbf{x}_j)$

    - **Training:**  $\max_{\lambda \geq 0} \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$

    - **Test:**  $g(\mathbf{x}) = \sum_{i \in SV} \lambda_i y_i K(\mathbf{x}_i, \mathbf{x}) + w_0$
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Practical considerations for non-linear SVMs

    - Start simple -> linear kernel.
        - Only two hyperparameters to consider; slack variable and tolerance for stopping criterion.
    - Then -> non-linear kernel. An RBF kernel is the standard choice.
        - Added complexity; kernel width.
    - Use validation data to select hyperparameters.
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: Kernelizing the least sum of squares classifier

    - **Recap (linear classifiers):** the LS classifier minimizes

    $$J(\mathbf{w}) = \sum_{i=1}^N \left(y_i - \mathbf{w}^T \mathbf{x}_i\right)^2$$

    - Setting the gradient to zero gives the normal equations

    $$\left(\sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^T\right) \mathbf{w} = \sum_{i=1}^N \mathbf{x}_i y_i$$

    - The solution therefore lies in the span of the training data:

    $$\mathbf{w} = \sum_{i=1}^N \alpha_i \mathbf{x}_i$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: Kernelizing LS — derivation

    - Substitute $\mathbf{w} = \sum_{i=1}^N \alpha_i \mathbf{x}_i$ into the normal equations:

    $$\sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^T \sum_{j=1}^N \alpha_j \mathbf{x}_j = \sum_{i=1}^N \mathbf{x}_i y_i$$

    - Define the Gram matrix $\mathbf{K}$ with $K_{ij} = \mathbf{x}_i^T \mathbf{x}_j$. The equations become

    $$\mathbf{K} \boldsymbol{\alpha} = \underline{\hspace{3cm}}$$

    - Solve for the coefficients:

    $$\boldsymbol{\alpha} = \underline{\hspace{3cm}}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: Kernelizing LS — the kernel trick

    - **Prediction** is then expressed through the coefficients:

    $$g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} = \sum_{i=1}^N \alpha_i \, \underline{\hspace{3cm}}$$

    - Replace the inner product by a non-linear kernel $K(\mathbf{x}_i, \mathbf{x})$ and the LS classifier becomes non-linear.

    - Takeaway: any algorithm whose solution can be written in terms of inner products can be kernelized.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 12</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
