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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 13</div>
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
        
    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: Kernelizing the least sum of squares classifier

    - **Goal:** rederive the LS solution, then rewrite it so that the kernel trick applies.

    - Setup: samples $\mathbf{x}_i \in \mathbb{R}^d$ are **column vectors**, stacked as rows of the design matrix
    $$\mathbf{X} = \begin{bmatrix} \mathbf{x}_1^T \\ \vdots \\ \mathbf{x}_N^T \end{bmatrix} \in \mathbb{R}^{N \times d}, \qquad \mathbf{y} \in \mathbb{R}^N$$

    - We want a linear discriminant $g(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$ with $\mathbf{w} \in \mathbb{R}^d$.

    - The LS classifier minimizes the sum of squared errors:
    $$J(\mathbf{w}) = \sum_{i=1}^N \left(y_i - \mathbf{w}^T \mathbf{x}_i\right)^2 = \left\| \mathbf{y} - \mathbf{X} \mathbf{w} \right\|^2$$

    - Gradient with respect to $\mathbf{w}$:
    $$\frac{\partial J(\mathbf{w})}{\partial \mathbf{w}} = -2 \mathbf{X}^T \left( \mathbf{y} - \mathbf{X} \mathbf{w} \right)$$

    - Setting it to zero gives the **normal equations** and the LS solution:
    $$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{y} \quad \Longrightarrow \quad \mathbf{w} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: Kernelizing LS — represent the weights with the data

    - We look for a solution of the form $\mathbf{w} = \mathbf{X}^T \boldsymbol{\alpha}$ with $\boldsymbol{\alpha} \in \mathbb{R}^N$.

    - This follows from the **pseudo-inverse identity**
    $$\left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T = \mathbf{X}^T \left(\mathbf{X} \mathbf{X}^T\right)^{-1}$$

    - Applying it to the LS solution:
    $$\mathbf{w} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y} = \mathbf{X}^T \left(\mathbf{X} \mathbf{X}^T\right)^{-1} \mathbf{y} = \mathbf{X}^T \boldsymbol{\alpha}, \qquad \boldsymbol{\alpha} = \left(\mathbf{X} \mathbf{X}^T\right)^{-1} \mathbf{y}$$

    - The matrix $\mathbf{K} = \mathbf{X} \mathbf{X}^T \in \mathbb{R}^{N \times N}$ is the **kernel matrix**, $K_{ij} = \mathbf{x}_i^T \mathbf{x}_j$, so $\boldsymbol{\alpha} = \mathbf{K}^{-1} \mathbf{y}$.

    - The kernel then appears directly in the discriminant:
    $$g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} = \left(\mathbf{X}^T \boldsymbol{\alpha}\right)^T \mathbf{x} = \boldsymbol{\alpha}^T \mathbf{X} \mathbf{x} = \sum_{i=1}^N \alpha_i \, \mathbf{x}_i^T \mathbf{x} = \sum_{i=1}^N \alpha_i K(\mathbf{x}_i, \mathbf{x})$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: Kernelizing LS — a loss on the coefficients

    - Substitute $\mathbf{w} = \mathbf{X}^T \boldsymbol{\alpha}$ back into the LS loss. Since $\mathbf{X} \mathbf{w} = \mathbf{X} \mathbf{X}^T \boldsymbol{\alpha} = \mathbf{K} \boldsymbol{\alpha}$,
    $$J(\boldsymbol{\alpha}) = \left\| \mathbf{y} - \mathbf{X} \mathbf{w} \right\|^2 = \left\| \mathbf{y} - \mathbf{K} \boldsymbol{\alpha} \right\|^2 = \sum_{i=1}^N \left( y_i - [\mathbf{K} \boldsymbol{\alpha}]_i \right)^2$$

    - This is now a loss in the coefficients $\boldsymbol{\alpha}$, and the data enters only through $\mathbf{K}$.

    - Gradient with respect to $\boldsymbol{\alpha}$:
    $$\frac{\partial J(\boldsymbol{\alpha})}{\partial \boldsymbol{\alpha}} = -2 \mathbf{K}^T \left( \mathbf{y} - \mathbf{K} \boldsymbol{\alpha} \right) = -2 \mathbf{K} \left( \mathbf{y} - \mathbf{K} \boldsymbol{\alpha} \right)$$
    using that $\mathbf{K} = \mathbf{X} \mathbf{X}^T$ is symmetric, so $\mathbf{K}^T = \mathbf{K}$.

    - Setting the gradient to zero:
    $$\mathbf{K} \left( \mathbf{y} - \mathbf{K} \boldsymbol{\alpha} \right) = \mathbf{0} \quad \Longrightarrow \quad \mathbf{K}^2 \boldsymbol{\alpha} = \mathbf{K} \mathbf{y}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: Kernelizing LS — the solution

    - From $\mathbf{K}^2 \boldsymbol{\alpha} = \mathbf{K} \mathbf{y}$, if $\mathbf{K}$ is invertible we can cancel one factor:
    $$\mathbf{K} \boldsymbol{\alpha} = \mathbf{y} \quad \Longrightarrow \quad \boldsymbol{\alpha} = \mathbf{K}^{-1} \mathbf{y}$$

    - In general $\mathbf{K}$ may be singular. Then use the pseudo-inverse, $\boldsymbol{\alpha} = \mathbf{K}^{+} \mathbf{y}$, or regularize with a ridge term $\lambda > 0$:
    $$\boldsymbol{\alpha} = \left( \mathbf{K} + \lambda \mathbf{I} \right)^{-1} \mathbf{y}$$

    - The discriminant only needs inner products (kernels):
    $$g(\mathbf{x}) = \sum_{i=1}^N \alpha_i K(\mathbf{x}_i, \mathbf{x})$$

    - **Kernel trick:** replace $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$ by a general kernel. The linear kernel recovers the original LS classifier; e.g. an RBF kernel makes it non-linear.

    - Takeaway: any algorithm whose solution can be written in terms of inner products can be kernelized.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 13</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
