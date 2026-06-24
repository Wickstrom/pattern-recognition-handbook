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
# Same content as notebooks/06/kernel_methods.ipynb, but authored as a
# reactive Marimo app so the deck reads cleanly as a series of slides
# (one cell per concept, matching the density_estimation notebook's
# structure).
# Run locally with `marimo edit notebooks/06/kernel_methods.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/kernel_methods.slides.json",
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
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Remembering SVMs

    - **Recall:**  From $L(\mathbf{w}, w_0)$: $\mathbf{w} = \sum_{i \in SV} \lambda_i y_i \mathbf{x}_i$

    - **Dual:**  $ \max_{\lambda \geq 0} \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j \rangle$

    - Subject to: $ \sum_i \lambda_i y_i = 0$
    - Note: $\langle \mathbf{x}_i, \mathbf{x} \rangle$ denotes the inner product

    ---

    - **In testing:** $ g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0 = \sum_{i \in SV} \lambda_i y_i \langle \mathbf{x}_i, \mathbf{x} \rangle + w_0 $
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with explicit mapping

    - Explicitly map $\mathbf{x}$, then use linear SVM.

    - Let $ \mathbf{z} = \begin{bmatrix} x_1^2 \\ \sqrt{2} x_1 x_2 \\ x_2^2 \end{bmatrix} $ where $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$

    ---

    - **Need:** $\mathbf{z}_i^T \mathbf{z}_j = x_{i1}^2 x_{j1}^2 + 2 x_{i1} x_{i2} x_{j1} x {j2} + x_{i2}^2 x_{j2}^2 = $
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with explicit mapping - training and testing

    - **Training:**: $ \sum_{i=1}^N \lambda_i + \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j \mathbf{z}_i^T \mathbf{z}_j $

    - $ \sum_{i=1}^N \lambda_i + \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j K(\mathbf{x}_i,\mathbf{x}_j) $

    - **Testing:**  $ g(\mathbf{z}) = \sum_{\mathbf{z}_i \in SV} \lambda_i y_i \mathbf{z}_i^T \mathbf{z}_j$

    - $ g(\mathbf{x}) = \sum_{\mathbf{x}_i \in SV} \lambda_i y_i K(\mathbf{x}_i, \mathbf{x})$

    - Can always find inner-product kernel $K(\mathbf{x}_i, \mathbf{x}_j)$!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Mercer's theorem

    - $K(\mathbf{x}_i,\mathbf{x}_j) = $

    - [Nice open access article on kernel methods for those who want to learn more.](https://arxiv.org/pdf/math/0701907)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Kernels

    - **Polynomials:**  $ K(\mathbf{x}_i, \mathbf{x}_j) = \left(\mathbf{x}_i^T,\mathbf{x}_j+1\right)^q \text{, where } q>0$

    - **RBF:**  $ K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{1}{2\sigma^2} \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$

    - **Tanh:**  $ K(\mathbf{x}_i, \mathbf{x}_j) = \tanh\left(\beta\, \mathbf{x}_i^T \mathbf{x}_j + \gamma\right)$
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

    - **Test:**  $g(\mathbf{x}) = \sum_{i \in SV} \lambda_i y_i K(\mathbf{x}_i, \mathbf{x}) +w_0$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Non-linear SVM as a network
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Non-separable classes

    - Remember: $ \mathbf{w}^T \mathbf{x} + w_0  \geq 1-\gamma$
    - Both classes: $ y_i \left(\mathbf{w}^T \mathbf{x}_i + w_0\right)  \geq 1-\gamma_i$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Non-separable classes - in practice

    - Don't want too many $\gamma_i > 0$
    - Minimize: $J(\mathbf{w}, w_0, \gamma_i) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \gamma_i$
    - Subject to: $ y_i \left( \mathbf{w}^T \mathbf{x}_i + w_0 \right) \geq 1 - \gamma_i$

    $$ \gamma_i \geq 0 $$

    ---

    - **Dual:** $ \max_{\lambda \geq 0} \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j \rangle $

    - Subject to: $ \sum_i \lambda_i y_i = 0$

    $$ 0 \leq \lambda_i \leq C $$
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
        """
    )
    return


if __name__ == "__main__":
    app.run()
