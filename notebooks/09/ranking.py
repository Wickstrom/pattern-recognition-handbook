# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///
#
# Marimo version of the Ranking lecture.
# Same content as notebooks/09/ranking.ipynb, but authored as a reactive
# Marimo app (one slide per concept, matching the density_estimation
# notebook's structure).
# Run locally with `marimo edit notebooks/09/ranking.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/ranking.slides.json",
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
    # Ranking
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction

    - Ranking is a fundamental task in data analysis.
    - Not only content, but link between content!
    - Most famous example, Google web-page ranking
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example

    - Consider the following directed graph
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Idea 1

    - Count backlinks
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Idea 2

    - Count ranks associated with backlinks
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarks to idea 2

    - Ranking becomes maths!
    - Unlogical "recommendation" process
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Idea 3

    - Multiple recommendations sum to one
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Idea 3 in matrix-vector form

    - Let $$ \mathbf{H}^T = \begin{bmatrix} 0 & 0 & 1 & \frac{1}{2} \\ \frac{1}{3} & 0 & 0 \\ \frac{1}{3} & \frac{1}{2} & 0 & \frac{1}{2} \\ \frac{1}{3} & \frac{1}{2} & 0 & 0 \end{bmatrix}$$
    - and $$ \boldsymbol{\Pi} = \begin{bmatrix} r_1 \\ r_2 \\ r_3 \\ r_4 \end{bmatrix}$$
    - Hence:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Note on idea 3

    - $\boldsymbol{\Pi}$ is an eigenvector of $\mathbf{H}^T$ for $\lambda = 1$
    - World's most famous eigenvector
    - Google PageRank algorithm (with some modifications)
    - Solution: $$ \boldsymbol{\Pi} = \begin{bmatrix} 0.4 \\ 0.1 \\ 0.3 \\ 0.2 \end{bmatrix}$$
    - Alternatively:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Google as a Markov chain

    - $\mathbf{H}$ is special!
        - Transition matrix in (homogeneous, discrete) Markov chain
    - $$ \mathbf{H} = \begin{bmatrix} 0 & \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \\ 0 & 0 & \frac{1}{2} & \frac{1}{2} \\ 1 & 0 & 0 & 0 \\ \frac{1}{2} & 0 & \frac{1}{2} & 0 \end{bmatrix}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Google as a Markov chain

    - $\boldsymbol{\Pi}$ has a special meaning
    - Theorem: If $\mathbf{H}: H_{ij}=p_{ij}$ is a transition matrix and $0 < p_{ij} < 1$ then
        - $\lambda_{max}=1$
        - $\boldsymbol{\Pi}_{max}$ is unique, positive, sums to one.
        - $\boldsymbol{\Pi}_{max}$ can always be found by the power method!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Markov chain example

    - $$ \mathbf{A}(\mathbf{H}^T) = \begin{bmatrix} 0.3 & 0.4 & 0.5 \\ 0.3 & 0.4 & 0.3 \\ 0.4 & 0.2 & 0.2 \end{bmatrix}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Markov chain example

    - Find:
        - P(start @ C, end @ B after 2 deliveries)
    - $P(CA)P(AB)+P(CB)P(BB)+P(CC)P(CB)=0.33$
    - $$ \begin{bmatrix} 0.41 & 0.38 & 0.37 \\ 0.33 & 0.34 & 0.33 \\ 0.26 & 0.28 & 0.30 \end{bmatrix} = \mathbf{A}\mathbf{A}=\mathbf{A}^2$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Markov chain example

    - Probabilities after 5 and 6 steps:
    - $$ \mathbf{A}^5 \approx \mathbf{A}^6 \approx \begin{bmatrix} 0.39 & 0.39 & 0.39 \\ 0.33 & 0.33 & 0.33 \\ 0.28 & 0.28 & 0.28 \end{bmatrix} $$
    - Converges!
    - Stationary distribution.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Markov chain example

    - Eigenvector:
        - Let $\mathbf{x}^{(0)}$ be our starting rank.
    - Update rank for each step in the chain.
    - Show ->
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The power method

    - Iterative; finds dominant eigenvector
    - Scaling after iteration in general.
    - Markov example:
        - $\mathbf{x}^T = [0.39, 0.33, 0.28]$ is the stationary distribution off the states in the chain.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Idea 4

    - Modify $\mathbf{H}$ such that $0 < p_{ij} < 1$.
    - Modification 1; Dangling nodes (no outlinks) creates zero rows in $\mathbf{H}$
    - $$ \mathbf{H}^T = \begin{bmatrix} 0 & \frac{1}{4} & \frac{1}{4} & \frac{1}{4} & \frac{1}{4} \\ 0 & 0 & \frac{1}{2} & \frac{1}{2} & 0 \\ 1 & 0 & 0 & 0 & 0 \\ \frac{1}{2} & 0 & \frac{1}{2} & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$
    - Replace zero row with row of 1/n probablities.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Modification 2

    - Google matrix: $$ \mathbf{G} = \alpha \mathbf{S} + (1-\alpha)\mathbf{E}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interpretation $(\alpha=0.85)$

    - 85% of the time, surfer follow links.
    - 15% of the time, surfer types URL (teleportation).
    - Personalization:
        - $\frac{1}{n}\mathbf{e}\mathbf{e}^T \rightarrow \mathbf{e}\mathbf{v}^T$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Final remarks

    - $\mathbf{H}$ is sparse (good).
    - $\mathbf{G}$ is dense but function of $\mathbf{H}$
    - Power method is nice! (no inverse)
    - Early report from Google indicated 50 iterations.
    - Updated on a frequent basis.
        """
    )
    return


if __name__ == "__main__":
    app.run()
