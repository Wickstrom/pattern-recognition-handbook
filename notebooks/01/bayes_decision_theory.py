# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "plotly",
#     "ucimlrepo",
# ]
# ///
#
# Marimo version of the Bayes decision theory lecture.
# Same content as notebooks/01/bayes_decision_theory.ipynb, but authored as
# a reactive Marimo app so the interactive figures use real widgets
# (sliders, dropdowns) instead of Plotly's baked-in animation frames.
# Run locally with `marimo edit notebooks/01/bayes_decision_theory.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/bayes_decision_theory.slides.json",
)


@app.cell
def _():
    import os
    from pathlib import Path

    # Make `mo.image(src="media/foo.png")` and the spambase fetch work
    # regardless of which directory `marimo edit` is launched from. We chdir
    # to the notebook's own directory so the relative `media/` path resolves
    # the same way it does for the deployed WASM build (where the page sits
    # next to `media/`). In WASM, `__file__` may not be set and the chdir
    # becomes a no-op — the browser still fetches the images via the page URL.
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
    # Bayes decision theory

    ## Introduction

    - In this lecture we will cover the following topics:
        - Bayesian decision theory
        - The naïve Bayes assumption
        - Bayes classifier minimizes probability of error
        - Risk minimization
        - Discriminant functions and decision surfaces
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ## Spam

    - It's the early 2000, and spam emails offering you to buy Viagra are flourishing.
    - Cannot just filter on keywords.
    - Mathematically, we want to find $P(\text{spam}|\text{content})$.
                """
            ),
            mo.hstack(
                [
                    mo.image(src="media/endingspam.png", width="200px"),
                    mo.image(src="media/spam_example.png", width="400px"),
                ],
                justify="start",
                gap=2,
            ),
        ],
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Tokenization and learning

    - Break email into words $W_i$
    - Discrete events: $P(W_i)\geq 0$
    - Notation
        - $S$: Spam (illegitimate messages)
        - $H$: Ham (legitimate messages)
    - How to get an expression for $P(\text{spam}|\text{content})$?
    - Bayes' rule:
        - $P(S|W_i)=\dfrac{P(W_i|S)P(S)}{P(W_i)}$
    - where:
        - $P(W_i)=P(W_i|S)P(S)+P(W_i|H)P(H)$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How to learn from data

    - Collect Spam and Ham and learn (estimate) $P(W_i|S)$ and $P(W_i|H)$!
    - Example: $W_i = \text{"Money"}$
    - Let us look at the data — pick which word to inspect below.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo, norm, np, plt


@app.cell
def _(fetch_ucirepo, np):
    spambase = fetch_ucirepo(id=94)  # Spambase dataset from UCI Machine Learning Repository
    X = spambase.data.features  # features of the dataset
    y = spambase.data.targets  # targets of the dataset

    ham_indices = np.where(y == 0)[0]
    spam_indices = np.where(y == 1)[0]

    # Pick a small handful of words that illustrate the spam/ham asymmetry.
    # (The 57 features in Spambase are sorted by frequency of each word /
    # character; the indices below are the standard ones for "money" and
    # "george".)
    word_index_map = {
        "money (idx 23)": 23,
        "george (idx 26)": 26,
    }
    return X, ham_indices, spam_indices, word_index_map


@app.cell
def _(X, ham_indices, mo, plt, spam_indices, word_index_map):
    # Pre-compute one figure per word and bundle them into tabs so the widget
    # and the visualization live in a single cell (one slide in slides view).
    word_tabs = {}
    for word_label, idx in word_index_map.items():
        word_name = word_label.split(" ")[0]
        ham_mean = float(X.iloc[ham_indices, idx].mean())
        spam_mean = float(X.iloc[spam_indices, idx].mean())

        fig_words, axes_words = plt.subplots(1, 2, figsize=(10, 3.5))

        axes_words[0].bar(
            [0, 1],
            [ham_mean, spam_mean],
            color=["#4c72b0", "#c44e52"],
        )
        axes_words[0].set_xticks([0, 1])
        axes_words[0].set_xticklabels(["Ham", "Spam"])
        axes_words[0].set_ylabel(f"Avg. frequency of '{word_name}'")
        axes_words[0].set_title("Spambase — mean occurrence")

        axes_words[1].hist(
            X.iloc[spam_indices, idx],
            bins=30, alpha=0.6, color="#c44e52", label="Spam",
        )
        axes_words[1].hist(
            X.iloc[ham_indices, idx],
            bins=30, alpha=0.6, color="#4c72b0", label="Ham",
        )
        axes_words[1].set_xlabel(f"Frequency of '{word_name}'")
        axes_words[1].set_ylabel("# emails")
        axes_words[1].legend()
        axes_words[1].set_title("Distribution")

        fig_words.tight_layout()

        word_tabs[word_name] = mo.vstack(
            [
                mo.md(
                    f"**Spam mean:** {spam_mean:.3f} &nbsp;&nbsp; "
                    f"**Ham mean:** {ham_mean:.3f}"
                ),
                mo.as_html(fig_words),
            ]
        )
        plt.close(fig_words)

    mo.ui.tabs(word_tabs)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Detection

    - Need $P(W_1 \cap W_2 \cap \cdots \cap W_K|S)$
    - Difficult!
        - Think about what this term is actually saying.
    - Need to simplify.
    - Note: modern large language models are modeling something very similar to the expression above.
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.image(src="media/gpt.png", width="600px"),
            mo.md(
                r"""
    **References:**

    - Radford et al. — *Improving Language Understanding by Generative Pre-Training*, 2018.
    - Bengio et al. — *A Neural Probabilistic Language Model*, 2003.
                """
            ),
        ],
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Naïve Bayes assumption

    - Assume independence!
    - $P(A \cap B)=P(A)P(B)$
    - Then we can compute $P(S|W_1 \cap W_2 \cap \cdots \cap W_K)$:

    $$P(S|W_1,\dots,W_K) = \frac{P(S)\prod_i P(W_i|S)}{P(W_1,\dots,W_K)}$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Bayes classifier

    - Shifting focus a bit.
    - How can we think more generally about classification using Bayes' rule?
    - First: think continuous random variables $\rightarrow$
    - Let us start with considering two classes $w_1$ and $w_2$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Core idea of Bayes classifier

    - Given a new sample $\mathbf{x}$, how can we decide if we want to classify it as belonging to class $w_1$?
    - What about:
        - $P(w_1|\mathbf{x}) > P(w_2|\mathbf{x})$?
    - else class $w_2$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Leveraging Bayes' rule

    - $P(w_1|\mathbf{x})$ is not straightforward to model.
    - Bayes' rule to the rescue:
        - $P(w_1)P(\mathbf{x}|w_1) > P(w_2)P(\mathbf{x}|w_2)$
    - Much easier to work with $\rightarrow$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with balanced dataset

    - Consider the case where $P(w_1)=P(w_2)=0.5$.
    - Draw $\rightarrow$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with unbalanced dataset

    - Consider the case where $P(w_1)=0.25$ and $P(w_2)=0.75$.
    - Draw $\rightarrow$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with overlapping Gaussians

    - Each tab is a different parameter setting — read the label to see exactly
      which parameters are in play. Means, standard deviations, and (in the last
      tab) the prior can all change.
    - Click the **"decision boundary"** entry in the legend to toggle it on or
      off. It starts **off** so the densities stand alone first; turn it on to
      see where the Bayes classifier would assign each $x$.
    - **Question for the class:** For each setting, predict where the decision
      boundary should fall *before* you toggle it on. Then check.
        """
    )
    return


@app.cell
def _():
    # Four parameter settings. Each one varies a single quantity from the
    # baseline (μ₁=0, μ₂=1, σ=0.5, P(w₁)=0.5) so students can isolate its
    # effect. Tab labels state the exact parameter values used in that figure.
    # Format: (label, mu1, mu2, sigma1, sigma2, prior_w1)
    settings = [
        ("μ₁=0, μ₂=1, σ₁=σ₂=0.5", 0.0, 1.0, 0.5, 0.5, 0.5),
        ("μ₁=0, μ₂=2, σ₁=σ₂=0.5", 0.0, 2.0, 0.5, 0.5, 0.5),
        ("μ₁=0, μ₂=1, σ₁=0.5, σ₂=1.5", 0.0, 1.0, 0.5, 1.5, 0.5),
        ("μ₁=0, μ₂=1, σ=0.5, P(w₁)=0.1", 0.0, 1.0, 0.5, 0.5, 0.1),
    ]
    return (settings,)


@app.cell
def _(mo, np, norm, settings):
    import plotly.graph_objects as go

    x_grid = np.linspace(-5, 6, 600)

    sample_tabs = {}
    for label, mu1, mu2, sigma1, sigma2, prior_w1 in settings:
        p1 = norm.pdf(x_grid, mu1, sigma1)
        p2 = norm.pdf(x_grid, mu2, sigma2)

        # Bayes decision boundary: where P(w₁)·P(x|w₁) == P(w₂)·P(x|w₂).
        # With equal priors this is just the crossing of the likelihoods.
        # With a skewed prior the boundary shifts toward the class that has
        # the lower prior — more evidence is needed before we assign to it.
        post1 = prior_w1 * p1
        post2 = (1.0 - prior_w1) * p2
        diff = post1 - post2
        crossings = np.where(np.diff(np.sign(diff)))[0]
        boundary = float(x_grid[crossings[0]]) if len(crossings) else None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_grid, y=p1, mode="lines",
            name="P(x|w₁)",
            line=dict(color="royalblue", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=x_grid, y=p2, mode="lines",
            name="P(x|w₂)",
            line=dict(color="crimson", width=2.5),
        ))

        # Decision boundary as a toggleable trace — the "decision boundary"
        # legend entry starts hidden (visible="legendonly") so the densities
        # read cleanly first; clicking the legend entry toggles it on.
        if boundary is not None:
            max_y = float(max(p1.max(), p2.max()))
            fig.add_trace(go.Scatter(
                x=[boundary, boundary],
                y=[0, max_y * 1.05],
                mode="lines",
                line=dict(color="black", dash="dash", width=2),
                name="decision boundary",
                visible="legendonly",
            ))

        fig.update_layout(
            height=440, width=950,
            margin=dict(t=10, b=40, l=40, r=20),
            xaxis=dict(title="x", showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        sample_tabs[label] = fig

    mo.ui.tabs(sample_tabs)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Probability of error

    - Want:
        - Algorithm that can solve task with high performance.
        - Think spam filter that removes all spam and keeps all ham.
    - Problem:
        - Real data is complicated, we will make mistakes!
        - What is the probability of making an error with a Bayes classifier?
    - The probability of error can be modelled as follows:

    $$
    \begin{aligned}
    P_e &= P(\mathbf{x}\in R_1, w_2) + P(\mathbf{x}\in R_2, w_1) \\
        &= P(\mathbf{x}\in R_1(w_2))P(w_2) + P(\mathbf{x}\in R_2(w_1))P(w_1)
    \end{aligned}
    $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Bayes classifier minimizes $P_e$

    - Note that $P_e$ can also be expressed in terms of an integral:

    $$P_e = \displaystyle\int_{R_1} P(w_2)P(\mathbf{x}|w_2)\,d\mathbf{x} + \int_{R_2} P(w_1)P(\mathbf{x}|w_1)\,d\mathbf{x}$$

    - Visual proof $\rightarrow$
    - Formal proof as extra exercise.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Going beyond the two-class setting

    - Given a new sample $\mathbf{x}$, how can we decide if we want to classify it as belonging to class $w_i$?
    - $P(w_i)P(\mathbf{x}|w_i) > P(w_j)P(\mathbf{x}|w_j)$ for all $j \neq i$.
    - Assign $\mathbf{x}$ to the largest $P(w_i)P(\mathbf{x}|w_i)$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Dealing with risk in the Bayes classifier

    - In many cases, certain types of errors have bigger consequences than others.
    - **Example 1:**
        - A spam email getting through the spam filter?
            - Annoying, but manageable.
        - A legitimate email with an important message getting caught in the filter?
            - Potentially very damaging.
    - **Example 2:**
        - A benign tumor being classified as malignant in breast cancer screening?
            - Not good; increases workload and chance of unnecessary treatment. But, "better safe than sorry".
        - A malignant tumor being missed?
            - Potentially fatal.
    - Can we incorporate this into the classifier?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Introducing risk factors to the Bayes classifier

    - Define:
        - $\lambda_{12}$: penalty for wrongly classifying $w_2$ as $w_1$
        - $\lambda_{21}$: penalty for wrongly classifying $w_1$ as $w_2$
    - Consider the two-class setting again:
        - $\lambda_{12}P(w_1)P(\mathbf{x}|w_1) > \lambda_{21}P(w_2)P(\mathbf{x}|w_2)$
    - Rewrite as a ratio:

    $$\frac{P(\mathbf{x}|w_1)}{P(\mathbf{x}|w_2)} > \frac{\lambda_{21}P(w_2)}{\lambda_{12}P(w_1)}$$

    - Need to know class-conditional distributions.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Investigating the simplest case — one-dimensional normal distributions

    - Let $P(\mathbf{x}|w_i)$ follow a normal probability density function.
    - One-dimensional case:

    $$P(x|w_i) = \frac{1}{\sqrt{2\pi\sigma_i^2}}\exp\!\left(-\frac{(x-\mu_i)^2}{2\sigma_i^2}\right)$$

    - General (multivariate) case:

    $$P(\mathbf{x}|w_i) = \frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}_i|^{1/2}}\exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^{\!\top}\mathbf{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\right)$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example with two-class problem

    - Consider a balanced dataset with class-conditional distributions $P(\mathbf{x}|w_1)\sim \mathcal{N}(\mu_1, \sigma)$ and $P(\mathbf{x}|w_2)\sim \mathcal{N}(\mu_2, \sigma)$.
    - Let $P(w_1)=P(w_2)$ — balanced dataset (equiprobable classes).
    - Expression for the ratio:

    $$\frac{P(\mathbf{x}|w_1)}{P(\mathbf{x}|w_2)} = \exp\!\left(-\frac{\|\mathbf{x}-\boldsymbol{\mu}_1\|^2 - \|\mathbf{x}-\boldsymbol{\mu}_2\|^2}{2\sigma^2}\right)$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Discriminant functions

    - Take a step back:
        - What is actually going on?
    - Input space is partitioned into regions. Two regions are separated by a decision surface.
    - Often mathematically convenient to describe the decision surface using a *discriminant function*:
        - $g_i(\mathbf{x}) = f(P(w_i)P(\mathbf{x}|w_i))$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Example for Gaussian case

    - Revisiting the Gaussian likelihood.
    - What could be a suitable choice for $f$?
    - In this case, $g_i(\mathbf{x})$ becomes:

    $$g_i(\mathbf{x}) = -\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^{\!\top}\mathbf{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i) - \tfrac{d}{2}\ln(2\pi) - \tfrac{1}{2}\ln|\mathbf{\Sigma}_i| + \ln P(w_i)$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Decision surfaces

    - Decision surface is where the discriminant functions are equal:
        - $g_{ij}(\mathbf{x}) = g_i(\mathbf{x})-g_j(\mathbf{x}) = 0$
    - Quadratic surface! Draw example $\rightarrow$
    - Often we assume equal covariance structure. What happens then?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Continue example for Gaussian case

    - Remember: $g_{ij}(\mathbf{x}) = g_i(\mathbf{x})-g_j(\mathbf{x}) = 0$
    - 1st term $\rightarrow$
    - 2nd term $\rightarrow$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Special case for equiprobable classes with the same covariance matrix

    - Assume $\boldsymbol{\Sigma} = \sigma \mathbf{I}$
    - Then the discriminant function $g_{ij}(\mathbf{x})$ becomes $\rightarrow$
    - Draw example $\rightarrow$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Programming exercises

    Below are programming exercises associated with this lecture. These
    cell blocks are starting points that load the data and prepare the
    problem so that you can get going with the implementation. There are
    also theoretical exercises, but due to copyright we cannot share them
    here. They will be made available in a private repository connected
    to the course.

    For the content covered in this lecture, we assume that we have
    knowledge about the probability density function of the data.
    Therefore, these programming exercises will focus on toy data where
    we have control over the data distribution. In future exercises, we
    will see how we can move on to real-world data.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    number_of_samples_in_each_class = 100

    mu1_toy = np.array([1.0, 1.0])
    mu2_toy = np.array([1.5, 1.5])
    sigma_toy = np.array([[0.2, 0.0], [0.0, 0.2]])

    x_train_1 = np.random.multivariate_normal(mu1_toy, sigma_toy, number_of_samples_in_each_class)
    x_train_2 = np.random.multivariate_normal(mu2_toy, sigma_toy, number_of_samples_in_each_class)
    x_train = np.concatenate((x_train_1, x_train_2))
    y_train = np.concatenate(
        (np.ones(number_of_samples_in_each_class), np.zeros(number_of_samples_in_each_class))
    )

    fig_toy, ax_toy = plt.subplots(figsize=(6, 6))
    ax_toy.scatter(x_train_1[:, 0], x_train_1[:, 1], s=120, facecolors="none",
                   edgecolors="black", linewidth=3.0, label="Class 1")
    ax_toy.scatter(x_train_2[:, 0], x_train_2[:, 1], s=120, facecolors="none",
                   edgecolors="blue", linewidth=3.0, label="Class 2")
    ax_toy.set_xlabel("x1")
    ax_toy.set_ylabel("x2")
    ax_toy.legend()
    fig_toy.tight_layout()

    mo.vstack(
        [
            mo.md(
                r"""
    ### Problem 2.12 from the book

    - Below is code that generates the data associated with problem 2.12
      from the book. Your task is to:
        - Design a Bayesian classifier.
        - Design a Bayesian classifier using the following risk parameters.
          How does this change the decision boundary?
            - $\lambda_{12}$: 1.0
            - $\lambda_{21}$: 0.5
        - Experiment with changing the mean and variance of each class.
          What do you observe?
                """
            ),
            mo.as_html(fig_toy),
        ],
        gap=2,
    )
    plt.close(fig_toy)
    return


if __name__ == "__main__":
    app.run()
