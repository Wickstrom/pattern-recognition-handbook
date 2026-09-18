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
# Marimo version of the ensemble methods lecture.
# Run locally with `marimo edit notebooks/14/ensemble_methods.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="medium",
    layout_file="layouts/ensemble_methods.slides.json",
)


@app.cell
def _():
    import os
    from pathlib import Path

    # Make relative paths (media/, layouts/) resolve the same way whether
    # `marimo edit` is launched from the repo root or from this directory,
    # and no-op harmlessly under the WASM/Pyodide export.
    if "__file__" in globals() and __file__:
        try:
            os.chdir(Path(__file__).resolve().parent)
        except OSError:
            pass

    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Ensemble methods

    ## Introduction

    - In this lecture we will cover two ensemble strategies: **parallel** and **sequential**.
    - We will cover:
        - Recap: decision trees and bias-variance trade-off
        - Parallel ensemble: Bagging and random forests
        - Sequential: AdaBoost, overlook of other boosting method
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 18</div>
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    return DecisionTreeClassifier, make_moons, np, plt, train_test_split


@app.cell
def _(mo):
    mo.md(r"""
    ## Recap: decision trees

    - Greedy, recursive partitioning of the feature space

    - Deep tree: low bias, high variance
    - Shallow tree: high bias, low variance

    **Ensambles** aim at reducing the bias-variance trade-off 🥇
    - Bagging $\rightarrow$ reduces the variance: perfect for unstable good learners
    - Boosting $\rightarrow$ reduces the bias: perfect in case of weak learners
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 18</div>
    """)
    return


@app.cell
def _(mo, DecisionTreeClassifier, make_moons, np, plt):
    _X, _y = make_moons(n_samples=200, noise=0.3, random_state=0)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for _ax, _seed in zip(_axes, [1, 2]):
        _rng = np.random.RandomState(_seed)
        _idx = _rng.choice(len(_X), size=len(_X), replace=True)
        _tree = DecisionTreeClassifier(random_state=0).fit(_X[_idx], _y[_idx])
        _gx, _gy = np.meshgrid(
            np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
            np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
        )
        _Z = _tree.predict(np.c_[_gx.ravel(), _gy.ravel()]).reshape(_gx.shape)
        _ax.contourf(_gx, _gy, _Z, alpha=0.25, cmap="coolwarm")
        _ax.scatter(_X[:, 0], _X[:, 1], c=_y, cmap="coolwarm", edgecolors="k", s=20)
        _ax.set_title(f"bootstrap resample, seed={_seed}")
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.suptitle("A single deep tree: small data changes, very different boundary")
    mo.vstack([
        mo.as_html(_fig),
        mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 18</div>"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How can we reduce variance? $\rightarrow$ **AVERAGING**

    - Consider $B$ predictors $z_B$, each with variance $\sigma^2$, pairwise correlation $\rho$:

    $$
    \mathrm{Var}(\frac{1}{B}\sum_{b=1}^Bz_b) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
    $$

    - The $\frac{1}{B}$ term vanishes as $B\to\infty$
    - The $\rho\sigma^2$ term does not: correlated predictors put a floor on how much averaging helps
    - Two ways to lower the variance: increase B (bagging), or/and decrease $\rho$ (random forest)
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 18</div>
    """
    ).callout(kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bagging (Breiman, 1996)

    **AIM: reducing the variance of unstable learners by averaging**

    how $\rightarrow$ averaging predictors!

    - Bootstrap: draw $N$ samples **with replacement** from the $N$-sample training set
    - Fit one predictor per bootstrap sample, $B$ total
    - Predict: mean (regression) or majority vote (classification)
    - Each bootstrap leaves out $\approx 1/e \approx 36.8\%$ of points
        - Free validation set: **out-of-bag (OOB) error**
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 18</div>
    """)
    return


@app.cell
def _(DecisionTreeClassifier, np):
    def fit_bagging(X, y, n_trees, rng, max_features=None):
        """Bootstrap `n_trees` deep trees. Passing `max_features` turns this
        into the random-forest variant further down."""
        n = len(X)
        trees = []
        for _ in range(n_trees):
            idx = rng.randint(0, n, size=n)
            tree = DecisionTreeClassifier(
                max_features=max_features, random_state=rng.randint(1_000_000)
            )
            tree.fit(X[idx], y[idx])
            trees.append(tree)
        return trees

    def predict_vote(trees, X):
        votes = np.stack([t.predict(X) for t in trees])
        return (votes.mean(axis=0) >= 0.5).astype(int)

    return fit_bagging, predict_vote


@app.cell
def _(mo):
    n_trees_bag = mo.ui.slider(
        start=1, stop=60, step=1, value=1, show_value=True, label="number of trees"
    )
    return (n_trees_bag,)


@app.cell
def _(fit_bagging, make_moons, mo, n_trees_bag, np, plt, predict_vote):
    _X, _y = make_moons(n_samples=200, noise=0.3, random_state=0)
    _rng = np.random.RandomState(0)
    _trees = fit_bagging(_X, _y, n_trees_bag.value, _rng)
    _gx, _gy = np.meshgrid(
        np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
        np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
    )
    _Z = predict_vote(_trees, np.c_[_gx.ravel(), _gy.ravel()]).reshape(_gx.shape)
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.contourf(_gx, _gy, _Z, alpha=0.25, cmap="coolwarm")
    _ax.scatter(_X[:, 0], _X[:, 1], c=_y, cmap="coolwarm", edgecolors="k", s=20)
    _n = n_trees_bag.value
    _ax.set_title(f"Bagging, {_n} tree{'s' if _n != 1 else ''}")
    _ax.set_xticks([])
    _ax.set_yticks([])
    mo.vstack([n_trees_bag, _fig, mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 18</div>""")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bagging beyond trees: KDE Naive Bayes

    Bagging is not restricted to decision trees.
    We can also create an ensemble of **KDE-based Naive Bayes classifiers**.

    For each ensemble member \(b=1,\ldots,B\):

    1. Draw a **bootstrap sample** \(D^{*(b)}\) from the training set.
    2. Sample a kernel width \(h_b\) around a reasonable reference value.
    3. Fit a KDE Naive Bayes classifier using \(D^{*(b)}\) and \(h_b\).

    \[
    \hat p_b(x_j\mid y=c)
    =
    \frac{1}{n_c h_b}
    \sum_{i:y_i=c}
    K\!\left(
    \frac{x_j-x_{ij}}{h_b}
    \right).
    \]

    Different bandwidths give different levels of smoothing:

    \[
    \boxed{
    \text{small }h
    \rightarrow
    \text{more flexible density}
    \qquad
    \text{large }h
    \rightarrow
    \text{smoother density}
    }
    \]

    Finally, combine the predictions of all classifiers:

    \[
    \hat y(x)
    =
    \operatorname{majority\ vote}
    \left\{
    h_1(x),\ldots,h_B(x)
    \right\}.
    \]

    **Take-home:** the bootstrap changes the training data, while the
    randomized kernel width adds extra diversity between the KDE learners.
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 18</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Helps most for **unstable** learners (deep trees);
    - Trees only depend on their own bootstrap sample: fits in parallel;
    - Does not reduce bias - still limited by how well one tree can fit;
    - if the predictors are strongly correlated (many trees can choose the same predictor when there is a strong one) $\rightarrow$ $\rho$ is dominating and averaging does not reduce the variance.
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 18</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Random forests (Breiman, 2001)

    - Bagging + **one extra source of randomness**: at each split, consider only $m$ of the $d$ features
        - Classification: $m \approx \sqrt{d}$. Regression: $m \approx d/3$
    - Why: bagging alone still lets the same strong feature dominate every tree - trees stay correlated
    - Restricting the feature subset decorrelates the trees, lowering $\rho$ in the variance formula above
    - trade-off between the strenght of an individual tree and reducing the correlation
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 18</div>
    """)
    return


@app.cell
def _(mo):
    n_trees_rf = mo.ui.slider(
        start=1, stop=60, step=1, value=20, show_value=True, label="number of trees"
    )
    m_rf = mo.ui.slider(
        start=1, stop=2, step=1, value=2, show_value=True, label="features per split (m)"
    )
    return m_rf, n_trees_rf


@app.cell
def _(fit_bagging, m_rf, make_moons, mo, n_trees_rf, np, plt, predict_vote):
    _X, _y = make_moons(n_samples=200, noise=0.3, random_state=0)
    _rng = np.random.RandomState(0)
    _trees = fit_bagging(_X, _y, n_trees_rf.value, _rng, max_features=m_rf.value)
    _gx, _gy = np.meshgrid(
        np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
        np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
    )
    _Z = predict_vote(_trees, np.c_[_gx.ravel(), _gy.ravel()]).reshape(_gx.shape)
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.contourf(_gx, _gy, _Z, alpha=0.25, cmap="coolwarm")
    _ax.scatter(_X[:, 0], _X[:, 1], c=_y, cmap="coolwarm", edgecolors="k", s=20)
    _ax.set_title(f"Random forest, {n_trees_rf.value} trees, m={m_rf.value}")
    _ax.set_xticks([])
    _ax.set_yticks([])
    mo.vstack([n_trees_rf, m_rf, _fig, mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 18</div>""")])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Bagging vs. random forest:** bagging introduces randomness in the *data*;
    random forest also introduces randomness in the *features*.
        

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 18</div>
    """
    ).callout(kind="success")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Boosting

    **AIM: reducing the bias of weak learners**

    how $\rightarrow$ sequentially training weak predictors to improve them!

    - Bagging: $B$ trees fit **independently**, in parallel
    - Boosting: fit trees **sequentially** - each one focused on the previous ensemble's mistakes
    - Weak learners combined into a strong one - here: decision stumps (depth-1 trees)
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 18</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deriving AdaBoost from the exponential loss

    - Labels $y_i \in \{-1, +1\}$. Ensemble so far: $F_{m-1}(x) = \sum_{k=1}^{m-1} \alpha_k h_k(x)$
    - Add one term, greedily, minimizing the exponential loss:

    $$
    (\alpha_m, h_m) = \arg\min_{\alpha, h} \sum_{i=1}^N \exp\big(-y_i (F_{m-1}(x_i) + \alpha\, h(x_i))\big)
    $$

    - Write $w_i^{(m)} = \exp(-y_i F_{m-1}(x_i))$ (fixed at step $m$, it only depends on the past). The objective becomes

    $$
    \sum_i w_i^{(m)} \exp(-\alpha y_i h(x_i))
    $$
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 18</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Solving for $h_m$ and $\alpha_m$

    - Split the sum into correct / wrong predictions. For any fixed $\alpha>0$, it is minimized by the $h$ minimizing the **weighted error**

    $$
    \varepsilon_m = \frac{\sum_i w_i^{(m)} \mathbb{1}[h(x_i)\neq y_i]}{\sum_i w_i^{(m)}}
    $$

    - Differentiate w.r.t. $\alpha$, set to zero:

    $$
    \alpha_m = \frac{1}{2}\ln\frac{1-\varepsilon_m}{\varepsilon_m}
    $$

    - Weight update, $w_i^{(m+1)} = w_i^{(m)} e^{-\alpha_m y_i h_m(x_i)}$: misclassified points reweighted up by $e^{\alpha_m}$, correct ones down by $e^{-\alpha_m}$
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 18</div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The AdaBoost algorithm (Freund & Schapire, 1997)

    - Initialize $w_i^{(1)} = 1/N$
    - For $m = 1, \dots, M$:
        - Fit $h_m$ on the weighted data (minimize $\varepsilon_m$)
        - $\alpha_m = \frac12 \ln\frac{1-\varepsilon_m}{\varepsilon_m}$
        - Reweight, then renormalize so weights sum to 1
    - Output: $F(x) = \mathrm{sign}\left(\sum_m \alpha_m h_m(x)\right)$
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 18</div>
    """)
    return


@app.cell
def _(DecisionTreeClassifier, np):
    def fit_adaboost(X, y, n_rounds):
        """Decision-stump AdaBoost. Labels y in {-1, +1}."""
        n = len(X)
        w = np.full(n, 1.0 / n)
        stumps, alphas = [], []
        for _ in range(n_rounds):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            pred = stump.predict(X)
            err = np.sum(w * (pred != y)) / np.sum(w)
            err = np.clip(err, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-alpha * y * pred)
            w = w / w.sum()
            stumps.append(stump)
            alphas.append(alpha)
        return stumps, alphas

    def predict_adaboost(stumps, alphas, X):
        agg = sum(a * s.predict(X) for s, a in zip(stumps, alphas))
        return np.sign(agg)

    return (fit_adaboost,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Does more rounds mean overfitting?

    - Training error keeps dropping as $M$ grows - does test error follow it back up?
    - Below: a noisy Two Moons problem, train and test error against $M$, on the **same** plot
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 18</div>
    """)
    return


@app.cell
def _(mo, fit_adaboost, make_moons, np, plt, train_test_split):
    _X, _y01 = make_moons(n_samples=300, noise=0.35, random_state=7)
    _y = np.where(_y01 == 0, -1, 1)
    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=0.5, random_state=7
    )

    _n_rounds_max = 150
    _stumps, _alphas = fit_adaboost(_X_train, _y_train, _n_rounds_max)

    # Vectorized: predict once per stump, then take a running (cumulative)
    # weighted sum over rounds - O(M) instead of refitting/repredicting the
    # full ensemble at every prefix length.
    _train_preds = np.array([s.predict(_X_train) for s in _stumps])
    _test_preds = np.array([s.predict(_X_test) for s in _stumps])
    _alphas_arr = np.array(_alphas)[:, None]

    _train_cumsum = np.cumsum(_alphas_arr * _train_preds, axis=0)
    _test_cumsum = np.cumsum(_alphas_arr * _test_preds, axis=0)

    _train_err = np.mean(np.sign(_train_cumsum) != _y_train[None, :], axis=1)
    _test_err = np.mean(np.sign(_test_cumsum) != _y_test[None, :], axis=1)

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.plot(range(1, _n_rounds_max + 1), _train_err, label="train error")
    _ax.plot(range(1, _n_rounds_max + 1), _test_err, label="test error")
    _ax.set_xlabel("number of rounds (M)")
    _ax.set_ylabel("classification error")
    _ax.set_title("AdaBoost: train reaches zero, test does not follow")
    _ax.legend()
    mo.vstack([
        mo.as_html(_fig),
        mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 18</div>"""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Outlook: gradient boosting

    - AdaBoost = forward stagewise fitting, specifically **of the exponential loss**
    - Gradient boosting: same recipe, any differentiable loss - fit each new tree to (minus) the loss gradient of the current ensemble
    - AdaBoost is recovered as the special case of the exponential loss
    - XGBoost, LightGBM, CatBoost
    

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 18</div>
    """)
    return


if __name__ == "__main__":
    app.run()
