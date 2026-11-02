# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.23.13", "numpy", "matplotlib", "scipy", "scikit-learn"]
# ///
# Keep layouts/gaussian_mixture_models.slides.json beside this file.

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", layout_file="layouts/gaussian_mixture_models.slides.json")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.mixture import GaussianMixture
    from matplotlib.patches import Ellipse

    def normal(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    return Ellipse, GaussianMixture, mo, normal, np, plt, stats


@app.cell
def _(GaussianMixture, normal, np):
    em_data = np.array([150.0, 158.0, 165.0, 175.0, 182.0, 190.0])

    def ll(p):
        _w, _m1, _m2, _s1, _s2 = p
        return float(np.log(_w * normal(em_data, _m1, _s1) + (1 - _w) * normal(em_data, _m2, _s2)).sum())
    _history = [(0.5, 155.0, 185.0, 13.0, 13.0)]
    for _ in range(30):
        _w, _m1, _m2, _s1, _s2 = _history[-1]
        _a = _w * normal(em_data, _m1, _s1)
        _b = (1 - _w) * normal(em_data, _m2, _s2)
        _r = _a / (_a + _b)
        _n1 = _r.sum()
        _n2 = len(_r) - _n1
        _mm1 = float((_r * em_data).sum() / _n1)
        _mm2 = float(((1 - _r) * em_data).sum() / _n2)
        _ss1 = float(np.sqrt((_r * (em_data - _mm1) ** 2).sum() / _n1))
        _ss2 = float(np.sqrt(((1 - _r) * (em_data - _mm2) ** 2).sum() / _n2))
        _history.append((float(_n1 / len(_r)), _mm1, _mm2, max(_ss1, 0.01), max(_ss2, 0.01)))
    em_history = _history
    em_ll = [ll(_p) for _p in _history]
    _rng = np.random.default_rng(8)
    _data = np.r_[_rng.normal(160, 6, 250), _rng.normal(184, 5, 220)].reshape(-1, 1)
    bic_values = [float(GaussianMixture(n_components=_k, random_state=0, reg_covar=1e-05, n_init=3).fit(_data).bic(_data)) for _k in range(1, 7)]
    return bic_values, em_data, em_history, em_ll


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Gaussian Mixture Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Our problem: heights without labels

    Imagine measuring **1,000 heights**. You see two concentrations near 165 and 180 cm, but you do not know which group each person belongs to.

    - **What do we observe?** Only height $x$.
    - **What is missing?** The group label $z$.
    - **What do we want?** Describe the groups and estimate membership.

    **Why start here?** GMMs address precisely this problem of *hidden labels*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_2 = mo.ui.slider(start=0, stop=30, step=1, value=15, label='Distance between means (cm)', show_value=True)
    return (control_2,)


@app.cell(hide_code=True)
def _(control_2, mo, np, plt):
    _ctl = control_2
    _rng = np.random.default_rng(41)
    _a = _rng.normal(165, 7, 650)
    _b = _rng.normal(165 + _ctl.value, 6, 350)
    _fig, _ax = plt.subplots(figsize=(9, 3.7))
    _ax.hist(np.r_[_a, _b], bins=32, color='#5b93d1', alpha=0.75, edgecolor='white')
    _ax.set(xlabel='Height (cm)', ylabel='Frequency', title='We observe data, not groups')
    _ax.grid(alpha=0.15)
    plt.close(_fig)
    mo.vstack([mo.md('## One histogram hides two populations\n\nThe source group is hidden in each histogram. Change the distance between the means.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Why is one Gaussian sometimes insufficient?

    A Gaussian describes **one center** and the spread around it.

    If a population has several concentrations, one bell curve may represent it poorly.

    **Key idea:** Combine several simple distributions instead of forcing one shape.

    A single Gaussian may still describe overlapping or unimodal data; a mixture is not always needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Gaussian

    $$\mathcal N(x\mid\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]$$

    - $\mu$ (**mean**) moves the center.
    - $\sigma$ (**standard deviation**) controls the width.
    - $\sigma^2$ is the **variance**.

    **Why the exponential?** Density decreases as we move away from the mean. **Why the factor in front?** It normalizes the area under the curve to 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_6 = mo.ui.slider(start=150, stop=190, step=1, value=165, label='Mean μ (cm)', show_value=True)
    return (control_6,)


@app.cell(hide_code=True)
def _(control_6, mo, normal, np, plt):
    _ctl = control_6
    _xx = np.linspace(130, 210, 500)
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _ax.plot(_xx, normal(_xx, _ctl.value, 7), lw=3, color='#377fca')
    _ax.axvline(_ctl.value, ls='--', color='#e49a36', label=f'μ = {_ctl.value}')
    _ax.fill_between(_xx, normal(_xx, _ctl.value, 7), alpha=0.14, color='#377fca')
    _ax.set(xlim=(130, 210), ylim=(0, 0.095), xlabel='Height', ylabel='Density')
    _ax.legend()
    _ax.grid(alpha=0.14)
    plt.close(_fig)
    mo.vstack([mo.md('##Mean $\\mu$\n\nMove the mean: the curve shifts without changing width.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    control_8 = mo.ui.slider(start=3, stop=20, step=1, value=7, label='Standard deviation σ', show_value=True)
    return (control_8,)


@app.cell(hide_code=True)
def _(control_8, mo, normal, np, plt):
    _ctl = control_8
    _xx = np.linspace(130, 210, 500)
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _ax.plot(_xx, normal(_xx, 165, _ctl.value), lw=3, color='#377fca')
    _ax.fill_between(_xx, normal(_xx, 165, _ctl.value), alpha=0.15, color='#377fca')
    _ax.axvline(165, ls='--', color='#e49a36')
    _ax.set(xlim=(130, 210), ylim=(0, 0.14), xlabel='Height', ylabel='Density', title='Area always equals 1')
    _ax.grid(alpha=0.14)
    plt.close(_fig)
    mo.vstack([mo.md('## Spread $\\sigma$\n\nChange the standard deviation. Why is a wider bell curve also lower?'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Density is not the probability of an exact value

    For a continuous variable, $P(X=172)=0$.

    The curve gives a **density**. Probabilities come from the **area over an interval**:

    $$P(170\le X\le175)=\int_{170}^{175}f(x)\,dx.$$

    **Why do we need density?** To compare how compatible a height is with different Gaussians. Never treat the curve's height as the probability of an exact value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_11 = mo.ui.slider(start=165, stop=195, step=1, value=175, label='Upper endpoint (cm)', show_value=True)
    return (control_11,)


@app.cell(hide_code=True)
def _(control_11, mo, normal, np, plt, stats):
    _ctl = control_11
    _xx = np.linspace(130, 205, 600)
    _yy = normal(_xx, 165, 7)
    _mask = (_xx >= 165) & (_xx <= _ctl.value)
    _pr = stats.norm.cdf(_ctl.value, 165, 7) - 0.5
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.plot(_xx, _yy, color='#377fca', lw=2.5)
    _ax.fill_between(_xx, _yy, where=_mask, color='#29a084', alpha=0.45)
    _ax.set(title=f'P(165 ≤ X ≤ {_ctl.value}) = {_pr:.3f}', xlabel='Height', ylabel='Density')
    _ax.grid(alpha=0.15)
    plt.close(_fig)
    mo.vstack([mo.md('## Probability = area under the curve\n\nChoose the upper endpoint of an interval starting at 165 cm. The shaded area is the probability.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introducing the GMM: a sum of weighted Gaussians

    With two groups:

    $$p(x)=\pi_1\mathcal N(x\mid\mu_1,\sigma_1^2)+\pi_2\mathcal N(x\mid\mu_2,\sigma_2^2)$$

    **Why a sum?** A point may come from the first **or** the second group; we add the different ways it could be generated.

    **Why weights $\pi_k$?** One group may occur more often than another.

    $$\pi_k\ge0,\qquad\sum_{k=1}^K\pi_k=1.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_14 = mo.ui.slider(start=0, stop=100, step=5, value=60, label='Group 1 weight (%)', show_value=True)
    return (control_14,)


@app.cell(hide_code=True)
def _(control_14, mo, normal, np, plt):
    _ctl = control_14
    _xx = np.linspace(135, 210, 500)
    _w = _ctl.value / 100
    _f1 = _w * normal(_xx, 165, 7)
    _f2 = (1 - _w) * normal(_xx, 180, 6)
    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _ax.plot(_xx, _f1, '--', color='#377fca', lw=2, label=f'π₁·N₁ ({_w:.2f})')
    _ax.plot(_xx, _f2, '--', color='#d98b39', lw=2, label=f'π₂·N₂ ({1 - _w:.2f})')
    _ax.plot(_xx, _f1 + _f2, color='#238b70', lw=3, label='Total GMM')
    _ax.set(xlabel='Height', ylabel='Density')
    _ax.legend()
    _ax.grid(alpha=0.15)
    plt.close(_fig)
    mo.vstack([mo.md("## Three curves: component 1, component 2, mixture\n\nChange the blue group's weight. The green line is always the sum of the two weighted curves."), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## General formula for $K$ groups

    $$\boxed{p(x)=\sum_{k=1}^{K}\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)}$$

    - $K$: number of components.
    - $\pi_k$: prior probability of choosing component $k$.
    - $\mu_k$: center of the component.
    - $\Sigma_k$: variance (1D) or covariance (multiple dimensions).

    **Question:** Must a GMM with $K$ components have $K$ peaks? **No.** Components can overlap and form a single peak.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How is an observation generated?

    **Step 1 —** Draw the latent label $z$:

    $$z\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_K).$$

    **Step 2 —** Given the chosen label, draw the observation:

    $$x\mid z=k\sim\mathcal N(\mu_k,\Sigma_k).$$

    **Why make this explicit?** The model *generates the group first and then the data point*; in real data, we see only the point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_18 = mo.ui.slider(start=30, stop=1000, step=10, value=200, label='N observations', show_value=True)
    return (control_18,)


@app.cell(hide_code=True)
def _(control_18, mo, normal, np, plt):
    _ctl = control_18
    _rng = np.random.default_rng(17)
    _z = _rng.random(1000) < 0.6
    _x = np.where(_z, _rng.normal(165, 7, 1000), _rng.normal(180, 6, 1000))
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.hist(_x[:_ctl.value], bins=30, density=True, color='#669bd3', edgecolor='white', alpha=0.77)
    _xx = np.linspace(130, 210, 400)
    _ax.plot(_xx, 0.6 * normal(_xx, 165, 7) + 0.4 * normal(_xx, 180, 6), color='#238b70', lw=3, label='Theoretical distribution')
    _ax.legend()
    _ax.set(xlabel='Height', ylabel='Density')
    _ax.grid(alpha=0.12)
    plt.close(_fig)
    mo.vstack([mo.md('## Generate data from the mixture\n\nIncrease the number of simulated observations. The latent groups start to emerge even though we see only their combined distribution.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Latent labels and soft clustering

    In the dataset we know $x_i$, **not** $z_i$. Rather than immediately choosing a single label, we calculate:

    $$P(z_i=k\mid x_i).$$

    This is a *posterior probability*.

    - **K-Means:** assigns each point to one group (hard clustering).
    - **GMM:** assigns a probability to each group (soft clustering).

    **Why is this useful?** Points in an overlap region remain uncertain instead of being forced into one group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayes: from densities to responsibilities

    $$\boxed{\gamma_{ik}=P(z_i=k\mid x_i)=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}}$$

    **Numerator:** group frequency $\times$ compatibility with the observation.

    **Denominator:** the contributions of **all** groups. It normalizes the result, so $\sum_k\gamma_{ik}=1$.

    **Name:** $\gamma_{ik}$ is the *responsibility*: how strongly component $k$ explains point $i$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayes: a calculation for 172 cm

    Use $\pi_1=0.6$, $\mu_1=165$, $\sigma_1=7$; $\pi_2=0.4$, $\mu_2=180$, $\sigma_2=6$.

    $$f_1(172)\approx0.03458,\quad f_2(172)\approx0.02734$$
    $$a_1=0.6\cdot0.03458\approx0.02075,\quad a_2=0.4\cdot0.02734\approx0.01094$$
    $$\gamma_1=\frac{a_1}{a_1+a_2}\approx0.655,\quad\gamma_2\approx0.345.$$

    **Why not just compare distances from the means?** Spread and weights matter too.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_23 = mo.ui.slider(start=140, stop=205, step=1, value=172, label='Height x', show_value=True)
    return (control_23,)


@app.cell(hide_code=True)
def _(control_23, mo, normal, np, plt):
    _ctl = control_23
    _xx = np.linspace(140, 205, 500)
    _a = 0.6 * normal(_xx, 165, 7)
    _b = 0.4 * normal(_xx, 180, 6)
    _r = _a / (_a + _b)
    _rx = 0.6 * normal(_ctl.value, 165, 7) / (0.6 * normal(_ctl.value, 165, 7) + 0.4 * normal(_ctl.value, 180, 6))
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _ax.plot(_xx, _r, color='#377fca', lw=3, label='P(group 1 | x)')
    _ax.plot(_xx, 1 - _r, color='#d98b39', lw=3, label='P(group 2 | x)')
    _ax.axvline(_ctl.value, ls=':', color='#555')
    _ax.scatter([_ctl.value], [_rx], color='#377fca', s=65, zorder=5)
    _ax.set(xlabel='Height', ylabel='Probability', ylim=(-0.03, 1.03), title=f'x={_ctl.value} → γ₁={_rx:.3f}, γ₂={1 - _rx:.3f}')
    _ax.legend()
    _ax.grid(alpha=0.12)
    plt.close(_fig)
    mo.vstack([mo.md('## Responsibility\n\nMove the height slider. Membership may be uncertain in overlap regions; responsibilities change toward the tails.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The learning problem

    So far we have known the parameters. With real data, we must estimate them:

    $$\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^{K}.$$

    The observed data are $X=\{x_1,\ldots,x_N\}$.

    **Goal:** Choose parameters that make the observed data as plausible as possible (**maximum likelihood**).

    But how can we estimate the components without knowing $z_i$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Likelihood: why a product and a sum?

    Assuming independent observations:

    $$L(\theta)=\prod_{i=1}^{N}\underbrace{\left[\sum_{k=1}^{K}\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\right]}_{\text{density of point }i}.$$

    - **Product across data points:** Each observation contributes to the overall plausibility.
    - **Sum across components:** Each point could come from any of the $K$ Gaussians.

    Do not confuse *likelihood* with a probability distribution over parameters: here the parameters are values to optimize.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why use log-likelihood?

    $$\boxed{\ell(\theta)=\sum_{i=1}^{N}\log\left(\sum_{k=1}^{K}\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\right)}$$

    **Why take the logarithm?** It turns a product into a sum and avoids numerical underflow.

    **Remaining difficulty:** A **sum over components** remains inside the logarithm, so the optimum cannot be found with the usual sample-mean formula.

    **Solution:** Alternate between estimating membership and updating parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EM: the cycle that solves the problem

    **E-step (Expectation)** — With fixed parameters, calculate $\gamma_{ik}=P(z_i=k\mid x_i)$.

    **M-step (Maximization)** — With fixed responsibilities, update $\pi_k$, $\mu_k$, and $\Sigma_k$.

    **Repeat** until convergence.

    **Why does this work?** If we knew membership, estimating the Gaussians would be easy. If we knew the Gaussians, estimating membership would be easy. EM uses this alternation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initializing EM

    Example: $\mu_1=160$, $\mu_2=185$, $\sigma_1=\sigma_2=10$, $\pi_1=\pi_2=.5$.

    These are **initial guesses**, not final values.

    **Why does initialization matter?** Likelihood is not necessarily convex; EM can settle on different local solutions.

    **In practice:** Initialize with K-Means and try multiple runs (`n_init`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E-step: recalculate every responsibility

    For each point $i$ and Gaussian $k$:

    $$\gamma_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}.$$

    **What changes?** The responsibilities. **What stays fixed?** The model parameters.

    Example row: $x_i=172$, $\gamma_{i1}=.655$, $\gamma_{i2}=.345$.

    Responsibilities in each row always sum to 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## M-step: updating the mean

    Define the effective number of points in the group:

    $$N_k=\sum_{i=1}^{N}\gamma_{ik}.$$

    The new mean is:

    $$\boxed{\mu_k^{new}=\frac{\sum_i\gamma_{ik}x_i}{N_k}}$$

    **Why a weighted mean?** A point that strongly belongs to a component influences its center more; an uncertain point influences it less.

    Example: $x=(160,170,180)$, $\gamma_{\cdot1}=(.9,.6,.1)$ gives $\mu_1=165$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## M-step: weighted variance

    In 1D:

    $$\boxed{(\sigma_k^2)^{new}=\frac{\sum_i\gamma_{ik}(x_i-\mu_k^{new})^2}{N_k}}$$

    **Why square the deviations?** To measure spread without positive and negative deviations canceling out.

    **Why use weights?** Variance should mainly describe the points assigned to that component.

    With $x=(160,170,180)$, $\gamma=(.9,.6,.1)$, and mean 165: variance is $37.5$ and $\sigma\approx6.12$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## M-step: updating the weights

    $$\boxed{\pi_k^{new}=\frac{N_k}{N}=\frac1N\sum_i\gamma_{ik}}$$

    **Why this formula?** $N_k$ counts the points *effectively* assigned to the group. Dividing by $N$ gives its overall share.

    Example: $N=3$, $N_1=.9+.6+.1=1.6$.

    $$\pi_1^{new}=1.6/3\approx0.533.$$

    **Check:** $\sum_k\pi_k^{new}=1$ automatically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Complete example: one EM iteration

    Data: $x=(160,170,180)$; assumed responsibilities for group 1: $(.9,.6,.1)$.

    1. **E-step already done:** We have $\gamma_{\cdot1}$, and $\gamma_{\cdot2}=(.1,.4,.9)$.
    2. **Effective counts:** $N_1=1.6$, $N_2=1.4$.
    3. **Means:** $\mu_1=165$, $\mu_2=(.1\cdot160+.4\cdot170+.9\cdot180)/1.4\approx175.71$.
    4. **Variances:** Each is a weighted mean of squared deviations.
    5. **Weights:** $\pi_1\approx.533$, $\pi_2\approx.467$.

    On the next iteration, recalculate responsibilities with the **new** parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_35 = mo.ui.slider(start=0, stop=30, step=1, value=0, label='EM iteration', show_value=True)
    return (control_35,)


@app.cell(hide_code=True)
def _(control_35, em_data, em_history, em_ll, mo, normal, np, plt):
    _ctl = control_35
    _p = em_history[_ctl.value]
    _xx = np.linspace(135, 205, 500)
    _f1 = _p[0] * normal(_xx, _p[1], _p[3])
    _f2 = (1 - _p[0]) * normal(_xx, _p[2], _p[4])
    _fig, _ax = plt.subplots(figsize=(9, 3.9))
    _ax.plot(_xx, _f1, '--', color='#377fca', lw=2, label=f'Group 1: μ={_p[1]:.1f}, σ={_p[3]:.1f}')
    _ax.plot(_xx, _f2, '--', color='#d98b39', lw=2, label=f'Group 2: μ={_p[2]:.1f}, σ={_p[4]:.1f}')
    _ax.plot(_xx, _f1 + _f2, color='#238b70', lw=3, label='Mixture')
    _ax.scatter(em_data, np.zeros_like(em_data), color='#272d38', zorder=5, s=45)
    _ax.set(xlabel='Height', ylabel='Density', title=f'Iteration {_ctl.value} · log-likelihood = {em_ll[_ctl.value]:.3f}')
    _ax.legend(fontsize=9)
    _ax.grid(alpha=0.13)
    plt.close(_fig)
    mo.vstack([mo.md('##Each slider position shows an E-step followed by an M-step.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    control_37 = mo.ui.slider(start=0, stop=30, step=1, value=10, label='Iteration', show_value=True)
    return (control_37,)


@app.cell(hide_code=True)
def _(control_37, em_ll, mo, plt):
    _ctl = control_37
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _ax.plot(range(len(em_ll)), em_ll, color='#238b70', lw=2.8, marker='o', ms=3)
    _ax.scatter([_ctl.value], [em_ll[_ctl.value]], color='#e49a36', s=110, zorder=5)
    _ax.set(xlabel='Iteration', ylabel='Log-likelihood', title=f'Current value: {em_ll[_ctl.value]:.3f}')
    _ax.grid(alpha=0.15)
    plt.close(_fig)
    mo.vstack([mo.md('## EM increases log-likelihood'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convergence and limitations of EM

    EM stops when the improvement in log-likelihood becomes very small or it reaches the iteration limit.

    **Property:** Standard EM iterations do not decrease log-likelihood.

    **But keep in mind:**
    - Convergence does **not** guarantee a global maximum.
    - A component may collapse onto one point ($\sigma^2\to0$), causing degeneracy.
    - Covariance regularization and multiple initializations help.
    """)
    return


@app.cell(hide_code=True)
def _(Ellipse, mo, np, plt):
    _rng = np.random.default_rng(32)
    _means = (np.array([165, 62]), np.array([181, 82]))
    _covariances = (np.array([[36, 24], [24, 64]]), np.array([[30, 18], [18, 49]]))
    _colors = ('#377fca', '#d98b39')
    _fig, _ax = plt.subplots(figsize=(7.2, 2.6))
    for _k in range(2):
        _points = _rng.multivariate_normal(_means[_k], _covariances[_k], size=90)
        _ax.scatter(_points[:, 0], _points[:, 1], s=15, alpha=0.45,
                    color=_colors[_k], label=f'Component {_k + 1}')
        _values, _vectors = np.linalg.eigh(_covariances[_k])
        _angle = np.degrees(np.arctan2(_vectors[1, 1], _vectors[0, 1]))
        _ax.add_patch(Ellipse(_means[_k], width=4 * np.sqrt(_values[1]),
                              height=4 * np.sqrt(_values[0]), angle=_angle,
                              fill=False, edgecolor=_colors[_k], linewidth=2))
    _ax.set(xlabel='Height (cm)', ylabel='Weight (kg)', xlim=(145, 200),
            ylim=(40, 105), title='Two features, two Gaussian components')
    _ax.legend(loc='upper left', fontsize=8)
    _ax.grid(alpha=0.12)
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.md('## From a line to a plane: multidimensional data\n\nWith height **and** weight, each observation becomes a vector:\n\n$$x_i=\\begin{bmatrix}\\mathrm{height}_i\\\\\\mathrm{weight}_i\\end{bmatrix},\\quad\\mu_k=\\begin{bmatrix}\\mu_{h,k}\\\\\\mu_{w,k}\\end{bmatrix}.$$\n\nComponents are no longer simple curves: they become Gaussian surfaces with **elliptical contours**.\n\n**Why make this extension?** Real data often require clustering with several features at once.'), mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does a covariance matrix add?

    $$\Sigma_k=\begin{bmatrix}\sigma_{h,k}^2&\operatorname{Cov}(h,w)\\\operatorname{Cov}(h,w)&\sigma_{w,k}^2\end{bmatrix}$$

    - Diagonal: spread of each feature.
    - Off-diagonal: how two variables increase or decrease together.

    **Why does it matter?** A cluster may be elongated and rotated; correlations determine its orientation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_42 = mo.ui.slider(start=-0.9, stop=0.9, step=0.1, value=0.6, label='Correlation ρ', show_value=True)
    return (control_42,)


@app.cell(hide_code=True)
def _(Ellipse, control_42, mo, np, plt):
    _ctl = control_42
    _rho = _ctl.value
    _cv = np.array([[1.4 ** 2, _rho * 1.4 * 0.85], [_rho * 1.4 * 0.85, 0.85 ** 2]])
    _vals, _vecs = np.linalg.eigh(_cv)
    _angle = np.degrees(np.arctan2(_vecs[1, 1], _vecs[0, 1]))
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    for _rad in [1, 2, 3]:
        _ax.add_patch(Ellipse((0, 0), width=2 * _rad * np.sqrt(_vals[1]), height=2 * _rad * np.sqrt(_vals[0]), angle=_angle, fill=False, edgecolor='#377fca', lw=2, alpha=1.1 - _rad * 0.2))
    _ax.scatter([0], [0], color='#e49a36', s=65)
    _ax.set(xlim=(-5, 5), ylim=(-3.2, 3.2), xlabel='Feature 1', ylabel='Feature 2', title=f'ρ = {_rho:+.1f}: covariance geometry')
    _ax.set_aspect('equal')
    _ax.grid(alpha=0.12)
    plt.close(_fig)
    mo.vstack([mo.md('##Ellipses and covariance\n\nChange the correlation. The ellipses rotate while keeping the same center.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The multivariate Gaussian

    $$\mathcal N(x\mid\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right]$$

    - $d$: dimension of the vector.
    - $|\Sigma|$: volume/normalization factor.
    - $\Sigma^{-1}$: accounts for scale and correlation.
    - The quadratic term is the **squared Mahalanobis distance**.

    **Why not Euclidean distance?** Moving along a direction with large spread is less surprising than moving along a narrow direction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multidimensional M-step

    The mean is still weighted:

    $$\mu_k^{new}=\frac{\sum_i\gamma_{ik}x_i}{N_k}.$$

    The variance becomes a matrix:

    $$\boxed{\Sigma_k^{new}=\frac{\sum_i\gamma_{ik}(x_i-\mu_k^{new})(x_i-\mu_k^{new})^T}{N_k}}$$

    **Why the outer product?** It captures both the variance along each dimension and covariance between dimensions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many Gaussians should we choose?

    We do not always know $K$. Too few components may underfit; too many add unnecessary complexity.

    **BIC** balances fit and number of parameters:

    $$\operatorname{BIC}=-2\ell(\hat\theta)+p\log N.$$

    **Why is lower better?** The first term rewards high likelihood; the second penalizes complexity. We test several values of $K$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_47 = mo.ui.slider(start=1, stop=6, step=1, value=2, label='Selected K', show_value=True)
    return (control_47,)


@app.cell(hide_code=True)
def _(bic_values, control_47, mo, np, plt):
    _ctl = control_47
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _ks = np.arange(1, 7)
    _ax.plot(_ks, bic_values, marker='o', color='#377fca', lw=2.6)
    _ax.scatter([_ctl.value], [bic_values[_ctl.value - 1]], color='#e49a36', s=125, zorder=5)
    _ax.set(xticks=_ks, xlabel='Number of Gaussians K', ylabel='BIC (lower is better)', title=f'Selected K={_ctl.value}, BIC={bic_values[_ctl.value - 1]:.1f}')
    _ax.grid(alpha=0.15)
    plt.close(_fig)
    mo.vstack([mo.md('## BIC: compare $K=1,\\ldots,6$\n\nFor synthetic data, the BIC minimum suggests balanced model complexity, not necessarily the number of real-world categories.'), _ctl, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## When should you use a GMM?

    **Useful when:**
    - You want membership probabilities (soft clustering).
    - Groups overlap.
    - Groups are elliptical and have different variances.
    - You want to estimate a density or identify poorly fitting data points.

    **Limitations:** Choosing $K$, sensitivity to initialization, Gaussian component assumptions, and possible covariance degeneracy.

    **Remember:** A mathematical component is not automatically a real-world category.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three formulas to remember

    **Model:**
    $$p(x)=\sum_{k=1}^{K}\pi_k\mathcal N(x\mid\mu_k,\Sigma_k).$$

    **E-step:**
    $$\gamma_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}.$$

    **M-step (mean):**
    $$\mu_k^{new}=\frac{\sum_i\gamma_{ik}x_i}{\sum_i\gamma_{ik}}.$$

    **Logical sequence:** Parameters → membership → new parameters → repeat.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solutions and conclusion

    1. The weights are probabilities of choosing the groups.
    2. Gaussians can overlap; the labels are hidden.
    3. The E-step updates **only** responsibilities.
    4. Data points should contribute in proportion to their compatibility with the cluster.
    5. **No:** EM can converge to local maxima or degenerate solutions.
    6. To describe spread and correlation across dimensions.

    **Takeaway:** A GMM is a probabilistic generative model. EM learns it by alternating **inference** about hidden labels and **estimation** of parameters.
    """)
    return


if __name__ == "__main__":
    app.run()
