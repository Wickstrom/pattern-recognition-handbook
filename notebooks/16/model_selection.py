# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.23.13", "numpy", "matplotlib", "scikit-learn"]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="medium",
    layout_file="layouts/model_selection.slides.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    return Rectangle, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model selection and performance evaluation


    - Generalization and unseen data
    - Training, validation and test sets
    - Cross-validation and model selection
    - Hyperparameter tuning
    - Performance metrics
    - A complete evaluation workflow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning from data

    Suppose we have **10,000 labelled images** of cats and dogs:

    $$\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^{N},\qquad y_i\in\{0,1\}.$$

    We learn a predictive function $f_{\boldsymbol{\theta}}(\mathbf{x})$ by minimizing empirical loss:

    $$\hat{\boldsymbol{\theta}}=\arg\min_{\boldsymbol{\theta}}\frac{1}{N}\sum_{i=1}^{N}L(f_{\boldsymbol{\theta}}(\mathbf{x}_i),y_i).$$

    **Question:** Will this model work on images it has never seen?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training error and generalization error

    **Training error** is measured on the examples used to learn the model:

    $$\hat R_{\mathrm{train}}(f)=\frac{1}{N}\sum_{i=1}^{N}L(f(\mathbf{x}_i),y_i).$$

    **Generalization error** is the expected loss on fresh examples:

    $$R(f)=\mathbb{E}_{(\mathbf{X},Y)\sim P_{\mathrm{data}}}[L(f(\mathbf{X}),Y)].$$

    We know the training data, but not the complete distribution $P_{\mathrm{data}}$.

    - Model A: 99% training accuracy, 72% on unseen data.
    - Model B: 91% training accuracy, 88% on unseen data.

    **Which model would you choose?**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    complexity = mo.ui.slider(1, 9, value=5, step=1, label="Model complexity", show_value=True)
    return (complexity,)


@app.cell(hide_code=True)
def _(complexity, mo, np, plt):
    _x = np.linspace(1, 9, 160)
    _train = .43 * np.exp(-.40 * (_x - 1)) + .025
    _general = .16 + .014 * (_x - 4.7)**2
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.plot(_x, _train, linewidth=3, label="Training error")
    _ax.plot(_x, _general, linewidth=3, label="Error on new data")
    _ax.axvline(complexity.value, color="gray", linestyle="--")
    _ax.scatter([complexity.value], [np.interp(complexity.value, _x, _general)], s=90, zorder=5)
    _ax.set(xlabel="Model complexity", ylabel="Illustrative error", ylim=(0, .55))
    _ax.grid(alpha=.15); _ax.legend()
    _fig.tight_layout(); plt.close(_fig)
    mo.vstack([mo.md("## Overfitting: more complex is not always better"), complexity, mo.as_html(_fig), mo.md("Training error can continue to decrease even when the error on unseen data increases.")], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimating performance on new data

    Evaluate a **fixed model** on $M$ new, independent examples:

    $$\hat R_{\mathrm{new}}(f)=\frac{1}{M}\sum_{i=1}^{M}L(f(\mathbf{x}_i),y_i).$$

    If the evaluation examples are i.i.d. from the target distribution:

    $$\mathbb{E}[\hat R_{\mathrm{new}}(f)\mid f]=R(f).$$

    - The estimate contains statistical uncertainty.
    - A larger evaluation sample usually makes it more stable.
    - Repeatedly using the test set to choose a model destroys independence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why do we need three sets?

    - **Training:** learn model parameters.
    - **Validation:** choose architecture, hyperparameters, and checkpoints.
    - **Test:** evaluate the final chosen model on untouched data.

    | Dataset | Updates model weights? | Guides model selection? |
    |---|---|---|
    | Training | Yes | Yes |
    | Validation | Usually not directly | Yes |
    | Test | No | **No** |

    A test set used repeatedly for tuning is no longer an independent final test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_share = mo.ui.slider(50, 80, value=70, step=5, label="Training (%)", show_value=True)
    return (train_share,)


@app.cell(hide_code=True)
def _(mo):
    val_share = mo.ui.slider(5, 25, value=15, step=5, label="Validation (%)", show_value=True)
    return (val_share,)


@app.cell(hide_code=True)
def _(Rectangle, mo, plt, train_share, val_share):
    _a, _b = train_share.value, val_share.value
    _c = 100 - _a - _b
    _fig, _ax = plt.subplots(figsize=(9, 2.3))
    for _left, _width, _color, _name in [(0, _a, '#377fca', 'Train'), (_a, _b, '#e49a36', 'Validation'), (_a+_b, _c, '#65a78b', 'Test')]:
        _ax.add_patch(Rectangle((_left, 0), _width, 1, facecolor=_color, edgecolor='white', linewidth=3))
        _ax.text(_left+_width/2, .5, f'{_name}\n{_width}%', ha='center', va='center', color='white', weight='bold', fontsize=10)
    _ax.set(xlim=(0, 100), ylim=(0, 1)); _ax.axis('off')
    _fig.tight_layout(); plt.close(_fig)
    mo.vstack([mo.md('## Train–validation–test split'), mo.hstack([train_share, val_share], gap=3), mo.as_html(_fig), mo.md(f'For 10,000 examples: **{_a*100:,} train**, **{_b*100:,} validation**, **{_c*100:,} test**. The percentages are illustrative.')], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters and hyperparameters

    Training adjusts **parameters** such as neural-network weights:

    $$\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_t-\eta\nabla_{\boldsymbol{\theta}}\hat R_{\mathrm{train}}(\boldsymbol{\theta}_t).$$

    Validation helps choose **hyperparameters**, such as learning rate $\eta$:

    | Learning rate | Validation loss |
    |---:|---:|
    | 0.1 | 0.52 |
    | **0.01** | **0.31** |
    | 0.001 | 0.43 |

    We choose $\eta=0.01$, but **0.31 is not an independent final evaluation**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    epoch = mo.ui.slider(1, 100, value=50, label="Training epoch", show_value=True)
    return (epoch,)


@app.cell(hide_code=True)
def _(epoch, mo, np, plt):
    _t = np.arange(1, 101)
    _train = .82 * np.exp(-_t/31) + .07
    _val = .18 + .00013 * (_t-50)**2 + .22*np.exp(-_t/13)
    _best = int(_t[np.argmin(_val)])
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.plot(_t, _train, linewidth=3, label='Training loss')
    _ax.plot(_t, _val, linewidth=3, label='Validation loss')
    _ax.axvline(epoch.value, linestyle='--', color='gray')
    _ax.axvline(_best, linestyle=':', color='#e49a36', linewidth=2, label=f'Best epoch {_best}')
    _ax.scatter(epoch.value, _val[epoch.value-1], s=85, zorder=6)
    _ax.set(xlabel='Epoch', ylabel='Loss', ylim=(0, 1))
    _ax.legend(); _ax.grid(alpha=.15)
    _fig.tight_layout(); plt.close(_fig)
    mo.vstack([mo.md('## Early stopping'), epoch, mo.as_html(_fig), mo.md('Save the checkpoint with the lowest validation loss. With *patience = 10*, stop after ten epochs without improvement and restore the best weights.')], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why not tune on the test set?

    If we compare many learning rates, architectures, or checkpoints using the test set, we gradually adapt our choices to its random fluctuations.

    - **Training** learns weights.
    - **Validation** influences the chosen model, even without contributing gradients.
    - **Test** is accessed only after decisions are fixed.

    A test accuracy of 87% estimates future performance. It does **not** guarantee 87% accuracy on every future dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The limitation of a single validation split

    With only **500 development examples**, the validation score depends on the particular split.

    - One split may give 84% accuracy.
    - Another split may give 90% accuracy.

    **K-fold cross-validation** rotates the validation subset:

    1. Divide development data into $K$ folds.
    2. Train a fresh model on $K-1$ folds.
    3. Validate on the remaining fold.
    4. Repeat until every fold has served as validation.

    The independent test set stays outside this process.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fold = mo.ui.slider(1, 5, value=1, label="Validation fold", show_value=True)
    return (fold,)


@app.cell(hide_code=True)
def _(Rectangle, fold, mo, plt):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    for _run in range(5):
        for _k in range(5):
            _color = '#e49a36' if _k == _run else '#377fca'
            _ax.add_patch(Rectangle((_k, 4-_run), .97, .88, facecolor=_color, edgecolor='white', linewidth=2))
            _ax.text(_k+.48, 4-_run+.44, 'Val' if _k==_run else 'Train', ha='center', va='center', color='white', fontsize=10)
        if _run+1 == fold.value:
            _ax.add_patch(Rectangle((-.05, 4-_run-.05), 5.04, .98, fill=False, linewidth=3, edgecolor='black'))
    _ax.set(xlim=(-.08, 5.02), ylim=(-.1, 5.05))
    _ax.set_xticks([.48,1.48,2.48,3.48,4.48], labels=[f'Fold {_k}' for _k in range(1,6)])
    _ax.set_yticks([4.44,3.44,2.44,1.44,.44], labels=[f'Run {_k}' for _k in range(1,6)])
    _ax.tick_params(length=0)
    for _spine in _ax.spines.values(): _spine.set_visible(False)
    _fig.tight_layout(); plt.close(_fig)
    mo.vstack([mo.md('## Five-fold cross-validation'), fold, mo.as_html(_fig), mo.md(f'**Run {fold.value}:** 400 examples for training, 100 for validation. Each run starts with a new model; weights are not carried over.')], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which of the five models do we keep?

    | Fold model | Validation accuracy |
    |---|---:|
    | $f_{\boldsymbol{\theta}_1}$ | 86% |
    | $f_{\boldsymbol{\theta}_2}$ | 90% |
    | $f_{\boldsymbol{\theta}_3}$ | 88% |
    | $f_{\boldsymbol{\theta}_4}$ | **91%** |
    | $f_{\boldsymbol{\theta}_5}$ | 85% |

    $$\mathrm{Accuracy}_{\mathrm{CV}}=\frac{0.86+0.90+0.88+0.91+0.85}{5}=0.88.$$

    **Usually, none.** Cross-validation evaluates a *training procedure*, not which set of fitted weights happened to score highest on its fold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refit: build the final model

    1. Evaluate candidate configurations with cross-validation.
    2. Choose the configuration $\lambda^*$ with the best mean validation score.
    3. Train a **new model** using all development examples.
    4. Evaluate that model on the held-out test set.

    $$\boldsymbol{\theta}_{\mathrm{final}}=\operatorname{Train}(\mathcal{D}_{\mathrm{development}},\lambda^*).$$

    The five cross-validation models are temporary. Retaining them and averaging their predictions would be **ensemble learning**, a different approach.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing the right cross-validation strategy

    | Strategy | Use case |
    |---|---|
    | **Stratified K-fold** | Approximate class proportions in each fold |
    | **Group K-fold** | Keep related examples (e.g. one patient) in the same group |
    | **Time-series split** | Respect time order; do not train on the future |

    **Early stopping:** use an inner validation subset if the outer validation fold is intended as an unbiased evaluation fold.

    $K=5$ and $K=10$ are common choices. More folds generally mean more training runs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameter selection

    **Recall: Support Vector Machines (SVMs).** An SVM finds a decision boundary that separates classes with a large margin. Some training points, called *support vectors*, determine that boundary.

    With the **radial basis function (RBF) kernel**, the boundary can be non-linear. The kernel measures similarity between two feature vectors:

    $$K(\mathbf{x}_i,\mathbf{x}_j)=\exp\left(-\gamma\lVert\mathbf{x}_i-\mathbf{x}_j\rVert^2\right).$$

    - **$C$ (regularization):** small $C$ allows more margin violations and favors a wider, smoother margin; large $C$ penalizes violations more strongly and tries to fit training data more closely.
    - **$\gamma$ (kernel width):** small $\gamma$ means each point influences a wider region, usually giving a smoother boundary; large $\gamma$ means more localized influence and potentially a more complex boundary.

    $C$ and $\gamma$ are **hyperparameters**, not learned SVM coefficients. We select them using cross-validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    C_choice = mo.ui.dropdown(options=["0.1", "1", "10", "100"], value="10", label="C")
    return (C_choice,)


@app.cell(hide_code=True)
def _(mo):
    gamma_choice = mo.ui.dropdown(options=["0.001", "0.01", "0.1"], value="0.01", label="Gamma")
    return (gamma_choice,)


@app.cell(hide_code=True)
def _(C_choice, gamma_choice, mo, np, plt):
    _scores = np.array([[79,82,80],[83,87,85],[86,91,88],[85,89,84]])
    _cs, _gammas = ['0.1','1','10','100'], ['0.001','0.01','0.1']
    _fig, _ax = plt.subplots(figsize=(7.5, 3.7))
    _ax.imshow(_scores, cmap='Blues', vmin=75, vmax=95)
    for _i in range(4):
        for _j in range(3):
            _ax.text(_j, _i, f'{_scores[_i, _j]}%', ha='center', va='center', fontsize=15, color='white' if _scores[_i,_j]>85 else 'black')
    _ax.scatter([_gammas.index(gamma_choice.value)], [_cs.index(C_choice.value)], s=1000, facecolors='none', edgecolors='#e49a36', linewidths=3)
    _ax.set_xticks(range(3), labels=_gammas); _ax.set_yticks(range(4), labels=_cs)
    _ax.set(xlabel='Gamma', ylabel='C')
    _fig.tight_layout(); plt.close(_fig)
    _selected = int(_scores[_cs.index(C_choice.value),_gammas.index(gamma_choice.value)])
    mo.vstack([mo.md('## Grid search with five-fold CV'), mo.hstack([C_choice, gamma_choice], gap=3), mo.as_html(_fig), mo.md(f'**Selected configuration:** mean CV accuracy {_selected}%. There are **12 configurations × 5 folds = 60 temporary fits**, followed by a final refit.')], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grid search versus random search

    **Grid search** evaluates every combination in a specified grid. With five hyperparameters and ten values each, there are $10^5$ configurations.

    **Random search** samples a fixed budget of combinations, for example 30:

    $$C\sim\operatorname{LogUniform}(10^{-3},10^{3}),\qquad \gamma\sim\operatorname{LogUniform}(10^{-4},10^{1}).$$

    The selected configuration is:

    $$\lambda^*=\arg\max_{\lambda\in\Lambda}\operatorname{Accuracy}_{\mathrm{CV}}(\lambda).$$

    A large search space often makes random search attractive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data leakage

    Consider standardization:

    $$z_j=\frac{x_j-\mu_j}{\sigma_j}.$$

    **Incorrect:** compute $\mu_j$ and $\sigma_j$ on the full dataset before cross-validation.

    **Correct:** estimate preprocessing on the **training part of each fold** and apply it to that fold's validation part.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The confusion matrix

    Suppose the positive class means *patient with the condition*. On 1,000 test cases:

    | Actual ↓ / Predicted → | Positive | Negative |
    |---|---:|---:|
    | **Positive** | **TP = 80** | **FN = 20** |
    | **Negative** | **FP = 90** | **TN = 810** |

    - **True positive:** a positive case correctly detected.
    - **False negative:** a positive case missed.
    - **False positive:** a false alarm.
    - **True negative:** a negative case correctly identified.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy can be misleading

    $$\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}=\frac{890}{1000}=0.89.$$

    But if only 100 of the 1,000 patients are positive, a classifier that **always predicts negative** achieves **90% accuracy** and detects no positive cases.

    $$\mathrm{Precision}=\frac{TP}{TP+FP}=\frac{80}{170}\approx0.471,$$

    $$\mathrm{Recall}=\frac{TP}{TP+FN}=\frac{80}{100}=0.80,$$

    $$\mathrm{Specificity}=\frac{TN}{TN+FP}=\frac{810}{900}=0.90.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F1-score and balanced accuracy

    $$F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}=\frac{2TP}{2TP+FP+FN}=\frac{160}{270}\approx0.593.$$

    $$\mathrm{BalancedAccuracy}=\frac{\mathrm{Sensitivity}+\mathrm{Specificity}}{2}=\frac{0.8+0.9}{2}=0.85.$$

    More generally:

    $$F_\beta=(1+\beta^2)\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\beta^2\mathrm{Precision}+\mathrm{Recall}}.$$

    - $\beta>1$: emphasize recall.
    - $0<\beta<1$: emphasize precision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    threshold = mo.ui.slider(.05, .95, value=.50, step=.05, label="Decision threshold", show_value=True)
    return (threshold,)


@app.cell(hide_code=True)
def _(mo, np, plt, threshold):
    _pos = np.linspace(.29, .99, 100)**1.1
    _neg = np.linspace(.01, .85, 900)**1.48
    _tp = int((_pos >= threshold.value).sum())
    _fp = int((_neg >= threshold.value).sum())
    _fn, _tn = 100-_tp, 900-_fp
    _precision = _tp/(_tp+_fp) if _tp+_fp else 0
    _recall = _tp/100
    _fig, _ax = plt.subplots(figsize=(9, 3.3))
    _ax.hist(_neg, bins=np.linspace(0,1,26), alpha=.7, label='Actual negative (900)')
    _ax.hist(_pos, bins=np.linspace(0,1,26), alpha=.7, label='Actual positive (100)')
    _ax.axvline(threshold.value, color='black', linestyle='--', linewidth=2)
    _ax.set(xlabel='Score', ylabel='Number of examples')
    _ax.legend(); _fig.tight_layout(); plt.close(_fig)
    mo.vstack([mo.md('## Changing the decision threshold'), threshold, mo.as_html(_fig), mo.md(f'**TP:** {_tp} · **FN:** {_fn} · **FP:** {_fp} · **TN:** {_tn}  |  **Precision:** {_precision:.1%} · **Recall:** {_recall:.1%}'), mo.md('*Illustrative score distributions, not the same predictions as the fixed confusion matrix.*')], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    from sklearn.metrics import roc_curve as _roc_curve, precision_recall_curve as _pr_curve, roc_auc_score as _roc_auc, average_precision_score as _ap
    _pos = np.linspace(.29, .99, 100)**1.1
    _neg = np.linspace(.01, .85, 900)**1.48
    _labels = np.r_[np.ones(100), np.zeros(900)]
    _scores = np.r_[_pos, _neg]
    _fpr, _tpr, _ = _roc_curve(_labels, _scores)
    _pr, _rc, _ = _pr_curve(_labels, _scores)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.7))
    _axes[0].plot(_fpr, _tpr, linewidth=3); _axes[0].plot([0,1], [0,1], '--', color='gray')
    _axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title=f'ROC · AUC {_roc_auc(_labels, _scores):.2f}')
    _axes[1].plot(_rc, _pr, linewidth=3); _axes[1].axhline(.1, linestyle='--', color='gray')
    _axes[1].set(xlabel='Recall', ylabel='Precision', title=f'PR · AP {_ap(_labels, _scores):.2f}')
    for _ax in _axes:
        _ax.set(xlim=(0,1), ylim=(0,1.03)); _ax.grid(alpha=.15)
    _fig.tight_layout(); plt.close(_fig)
    mo.vstack([mo.md('## ROC and precision–recall curves'), mo.as_html(_fig), mo.md('**ROC-AUC** summarizes ranking, not probability calibration. **Precision–recall** is particularly informative when positives are rare; its baseline equals positive-class prevalence (10% here).')], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiclass classification

    For $C$ classes, compute metrics per class and aggregate:

    $$F_{1,\mathrm{macro}}=\frac{1}{C}\sum_{c=1}^{C}F_{1,c},$$

    $$F_{1,\mathrm{weighted}}=\sum_{c=1}^{C}\frac{n_c}{N}F_{1,c}.$$

    - **Macro:** each class receives equal weight.
    - **Weighted:** each class receives weight proportional to support.
    - **Micro:** aggregate TP, FP and FN before computing metrics.

    In single-label multiclass classification, micro-F1 equals accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regression metrics

    $$\mathrm{MAE}=\frac{1}{N}\sum_i|y_i-\hat y_i|,$$

    $$\mathrm{MSE}=\frac{1}{N}\sum_i(y_i-\hat y_i)^2,$$

    $$\mathrm{RMSE}=\sqrt{\mathrm{MSE}},$$

    $$R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.$$

    - RMSE penalizes large errors more than MAE.
    - $R^2$ compares against the mean-prediction baseline and may be negative.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing metrics for the task

    | Application | Useful measures |
    |---|---|
    | Balanced classification | Accuracy, macro-F1 |
    | Rare-class screening | Recall, precision, PR-AUC |
    | Fraud detection | Precision, recall, expected cost |
    | Ranking | ROC-AUC, average precision |
    | Reliable probabilities | Log loss, Brier score, calibration |
    | Regression | MAE, RMSE, $R^2$ |

    **Select the evaluation metric before looking at the results.** Otherwise, the metric itself becomes another tuning choice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A complete evaluation workflow

    1. Split off an **independent test set**.
    2. Put standardization and the classifier in one **pipeline**.
    3. Perform **stratified five-fold CV** on development data.
    4. Tune with **GridSearchCV**.
    5. Select the best configuration by mean CV accuracy.
    6. **Refit** a new model on all development data.
    7. Evaluate once on the test set using appropriate metrics.

    The next slide presents this protocol using a built-in scikit-learn dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_demo = mo.ui.run_button(label="Run GridSearchCV + refit + test")
    return (run_demo,)


@app.cell(hide_code=True)
def _(mo, run_demo):
    if run_demo.value:
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        _X, _y = load_breast_cancer(return_X_y=True)
        _X_dev, _X_test, _y_dev, _y_test = train_test_split(
            _X, _y, test_size=0.2, stratify=_y, random_state=8
        )
        _search = GridSearchCV(
            make_pipeline(StandardScaler(), SVC()),
            {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01]},
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=8),
            scoring="accuracy",
        )
        _search.fit(_X_dev, _y_dev)
        _pred = _search.predict(_X_test)
        _matrix = confusion_matrix(_y_test, _pred)
        _result = mo.md(
            f"**Built-in breast cancer dataset:** {len(_X_dev)} development and {len(_X_test)} test examples.\n\n"
            f"- Best settings: `{_search.best_params_}`\n"
            f"- Mean cross-validation accuracy: **{_search.best_score_:.3f}**\n"
            f"- Independent test accuracy: **{accuracy_score(_y_test, _pred):.3f}**\n"
            f"- Independent test balanced accuracy: **{balanced_accuracy_score(_y_test, _pred):.3f}**\n"
            f"- Test confusion matrix (rows: actual, columns: predicted): `{_matrix.tolist()}`"
        )
    else:
        _result = mo.md(
            "**Dataset:** 569 breast cancer examples with 30 measured features.\n\n"
            "- Reserve 20% as an independent test set.\n"
            "- Tune the SVM's $C$ and $\\gamma$ using stratified five-fold cross-validation.\n"
            "- Refit on development data and evaluate once on the test set.\n\n"
            "In the browser slide deck, click the button to run the example and see its results."
        )

    mo.vstack([mo.md("## Model selection: complete example"), run_demo, _result])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Take-home messages

    - **Training error is not generalization error.**
    - **Validation** guides choices, including early stopping.
    - **K-fold CV** evaluates configurations using temporary models.
    - **Refit** trains the final model on all development data.
    - **Test** remains independent of model selection.
    - **Metrics** should reflect the application and the costs of errors.
    - Even a correctly evaluated model can fail under **distribution shift**.

    **Our objective is not merely the highest score, but a reliable procedure for model selection and performance estimation.**
    """)
    return


if __name__ == "__main__":
    app.run()
