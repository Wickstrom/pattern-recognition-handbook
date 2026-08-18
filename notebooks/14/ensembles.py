# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///
#
# Ensembles lecture — placeholder deck (under development).
#
# This is a placeholder notebook: the actual slides and exercises for this
# lecture are not written yet. It exists so the lecture appears in the
# course index and the CI slide build has something to render.
#
# Run locally with `marimo edit notebooks/14/ensembles.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/ensembles.slides.json",
)


@app.cell
def _():
    import marimo as mo
    return mo


@app.cell
def _(mo):
    mo.md(
        r"""
    # Ensemble methods

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 3</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    > **Under development.** This lecture is not yet written. It will cover
    > combining multiple models (bagging, random forests, boosting, stacking)
    > into stronger predictors for classification and regression.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 3</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Planned topics

    - Bagging
    - Random forests
    - Boosting
    - Stacking

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 3</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
