# AGENTS.md

Lecture notebook repo for FYS-3012 Pattern Recognition (UiT). Content, not software — no test/lint/typecheck suite and no `Makefile`. Optimize edits for legibility on projected slides and reproducibility of figures.

## Commands

```bash
uv sync                                       # install deps into .venv from pyproject.toml + uv.lock
uv run jupyter lab                            # open exercise notebooks in a browser
uv run marimo edit notebooks/01/bayes_decision_theory.py   # edit a single deck locally
uv run marimo export html-wasm <path>.py -o /tmp/<name> --mode run --no-show-code  # local WASM build
```

`uv run marimo edit` on a single `.py` reads its PEP 723 header and resolves deps into an ephemeral venv — no `uv sync` required for one-off exploration. The same PEP 723 header is what makes the file runnable in the browser via the exported WASM bundle.

There is no `make`, `pytest`, `ruff`, `mypy`, `pre-commit`, etc. CI (`.github/workflows/publish-slides.yml`) is the only "build".

## Layout

- `notebooks/<NN>/` — one folder per lecture (`00`–`13`). Each holds one or more notebooks, an optional `media/` (images, `.npz` datasets), and (for Marimo decks) a `layouts/<base>.slides.json`.
- `notebooks/_shared/math.css` — shared KaTeX font-size overrides; referenced via `css_file="../_shared/math.css"` in the marimo `App(...)` constructor of NN3, kernel methods, linear DR, non-linear DR. marimo's `export html-wasm` inlines it into the generated HTML — no separate asset to ship.
- `pyproject.toml` + `uv.lock` — project deps.
- `index.html` — landing page served at the GitHub Pages root; lists each deck. **`README.MD` is the canonical content index** — update there first. `index.html` can lag.
- `_config.yml` — leftover Jupyter Book config; **not used** by the current pipeline (CI uses raw `nbconvert` + `marimo export`, not `jupyter-book build`). Safe to leave alone; do not "fix" it expecting it to drive a build.

## Two notebook formats coexist for the same lecture

Each lecture that has been migrated to Marimo ships **both** files in the same folder:

| File | Format | Purpose |
| --- | --- | --- |
| `<name>.ipynb` | Jupyter | Human-authored source; content-equal to the `.py`. Used by `nbconvert --to slides` **only** if no `.py` sibling exists. |
| `<name>.py` | Marimo (PEP 723 header) | Reactive runtime — uses `mo.ui.slider`, `mo.ui.dropdown`, etc. instead of baked-in animations. Layout is configured via `layouts/<name>.slides.json` and read at export time. |
| `<name>_exercise.ipynb` | Jupyter | Student-facing; **always skipped by CI**. Never ship a `.py` for these. |

When editing a migrated lecture, keep `.ipynb` and `.py` content-equal in text and figures — the README explicitly states this.

## Marimo cell-pattern gotcha

Cells written as:

```python
@app.cell
def _(mo):
    mo.md(r"""# Title""")
    return
```

…work correctly. Marimo's parser strips the trailing `return` and treats the previous expression (`mo.md(...)`) as the cell's `last_expr`, so the cell has a real output. Writing the explicit `return mo.md(...)` is a **SyntaxError** (the cell body is at module level after the function wrapper is stripped). Do not "fix" the trailing `return` into `return mo.md(...)`.

Each name must be owned by exactly one cell — Jupyter-style redefinition is rejected. When porting from `.ipynb`, suffix cell-local datasets/figures (e.g. `X_xor`, `fig_xor`) to avoid future collisions.

## CI rules (`.github/workflows/publish-slides.yml`)

Trigger: push to `main`/`master`, or manual dispatch. Two jobs: `build` → `deploy`.

For each `notebooks/*/*.ipynb`:
1. Skip if basename matches `*_exercise` (student notebooks).
2. Skip if a sibling `.py` exists (the Marimo version will be exported instead).
3. Otherwise: `jupyter nbconvert --to slides` into `_slides/<dir>/<base>.slides.html`, after copying `<dir>/media/` alongside it.

For each `notebooks/*/*.py`:
1. `marimo export html-wasm <nb> -o _slides/<dir>/<base> --mode run --no-show-code` — output is a directory containing `index.html`, `assets/`, `.nojekyll`. The notebook's `layout_file` is inlined as a base64 data URI in the HTML.
2. Copy `<dir>/media/` into the output dir so relative image paths resolve.

Then: inject cell-tag CSS for `celltag_remove-input` / `celltag_remove-cell` (nbconvert doesn't ship it), ensure `_slides/.nojekyll` exists, copy `index.html` in, upload as Pages artifact, deploy via `actions/deploy-pages@v5`.

`concurrency.group: pages, cancel-in-progress: false` prevents racing deploys.

**Marimo 0.23 CLI quirk:** `html_head_file=` and `css_file=` in `marimo.App(...)` resolve relative paths against the notebook's *basename*, not the source directory, so `../_shared/math.css` silently no-ops in the WASM bundle. That's why `math.css` is *also* loaded at export time by the same CI step (the CI snippet inlines its contents into the generated HTML), and why new math-heavy decks should follow the same dual-load pattern rather than relying on the App config.

## Repo-specific conventions

- **Figure stability**: notebooks that roll random samples call `np.random.seed(0)` (or similar) at the top. Do not remove or change the seed without expecting figures to reshuffle on re-render.
- **Marimo chdir**: Marimo `.py` notebooks chdir to their own folder at startup so `mo.image(src="media/foo.png")` resolves the same in `marimo edit` as in the deployed WASM. The `if "__file__" in globals() and __file__:` guard makes this a no-op under WASM. Preserve this pattern when adding new decks.
- **Shared math CSS**: NN3, kernel methods, linear DR, non-linear DR load `notebooks/_shared/math.css` for larger KaTeX. Add new math-heavy decks here too.
- **Datasets**:
  - Synthetic inline (Gaussians, uniform, XOR, cosine, hand-picked 2-D points).
  - `sklearn.datasets.make_moons`, `make_swiss_roll`, `load_wine` — no download.
  - Iris via `ucimlrepo.fetch_ucirepo(id=53)` — **needs internet on first run**, cached afterwards.
- **Exercise notebooks have no `# TODO` markers**. The question text + starter cell are the only signal of where students should add code. Don't add placeholders.
- **Theoretical exercises** are deliberately not in this public repo (copyright). Distributed separately through the private course portal.

## Deploy target

GitHub Pages at `https://wickstrom.github.io/pattern-recognition-handbook/`. Marimo decks render as WASM (Python + deps ship in the page); nbconvert decks are static Reveal.js. Both are tagged in `index.html` accordingly.

## Math notation conventions

- Bold vectors/matrices via `\mathbf{}` for Latin, `\boldsymbol{}` for Greek.
- Display math (`$$...$$` on its own line) for prominent equations (matrices, derivatives, key ratios); inline `$...$` for text-embedded math.
- `\limits` on `\sum` when summands should appear above/below (`\sum\limits_{i=1}^{M}`).
- No blank lines between bullets and `$$...$$` blocks in markdown cells (terminates the list and renders math as inline `<span>`).
- No stray `= $` at end of line (opens math mode that's never closed within the line).
- Use `\operatorname{trace}` rather than `\text{trace}`.
