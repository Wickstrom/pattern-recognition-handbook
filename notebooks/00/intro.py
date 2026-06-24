# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///
#
# Marimo version of the introduction lecture.
# Same content as notebooks/00/intro.ipynb, but authored as a reactive
# Marimo app for the WASM build on GitHub Pages. This notebook is purely
# text + media (no Python computation), so the conversion is a 1:1 port
# of the markdown cells.
# Run locally with `marimo edit notebooks/00/intro.py` or export to WASM
# for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/intro.slides.json",
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
    # Pattern Recognition

    We live in an age of data abundance. Data of various types such as imagery, text, graphs, tabular, time series, and more are being collected on a daily basis across a wide range of societal settings. This data has hidden information that can be leveraged to improve problem solving, inform decision making, make scientific discoveries, and many other possibilities. But how do we find and extract information from data? Pattern recognition is the scientific discipline whose goal is the classification of objects into a number of cateogries and classes. Pattern recognition is an integral part of mist machine intelligence systems built for decision making, and provides essential tools for analyzing and understanding data.

    This course will give an in-depth introduction to the disipline of pattern recognition, and provide students with a wide range of essential tools that any machine learning practitioner should have in their tool belt.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Overview of course

    - The course into 5 main components:
        - Bayes decision theory.
        - Linear classifiers.
        - Non-linear classifiers.
        - Data transformations.
        - Clustering.
    - Content for each component is contained in notebooks. The notebooks
      are presented in class as slides made with [Marimo](https://marimo.io)
      and hosted with [GitHub Pages](https://pages.github.com/). For this
      reason, the text in the notebooks is organized with bullet points.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What will you have at the end of this course?

    1. You will have a deep understanding of the fundamental techniques and algorithms that are frequently used in practice:
        - "The first thing you try"-techniques like linear classifiers, k-means clustering, and principal component analysis will be covered in-depth and implemented from scratch.
    2. You will learn how to think more theoretically about how to approach a data analysis task.
        - Detailed derivations of complex topics like kernel methods, neural networks, and Laplacian Eigenmaps.
    3. You will derive state-of-the-art algorithms from first principles.
        - Derivation of widely used algorithms like Support Vector Machines.
    4. You will have hands-on experience from working with diverse data and tasks.
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Applications of pattern recognition - PCA in computer vision

    [Video from DINOv2 repository](https://github.com/facebookresearch/dinov2)
                """
            ),
            mo.video(src="media/dino.mp4", controls=True, style={"max-width": "100%", "height": "auto"}),
        ],
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Applications of pattern recognition - Temperature scaled softmax in GPT

    [Image from 3Blue1Brown.](https://www.3blue1brown.com/lessons/gpt)
                """
            ),
            mo.image(src="media/gptsoftmax.png", width="100%"),
        ],
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Applications of pattern recognition - Bayes decision theory in image generation models

    [Image from GroundingDino](https://github.com/IDEA-Research/GroundingDINO/tree/main) and from [Lil'log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
                """
            ),
            mo.image(src="media/grounding_dino.png", width="100%"),
            mo.image(src="media/diffusion1.png", width="100%"),
            mo.image(src="media/diffusion2.png", width="100%"),
        ],
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### University courses

    These notebooks are currently adopted in [FYS-3012 Pattern Recognition](https://uit.no/utdanning/emner/emne/873965/fys-3012) at UiT the Arctic University of Tromsø.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Roadmap

    This is an early version of the course. There might be imprecisions and errors. Also, some chapters might undergo significant revisions and changes.

    I am planning to add more content over time to cover additional topics in new chapters and to extend the existing ones with new examples and explanations. If there is a specific topic you feel is missing or passages that you feel are not clear enough, open an <a class="github-button" href="https://github.com/Wickstrom/pattern-recognition-handbook/issues" data-color-scheme="no-preference: light; light: light; dark: dark;" data-icon="octicon-issue-opened" aria-label="Issue FilippoMB/python-time-series-handbook on GitHub">Issue</a> on the repository on Github.

    ### A note on deep learning

    This course intentiaonally avoids advanced deep learning topics. This is mainly to not overlap with the dedicated deep learning course at UiT [FYS-3033 Deep Learning](https://en.uit.no/education/courses/course?p_document_id=859747). However, the foundation of deep learning (neural networks) is introduced from first principles. We go all the way from the Perceptron to mulitlayer perceptrons, coding everything from scratch. That means, those who follow this course will have a fundamental understanding of how neural networks work. Then, the dedicated deep learning course expands on this to cover the fundamental and most recent topics within the field of deep learning.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Practical information for 2026 version of UiT course
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Teaching staff

    1. Associate professor Kristoffer Wickstrøm - course responsible
    2. Associate professor Veronica Lachi - co-leader of the course
    3. PhD candidate Riccardo Gelato - teaching assistant
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Lecture and exercises

    1. Lectures:
        - Fridays from 08:15 - 10:00
        - Mondays from 10:15 - 12:00
    2. Exercises:
        - Thursdays from 10:15 - 12:00
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Exam

    1. Home exam counting 50% of grade:
        - Starts in week 40, handed in in week 42.
    2. School exam in week 49 counting 50% of grade.
        """
    )
    return


if __name__ == "__main__":
    app.run()
