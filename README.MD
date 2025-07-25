# Pattern Recognition

THIS PAGE IS UNDER CONSTRUCTION

This is the collection of notebooks for the course *Pattern Recognition*.
You can view and execute the notebooks by clicking on the buttons below.

## 📑 Content

1. **Introduction to pattern recognition** 
   - What is pattern recognition?
   - Applications of pattern recognition
   - Practical information

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/Wickstrom/pattern-recognition-handbook/blob/main/notebooks/00/intro.ipynb)

<br>

2. **Bayes decision theory** 
   - Motivation
   - Naïve Bayes assumption
   - Bayes classifier
   - Probablity of error
   - Dealing with risk

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/Wickstrom/pattern-recognition-handbook/blob/main/notebooks/01/bayes_decision_theory.ipynb)

<br>

3. **Density estimation I** 
   - Parametric density estimation
   - Likelihood function
   - Maximum-likelihood estimation
   - Non-parametric density estimation
   - Histogram approach
   - Parzen windowing

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/Wickstrom/pattern-recognition-handbook/blob/main/notebooks/02/density_estimation.ipynb)

<br>

4. **Density estimation II** 
   - K nearest neighbor
   - Mixture models

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/Wickstrom/pattern-recognition-handbook/blob/main/notebooks/02/mixture_models.ipynb)

<br>

5. **Linear classifiers I** 
   - Mean squared error (MSE) classifier
   - Least sum of squares classifier
   - Widrow-Hoff algorithm
   - Perceptron

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/Wickstrom/pattern-recognition-handbook/blob/main/notebooks/03/linear_classifiers.ipynb)

<br>

4. **Linear classifiers II** 
   - Support vector machines

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/Wickstrom/pattern-recognition-handbook/blob/main/notebooks/03/support_vector_machine.ipynb)

<br>

## 💻 How to code locally

To run the notebooks locally the recommended steps are the following:

1. Download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

2. Download the [env.yml](https://github.com/FilippoMB/python-time-series-handbook/blob/main/env.yml) file.

3. Open the shell and navigate to the location with the `yml` file you just downloaded.
    - If you are on Windows, open the Miniconda shell.

4. Install the environment with 
   ```{bash}
   > conda env create -f env.yml
   ```

5. Activate your environment: 
   ```{bash}
   > conda activate pytsa
   ```

6. Go to the folder with the notebooks

7. Launch Jupyter Lab with the command 
   ```{bash}
   > jupyter lab
   ```

## 🎥 Notebook format and slides

The notebooks are structured as a sequence of slides to be presented using [RISE](https://rise.readthedocs.io/en/latest/).
If you open a notebook you will see the following structure:

<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_nb.png" style="width: 50%" align="center">

The top-right button indicates the type of slide, which is stored in the metadata of the cell. To enable the visualization of the slide type you must first install RISE and then on the top menu select `View -> Cell Toolbar -> Slieshow`. Also, to split the cells like in the example, you must enable `Split Cells Notebook` from the [nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/index.html).

By pressing the `Enter\Exit RISE Slideshow` button at the top you can enter the slideshow presentation.

<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_rise.png" style="width: 40%" align="center">
<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_rise2.png" style="width: 40%" align="center">
<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_rise3.png" style="width: 40%" align="center">

See the [RISE documentation](https://rise.readthedocs.io/en/latest/) for more info.

## 📝 Citation

If you are using this material in your courses or in your research, please consider citing it as follows:

````bibtex
@misc{bianchi2024tsbook,
  author       = {Kristoffer Wickstrøm},
  title        = {Pattern Recognition Course},
  year         = {2025},
  howpublished = {Online},
  url          = {https://github.com/Wickstrom/pattern-recognition-handbook}
}
````

This repository is heavily based on [Time Series Analysis with Python](https://github.com/FilippoMB/python-time-series-handbook) by my collegue at UiT Filippo Maria Bianchi. If you find my repository useful, please head over there and give him a star.

````bibtex
@misc{bianchi2024tsbook,
  author       = {Filippo Maria Bianchi},
  title        = {Time Series Analysis with Python},
  year         = {2024},
  howpublished = {Online},
  url          = {https://github.com/FilippoMB/python-time-series-handbook}
}
````
