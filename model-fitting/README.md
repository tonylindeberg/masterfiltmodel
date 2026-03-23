#  Modelling the 8 ”master key filters” with idealized discrete scale-space filters

The Jupyter notebooks [keyfilteranal.ipynb](https://github.com/tonylindeberg/masterfiltmodel/blob/main/model-fitting/keyfilteranal.ipynb) and [keyfilteranal-x1.ipynb](https://github.com/tonylindeberg/masterfiltmodel/blob/main/model-fitting/keyfilteranal-x1.ipynb) contain the Python code that was used for:

* computing the statistical measures of characteristic properties of the 8 ”master key filters” in Section 3, and</li>

* computing the filter parameters for the corresponding discrete scale-space filters that are proposed as idealized models of the ”master key filters” in Section 4</li>

in the following paper:
<p>

* Lindeberg, Babaiee and Kiasari (2025) "Modelling and analysis of the 8 filters from the 'master key filters hypothesis' for depthwise-separable deep networks in relation to idealized receptive fields based on scale-space theory", Journal of Mathematical Imaging and Vision, to appear. ([preprint at arXiv: 2509.12746](https://doi.org/10.48550/arXiv.2509.12746))

The Python file [filteranal.ipynb](https://github.com/tonylindeberg/masterfiltmodel/blob/main/model-fitting/filteranal.ipynb) contains a set of library functions used in the Jupyter notebooks.

<p>
Beyond standard library functions, this code depends on the pyscsp library available from GitHub at https://github.com/tonylindeberg/pyscsp or via PyPi: ”pip install pyscsp”.
<p>
Please, note, however, that this code has not been refactored into a fully automated processing chain. Instead, the original code used for the model fitting underlying the article is provided here.
