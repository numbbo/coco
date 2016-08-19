## Folder contents ##

- `archive-update/` contains internal code for combining the archives of algorithms to create/update the hypervolume reference values for the bbob-biobj test suite

- `log-reconstruction/` contains internal code for reconstructing output of the `bbob-biobj` logger from archive files (needed when the hypervolume reference values are updated)

## Testing instructions ##

In order to test the scripts contained in `archive-update/` and `log-reconstruction/`, the `pytest` module (and possibly the `six` module) need to be installed in python. 

Using anaconda:

    conda install pytest

Using other python distributions:

    pip install pytest
    pip install six
