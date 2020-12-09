# Wildfire-TPU

Percolation model of wildland fire spread implemented in TensorFlow.

Individual fire cases are created as instances of the class FireCase. FireCases must be assembled into a FireBatch to be run. See `single.py` for an example of setting up and running a single case.

The script `ijwf.py` was used to generate the *wind* and *wind-slope* datasets, and `ijwf_changewind.py` was used for the *dynamic wind* and *realistic* datasets.