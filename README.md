# Crop Clasification
Machine learning algorithms to classify crops using bi-temporal data fusion of SAR and optical data.

## Installation
### Create virtual environment and install dependencies
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ pip install .
```
To check the installation:
```
$ pip freeze | grep crop_classification
```

All dependencies can be found in `requirements.txt`.


## Reproducing the results
```
$ softmax
$ ANN
$ decision_tree
$ random_forest
```