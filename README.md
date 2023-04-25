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

## Data
The dataset used for this study can be downloaded from the below link:
```
https://archive.ics.uci.edu/ml/machine-learning-databases/00525/data.zip

 @article{khosravi_2020, place={Tehran, I.R. Iran}, title={Crop mapping using fused optical-radar data set}, publisher={Department of Remote Sensing &amp; GIS, Faculty of Geography, University of Tehran}, author={Khosravi, Iman}, year={2020}, month={Jun}} 
```
Unzip the downloaded file and place it under `./data`.

## Reproducing the results
```
$ softmax
$ ANN
$ decision_tree
$ random_forest
```