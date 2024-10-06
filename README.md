# Coriolis Flight Data Analysis 
### An experimental project exploring data science, c++ integration, and basic machiene learning.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Dependencies](#dependencies)
- [Interim results](#interim-results)
- [LICENSE](#license)

## Installation 
#### 1. Clone the repo
```
git clone https://github.com/janiejestemja/coriolis_flights.git
```
#### 2. Install the dependencies

```
pip install -r requirements.txt

```
#### 3. Additional setup
##### - Python 3.12.3
##### - g++ (Ubuntu 13.2.0-23ubuntu4) or higher

Make sure to have g++ installed via:
```
g++ --version
```
and if not install it via: 
```
sudo apt install g++
```
How to build:

Change directory to ```~/coriolis_flights/coriolis_module/``` and run:
```
python setup.py build_ext --inplace
```
If build is successful a ".so" will be compiled and saved in the current directory. 


*For users of virtual environments:* 

please do not forget to copy/move the .so file to the python packages location.
```
path_to/venv/lib/python/site-packages/
```

*For users not familiar with pythons virtual environments visit [python documentation](https://docs.python.org/3/library/venv.html) for more information.*


## Dependencies
#### Versions of Python libraries in use:

- numpy - 2.1.0 
- pandas - 2.2.2
- geopandas - 1.0.1 

- matplotlib - 3.9.2 
- seaborn - 0.13.2

- shapely - 2.0.6
- geodatasets - 2024.8.0

- scikit-learn - 1.5.2
- scipy - 1.14.1
- joblib - 1.4.2

#### Text-Editor: 
- Jupyter Notebook - 7.2.1

## Data
- the core dataset of the project consists of flight information for around 3 million flights, i stumbled over on kaggle.
- the additional dataset i found on kaggle after i figured out "IATA" codes are primary keys for U.S. airports to gain access to geographical coordinates.

## Interim Results
###### *"Despite our analysis our results remain questionable and insignificant." - data-scepticist.*

*Section yet to be written...*

## License 
[MIT](LICENSE.txt)
