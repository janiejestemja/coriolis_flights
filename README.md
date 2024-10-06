# Coriolis Flight Data Analysis 
### A Learning Journey into Data Science and C++ Integration,
####  and basics of machiene learning.
###### An experimental learning project.

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
- Python 3.12.3
- g++ (Ubuntu 13.2.0-23ubuntu4) 13.2.0


How to build:

Change directory to ~/coriolis_module/ and run:
```
python setup.py build_ext --inplace
```
If build is successful a .so will be compiled and stored in the current directory. 


*Reminder for users of virtual environments:* 

do not forget to copy/move the .so file to the python packages location.
```
path_to/venv/lib/python/site-packages/
```

## 4. Data
- the core dataset of the project consists of flight information for around 3 million flights i stumbled over on kaggle.
- the additional dataset i found on kaggle after i figured out "IATA" codes are primary keys for U.S. airports to gain access to geographical coordinates.

## 5. Interim Results
###### *"Despite our analysis our results remain questionable and insignificant."*

*Section yet to be written...*

## 6. License 
[MIT](LICENSE)
