# Coriolis Flight Data Analysis 
### An experimental project exploring data science, c++ integration, and machiene learning.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Dependencies](#dependencies)
- [Interim results](#interim-results)
- [LICENSE](#license)

# Installation 
  1. Clone the repo
```bash
git clone https://github.com/janiejestemja/coriolis_flights.git
```
  2. Install the dependencies
```bash
pip install -r requirements.txt
```
  3. Additional setup
     - Python 3.12.3
     - g++ (Ubuntu 13.2.0-23ubuntu4) or higher

     1. Debian
        - *Checking for compiler*


        Make sure to have g++ installed via:
        ```bash
        g++ --version
        ```
        and if not install it via: 
        ```bash
        sudo apt install g++
        ```


        - *Checking for "<Python.h>"*


        Before building run 
        ```bash
        dpkg -l | grep python3-dev
        ```
        to check if necessary header for the c++ is in place.


        If not install it via: 
        ```bash
        sudo apt install python3-dev
        ```


     2. Fedora
        - *Checking for compiler*


        Make sure to have g++ installed via:
        ```bash
        g++ --version
        ```
        and if not install it via: 
        ```bash
        sudo dnf install g++
        ```


        - *Checking for "<Python.h>"*


        Before building run 
        ```bash
        dnf list installed | grep python3-devel
        ```
        to check if necessary header for the c++ is in place.


        If not install it via: 
        ```bash
        sudo dnf install python3-devel
        ```


## Building the module


Change directory to ```~/coriolis_flights/coriolis_module/``` and run:
```bash
python setup.py build_ext --inplace
```
If build is successful a ".so" will be compiled and saved in the current directory. 


*For users of virtual environments:* 
please do not forget to copy/move the .so file to the python packages location.
```bash
path_to/venv/lib/python/site-packages/
```
*For users not familiar with pythons virtual environments visit [python documentation](https://docs.python.org/3/library/venv.html) for more information.*


# Dependencies
## Versions of Python libraries in use:
- numpy - 2.1.0 
- pandas - 2.2.2
- geopandas - 1.0.1 
- geodatasets - 2024.8.0
- scikit-learn - 1.5.2

## Version of Text-Editor: 
- Jupyter Notebook - 7.2.1


# Data
- the core dataset of the project consists of flight information for around 3 million flights.
- the additional dataset is for geographical coordinates correspondend to the airports in (most) of the core dataset.


# Interim Results
###### *"Despite our analysis our results remain questionable and insignificant." - data-scepticist.*


**Evaluations are experimental, use with caution.**



### Coriolis drift calculations [^1] [^2]
2,730,145 : Rows in dataframe at time of evaluation.


- 3,506,000,878.64 km : total travelled distance
- 928,589,014.17 km : total drifted distance
- 0.2649 : average factor 
- 26.49% : average percentage for drift per distance


### C++ implementation
Most of the functions in **coriolis_functions.py** have a *translation* in **coriolis_module.cpp** in addition to corresponding wrapper-functions. 


- Calculations in *"pure"* python ran for 5,468.66 seconds (92.81 minutes).[^1] 
- Calculations in c++ ran for 1,671.03 seconds (27.85 minutes).[^2]


### Machiene learning


Does delay at departure affect delay at arrival?
- per one minute of delay at departure the delay at arrival increases by one minute as well (evaluation of slope) [^3] [^4].
- the intercept of -5.91 can be evaluated or interpreted as follows: *"If a flight-departure has a delay of -5.91 minutes this flight will have no delay at arrival."* or *"For a flight to arrive without delay the departure has be 5.91 minutes early."* [^3] [^4]


[^1]: See ```coriolis_analysis_02.ipynb```.
[^2]: See ```coriolis_analysis_02b.ipynb```.
[^3]: See ```coriolis_analysis_3.ipynb```.
[^4]: Both models used for this prediction had a [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) of R² = 0.93.


*Section yet to be written...*


## License 
[MIT](LICENSE.txt)
