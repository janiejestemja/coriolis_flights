# Coriolis Flight Data Analysis 
### An experimental project exploring data science, c++ integration, and basic machiene learning.

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


## *Actually building the module*


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


Evaluations are experimental, use with caution.


2,730,145 : Rows in dataframe at time of evaluation.


### Coriolis drift calculations
- 3,506,000,878.64 : total travelled distance [km]
- 928,589,014.17 : total drifted distance [km]
- 0.2649 : average factor 
- 26.49% : average percentage for drift per distance


### C++ implementation
Most of the functions in **coriolis_functions.py** have a *translation* in **coriolis_module.cpp** in addition to corresponding wrapper-functions. 


- **coriolis_analysis_02.ipynb** does calculations in *"pure"* python, and ran for 5,468.66 seconds (92.81 minutes). 
- **coriolis_analysis_02b.ipynb** does most of the calculations in c++, and ran for 1,671.03 seconds (27.85 minutes).


##### To understand the term 'calculations' further feel free to take a look at the .ipynb files mentioned.


### Machiene learning
*Section yet to be written...*


## License 
[MIT](LICENSE.txt)
