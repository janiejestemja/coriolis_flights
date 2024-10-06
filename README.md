# Coriolis Flight Data Analysis – A Learning Journey in Data Science and C++ Integration.
An experimental learning project - correctness not guaranteed.

## Installation 
1. Clone the repo
```
git clone https://github.com/janiejestemja/coriolis_flights.git
```
2. Install the dependencies:
```
pip install -r requirements.txt
```
3. Additional setup:
Tested with:
- Python 3.12.3
- g++ (Ubuntu 13.2.0-23ubuntu4) 13.2.0
How to build:

Change directory to ~/coriolis_module/ and run:
```
python setup.py build_ext --inplace
```
If build is successful a .so will be compiled and stored in the current directory. 

Reminder for users of virtual environments: 
do not forget to copy/move the .so file to your packages.
```
path_to/venv/lib/python/site-packages/
```

