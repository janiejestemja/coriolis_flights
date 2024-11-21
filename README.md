# Coriolis Flights 

---

*How to uncover insignificant truths that don't matter...*

---
## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [License](#license)

---

# Dependencies
## General Dependecies
- `Fedora` workstation 41
- `g++` (GCC) 14.2.1 (Red Hat 14.2.1-3)
- `Python` - 3.13.0

### Versions of libraries in use
- `setuptools` - 75.3.0
- `numpy` - 2.1.3

---

# Installation
## Building the C++ Module

To build the module:

1. Ensure `g++` is installed:
   ```bash
   g++ --version
2. Verify necessary headers are in place with:
   ```bash
   dnf list --installed | grep python3-devel
   ```
3. Change to the directory:
   ```bash
   cd ~/coriolis_flights/coriolis_module/
   ```
4. Run setup script:
   ```bash
   python setup.py build_ext --inplace
   ```
   If successful, a `.so` file will be compiled in the directory.

---

# How to use

## The Python Module
```python
from coriolis_functions import coriolis_functions
```

## The C++ Module
```python
from coriolis_module import coriolis_module
```

---

# License 
[MIT](LICENSE.txt)