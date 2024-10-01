#include <Python.h>
#include <cmath>
#include <vector>

// Function to calculate the direction vector between two lat/lon points
std::vector<double> direction_vector(double lat1, double lon1, double lat2, double lon2) {
    lat1 = lat1 * M_PI / 180.0;
    lon1 = lon1 * M_PI / 180.0;
    lat2 = lat2 * M_PI / 180.0;
    lon2 = lon2 * M_PI / 180.0;

    double x1 = std::cos(lat1) * std::cos(lon1);
    double y1 = std::cos(lat1) * std::sin(lon1);
    double z1 = std::sin(lat1);

    double x2 = std::cos(lat2) * std::cos(lon2);
    double y2 = std::cos(lat2) * std::sin(lon2);
    double z2 = std::sin(lat2);

    std::vector<double> direction = {x2 - x1, y2 - y1, z2 - z1};
    
    double norm = std::sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]);
    direction[0] /= norm;
    direction[1] /= norm;
    direction[2] /= norm;
    
    return direction;
}

// Wrapper function
static PyObject* py_direction_vector(PyObject* self, PyObject* args) {
    double lat1, lon1, lat2, lon2;
    if (!PyArg_ParseTuple(args, "dddd", &lat1, &lon1, &lat2, &lon2)) {
        return nullptr;
    }
    std::vector<double> direction = direction_vector(lat1, lon1, lat2, lon2);
    return Py_BuildValue("ddd", direction[0], direction[1], direction[2]);
}

// Method definitions for the module
static PyMethodDef CoriolisMethods[] = {
    {"direction_vector", py_direction_vector, METH_VARARGS, "Calculate direction vector between two lat/lon points"},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// Module definition
static struct PyModuleDef coriolismodule = {
    PyModuleDef_HEAD_INIT,
    "coriolis_module", // Module name
    nullptr,          // Module documentation
    -1,               // Size of per-interpreter state of the module
    CoriolisMethods   // Methods defined in the module
};

// Module initialization function
PyMODINIT_FUNC PyInit_coriolis_module(void) {
    return PyModule_Create(&coriolismodule);
}
