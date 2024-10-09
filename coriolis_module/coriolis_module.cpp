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

    double norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    direction[0] /= norm;
    direction[1] /= norm;
    direction[2] /= norm;

    return direction;
}

// Wrapper for direction_vector
static PyObject* py_direction_vector(PyObject* self, PyObject* args) {
    double lat1, lon1, lat2, lon2;
    if (!PyArg_ParseTuple(args, "dddd", &lat1, &lon1, &lat2, &lon2)) {
        return nullptr;
    }
    std::vector<double> direction = direction_vector(lat1, lon1, lat2, lon2);
    return Py_BuildValue("ddd", direction[0], direction[1], direction[2]);
}

// Function to calculate the rotation matrix based on latitude
std::vector<std::vector<double>> rotation_matrix(double latitude) {
    double lat_rad = latitude * M_PI / 180.0; // Convert latitude to radians
    const double omega = 7.2921159e-5; // rad/s

    return {
        {0, -omega * std::cos(lat_rad), omega * std::sin(lat_rad)},
        {omega * std::cos(lat_rad), 0, 0},
        {-omega * std::sin(lat_rad), 0, 0}
    };
}

// Wrapper for rotation_matrix
static PyObject* py_rotation_matrix(PyObject* self, PyObject* args) {
    double latitude;
    if (!PyArg_ParseTuple(args, "d", &latitude)) {
        return nullptr;
    }
    auto matrix = rotation_matrix(latitude);
    return Py_BuildValue("ddd(ddd)", 
        matrix[0][0], matrix[0][1], matrix[0][2],
        matrix[1][0], matrix[1][1], matrix[1][2],
        matrix[2][0], matrix[2][1], matrix[2][2]);
}

// Function to calculate Earth's radius at a given latitude
double radius_at_latitude(double lat) {
    double lat_rad = lat * M_PI / 180.0; // Convert latitude from degrees to radians
    return 6371.0 * (1 - 0.5 * (1 - std::cos(lat_rad))) + 
           6378.137 * (0.5 * (1 - std::cos(lat_rad)));
}

// Wrapper for radius_at_latitude
static PyObject* py_radius_at_latitude(PyObject* self, PyObject* args) {
    double lat;
    if (!PyArg_ParseTuple(args, "d", &lat)) {
        return nullptr;
    }
    double radius = radius_at_latitude(lat);
    return Py_BuildValue("d", radius);
}

// Coriolis acceleration calculation
std::vector<std::vector<double>> coriolis_acc(double lat1, double lat2, const std::vector<double>& scaled_direction, double time, int num_steps) {
    std::vector<std::vector<double>> coriolis_accelerations(num_steps, std::vector<double>(3));

    for (int i = 0; i < num_steps; ++i) {
        double t = (time * i) / (num_steps - 1);
        double current_latitude = lat1 + (t / time) * (lat2 - lat1);
        
        auto R = rotation_matrix(current_latitude);
        // Calculate Coriolis acceleration
        for (int j = 0; j < 3; ++j) {
            coriolis_accelerations[i][j] = R[j][0] * scaled_direction[0] +
                                            R[j][1] * scaled_direction[1] +
                                            R[j][2] * scaled_direction[2];
        }
    }
    return coriolis_accelerations;
}

// Wrapper for coriolis_acc
static PyObject* py_coriolis_acc(PyObject* self, PyObject* args) {
    double lat1, lat2, time;
    int num_steps;
    PyObject* py_scaled_direction;

    if (!PyArg_ParseTuple(args, "ddOdi", &lat1, &lat2, &py_scaled_direction, &time, &num_steps)) {
        return nullptr;
    }

    // Extract scaled_direction from Python list or tuple
    std::vector<double> scaled_direction;
    if (PyList_Check(py_scaled_direction) || PyTuple_Check(py_scaled_direction)) {
        for (Py_ssize_t i = 0; i < PyList_Size(py_scaled_direction); ++i) {
            scaled_direction.push_back(PyFloat_AsDouble(PyList_GetItem(py_scaled_direction, i)));
        }
    }

    auto accelerations = coriolis_acc(lat1, lat2, scaled_direction, time, num_steps);
    
    // Build Python list of lists for output
    PyObject* py_result = PyList_New(num_steps);
    for (size_t i = 0; i < accelerations.size(); ++i) {
        PyObject* py_inner = PyList_New(3);
        for (size_t j = 0; j < 3; ++j) {
            PyList_SetItem(py_inner, j, Py_BuildValue("d", accelerations[i][j]));
        }
        PyList_SetItem(py_result, i, py_inner);
    }
    
    return py_result;
}

// Calculate velocities based on Coriolis acceleration
std::vector<std::vector<double>> calculate_velocities(const std::vector<std::vector<double>>& coriolis_acceleration, double time, int num_steps) {
    std::vector<std::vector<double>> coriolis_velocity(num_steps, std::vector<double>(3, 0.0));

    for (int i = 1; i < num_steps; ++i) {
        double dt = time / (num_steps - 1);
        for (int j = 0; j < 3; ++j) {
            coriolis_velocity[i][j] = coriolis_velocity[i - 1][j] + 
                0.5 * (coriolis_acceleration[i][j] + coriolis_acceleration[i - 1][j]) * dt;
        }
    }
    return coriolis_velocity;
}

// Wrapper for calculate_velocities
static PyObject* py_calculate_velocities(PyObject* self, PyObject* args) {
    PyObject* py_coriolis_acceleration;
    double time;
    int num_steps;

    if (!PyArg_ParseTuple(args, "OdI", &py_coriolis_acceleration, &time, &num_steps)) {
        return nullptr;
    }

    // Extract coriolis_acceleration from Python list of lists
    std::vector<std::vector<double>> coriolis_acceleration;
    if (PyList_Check(py_coriolis_acceleration)) {
        for (Py_ssize_t i = 0; i < PyList_Size(py_coriolis_acceleration); ++i) {
            PyObject* inner_list = PyList_GetItem(py_coriolis_acceleration, i);
            std::vector<double> inner_vector;

            if (PyList_Check(inner_list)) {
                for (Py_ssize_t j = 0; j < PyList_Size(inner_list); ++j) {
                    inner_vector.push_back(PyFloat_AsDouble(PyList_GetItem(inner_list, j)));
                }
            }
            coriolis_acceleration.push_back(inner_vector);
        }
    }

    auto velocities = calculate_velocities(coriolis_acceleration, time, num_steps);

    // Build Python list of lists for output
    PyObject* py_result = PyList_New(num_steps);
    for (size_t i = 0; i < velocities.size(); ++i) {
        PyObject* py_inner = PyList_New(3);
        for (size_t j = 0; j < 3; ++j) {
            PyList_SetItem(py_inner, j, Py_BuildValue("d", velocities[i][j]));
        }
        PyList_SetItem(py_result, i, py_inner);
    }
    
    return py_result;
}

// Calculate drift distances based on velocities
std::vector<std::vector<double>> calculate_drift_distances(const std::vector<std::vector<double>>& coriolis_velocity, double time, int num_steps) {
    std::vector<std::vector<double>> coriolis_drift_distance(num_steps, std::vector<double>(3, 0.0));

    for (int i = 1; i < num_steps; ++i) {
        double dt = time / (num_steps - 1);
        for (int j = 0; j < 3; ++j) {
            coriolis_drift_distance[i][j] = coriolis_drift_distance[i - 1][j] + 
                0.5 * (coriolis_velocity[i][j] + coriolis_velocity[i - 1][j]) * dt;
        }
    }
    return coriolis_drift_distance;
}

// Wrapper for calculate_drift_distances
static PyObject* py_calculate_drift_distances(PyObject* self, PyObject* args) {
    PyObject* py_coriolis_velocity;
    double time;
    int num_steps;

    if (!PyArg_ParseTuple(args, "OdI", &py_coriolis_velocity, &time, &num_steps)) {
        return nullptr;
    }

    // Extract coriolis_velocity from Python list of lists
    std::vector<std::vector<double>> coriolis_velocity;
    if (PyList_Check(py_coriolis_velocity)) {
        for (Py_ssize_t i = 0; i < PyList_Size(py_coriolis_velocity); ++i) {
            PyObject* inner_list = PyList_GetItem(py_coriolis_velocity, i);
            std::vector<double> inner_vector;

            if (PyList_Check(inner_list)) {
                for (Py_ssize_t j = 0; j < PyList_Size(inner_list); ++j) {
                    inner_vector.push_back(PyFloat_AsDouble(PyList_GetItem(inner_list, j)));
                }
            }
            coriolis_velocity.push_back(inner_vector);
        }
    }

    auto drift_distances = calculate_drift_distances(coriolis_velocity, time, num_steps);

    // Build Python list of lists for output
    PyObject* py_result = PyList_New(num_steps);
    for (size_t i = 0; i < drift_distances.size(); ++i) {
        PyObject* py_inner = PyList_New(3);
        for (size_t j = 0; j < 3; ++j) {
            PyList_SetItem(py_inner, j, Py_BuildValue("d", drift_distances[i][j]));
        }
        PyList_SetItem(py_result, i, py_inner);
    }

    return py_result;
}

// Calculate total drift based on the row data
double calculate_total_drift(const std::vector<double>& row, double airtime, int num_steps = 100) {
    double average_velocity = row[0] / airtime; // Assuming row[0] is 'haversine_distance'
    std::vector<double> scaled_direction = { 
        row[1] * average_velocity, // x_direction
        row[2] * average_velocity, // y_direction
        row[3] * average_velocity  // z_direction
    };

    auto coriolis_accelerations = coriolis_acc(row[4], row[5], scaled_direction, airtime, num_steps); // row[4] and row[5] are latitudes
    auto coriolis_velocity = calculate_velocities(coriolis_accelerations, airtime, num_steps);
    auto coriolis_drift_distance = calculate_drift_distances(coriolis_velocity, airtime, num_steps);
    
    // Calculate magnitude of the final drift distance
    double total_drift = std::sqrt(std::pow(coriolis_drift_distance.back()[0], 2) + 
                                    std::pow(coriolis_drift_distance.back()[1], 2) + 
                                    std::pow(coriolis_drift_distance.back()[2], 2));
    return total_drift;
}

// Wrapper for calculate_total_drift
static PyObject* py_calculate_total_drift(PyObject* self, PyObject* args) {
    PyObject* py_row;
    double airtime;
    int num_steps = 100; // Default value

    if (!PyArg_ParseTuple(args, "Od|i", &py_row, &airtime, &num_steps)) {
        return nullptr;
    }

    // Extract row data from Python list
    std::vector<double> row;
    if (PyList_Check(py_row)) {
        for (Py_ssize_t i = 0; i < PyList_Size(py_row); ++i) {
            row.push_back(PyFloat_AsDouble(PyList_GetItem(py_row, i)));
        }
    }

    double total_drift = calculate_total_drift(row, airtime, num_steps);
    return Py_BuildValue("d", total_drift);
}

// Method definitions for the module
static PyMethodDef CoriolisMethods[] = {
    {"direction_vector", py_direction_vector, METH_VARARGS, "Calculate direction vector between two lat/lon points"},
    {"rotation_matrix", py_rotation_matrix, METH_VARARGS, "Calculate the rotation matrix based on latitude"},
    {"radius_at_latitude", py_radius_at_latitude, METH_VARARGS, "Calculate Earth's radius at a given latitude"},
    {"coriolis_acc", py_coriolis_acc, METH_VARARGS, "Calculate Coriolis accelerations"},
    {"calculate_velocities", py_calculate_velocities, METH_VARARGS, "Calculate velocities from Coriolis acceleration"},
    {"calculate_drift_distances", py_calculate_drift_distances, METH_VARARGS, "Calculate drift distances from velocities"},
    {"calculate_total_drift", py_calculate_total_drift, METH_VARARGS, "Calculate total drift from row data"},
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
