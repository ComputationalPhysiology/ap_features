#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "cost_terms.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>


static PyObject *_get_num_cost_terms(PyObject *self, PyObject *args)
{

    if (!PyArg_ParseTuple(args, ""))
        return NULL;


    return PyLong_FromLong(get_num_cost_terms());
}

static PyObject *_apd(PyObject *self, PyObject *args)
{
    PyArrayObject *y_arr, *t_arr;
    int factor;

    if (!PyArg_ParseTuple(args, "OOi", &y_arr, &t_arr, &factor))
        return NULL;

    if (PyErr_Occurred())
        return NULL;

    if (!PyArray_Check(y_arr)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a numeric numpy array");
        return NULL;
    }

    if (!PyArray_Check(t_arr)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a numeric numpy array");
        return NULL;
    }

    int64_t y_size = PyArray_SIZE(y_arr);
    double *y_data;
    npy_intp y_dims[] = {[0] = y_size};
    PyArray_AsCArray((PyObject **) &y_arr, &y_data, y_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));

    int64_t t_size = PyArray_SIZE(y_arr);
    double *t_data;
    npy_intp t_dims[] = {[0] = t_size};
    PyArray_AsCArray((PyObject **) &t_arr, &t_data, t_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));

    if (PyErr_Occurred())
        return NULL;

    if (y_size != t_size) {
        PyErr_SetString(PyExc_ValueError, "Length of time array and y array must be the same");
        return NULL;
    }

    double *y_copy = malloc(sizeof(double) * t_size);
    memcpy(y_copy, y_data, sizeof(double) * t_size);
    double value = apd(y_data, t_data, factor, y_size, y_copy);

    free(y_copy);

    return PyFloat_FromDouble(value);
}

static PyObject *_apd_up_xy(PyObject *self, PyObject *args)
{
    PyArrayObject *y_arr, *t_arr;
    int factor_x, factor_y;

    if (!PyArg_ParseTuple(args, "OOii", &y_arr, &t_arr, &factor_x, &factor_y))
        return NULL;

    if (PyErr_Occurred())
        return NULL;

    if (!PyArray_Check(y_arr)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a numeric numpy array");
        return NULL;
    }

    if (!PyArray_Check(t_arr)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a numeric numpy array");
        return NULL;
    }

    int64_t y_size = PyArray_SIZE(y_arr);
    double *y_data;
    npy_intp y_dims[] = {[0] = y_size};
    PyArray_AsCArray((PyObject **) &y_arr, &y_data, y_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));

    int64_t t_size = PyArray_SIZE(y_arr);
    double *t_data;
    npy_intp t_dims[] = {[0] = t_size};
    PyArray_AsCArray((PyObject **) &t_arr, &t_data, t_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));

    if (PyErr_Occurred())
        return NULL;

    if (y_size != t_size) {
        PyErr_SetString(PyExc_ValueError, "Length of time array and y array must be the same");
        return NULL;
    }

    double *y_copy = malloc(sizeof(double) * t_size);
    memcpy(y_copy, y_data, sizeof(double) * t_size);
    double value = apd_up_xy(y_data, t_data, factor_x, factor_y, y_size, y_copy);

    free(y_copy);

    return PyFloat_FromDouble(value);
}


static PyMethodDef methods[] = {
        {"apd", _apd, METH_VARARGS, "Action potential duration"},
        {"get_num_cost_terms", _get_num_cost_terms, METH_VARARGS, "Get the number of cost terms"},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef _libap_features = {PyModuleDef_HEAD_INIT, "_libap_features",
                                             "ap-features C library", -1, methods};


PyMODINIT_FUNC PyInit__libap_features()
{
    PyObject *m;

    m = PyModule_Create(&_libap_features);
    if (m == NULL)
        return NULL;
    import_array();
    return m;
}
