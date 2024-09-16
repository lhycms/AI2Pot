#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <vector>

#include "../include/altbc.h"


int initialize_npy(void) {
    if (PyArray_API)
        return 0;
    else
        import_array1(-1);
}

/**
 * @brief 
 * 
 * @param self 
 * @param args 
 *  @param py_inum: PyLong
 *  @param py_list: PyArray (NPY_INT32)
 *  @param py_numneigh: PyArray (NPY_INT32)
 *  @param py_firstneigh: PyArray (NPY_INT32)
 *  @param py_rcs: PyArray (NPY_FLOAT32 or NPY_FLOAT64)
 *  @param py_types: PyArray (NPY_INT32)
 *  @param py_nghost: PyLong
 *  @param Py_umax_num_neigh_atoms: PyLong
 * @return PyObject* 
 */
static PyObject* py_find_long_short_bonds(PyObject* self, PyObject* args) {
    int npy_mark = initialize_npy();

    PyObject* py_inum;
    PyObject* py_ilist;
    PyObject* py_numneigh;
    PyObject* py_firstneigh;
    PyObject* py_rcs;
    PyObject* py_types;
    PyObject* py_nghost;
    PyObject* py_umax_num_neigh_atoms;
    PyObject* py_angle_threshold;
    if (!PyArg_ParseTuple(args, 
                          "OOOOOOOOO",
                          &py_inum,
                          &py_ilist,
                          &py_numneigh,
                          &py_firstneigh,
                          &py_rcs,
                          &py_types,
                          &py_nghost,
                          &py_umax_num_neigh_atoms,
                          &py_angle_threshold))
        return NULL;
    if (!PyLong_Check(py_inum) ||
        !PyArray_Check(py_ilist) ||
        !PyArray_Check(py_numneigh) ||
        !PyArray_Check(py_firstneigh) ||
        !PyArray_Check(py_rcs) ||
        !PyArray_Check(py_types) ||
        !PyLong_Check(py_nghost) || 
        !PyLong_Check(py_umax_num_neigh_atoms) ||
        !PyFloat_Check(py_angle_threshold))
    {
        PyErr_SetString(PyExc_TypeError, "Please check the parameters in py_find_long_short_bonds().\n");
        return NULL;
    }
    int inum = (int)PyLong_AsLong(py_inum);
    int npy_int_dtype = PyArray_TYPE((PyArrayObject*)py_ilist);
    if (npy_int_dtype != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Please NPY_INT32 as parameters in py_find_long_short_bonds().\n");
        return NULL;
    }
    int npy_float_dtype = PyArray_TYPE((PyArrayObject*)py_numneigh);
    int* py_ilist_data = (int*)PyArray_DATA((PyArrayObject*)py_ilist);
    int* py_numneigh_data = (int*)PyArray_DATA((PyArrayObject*)py_numneigh);
    int* py_firstneigh_data = (int*)PyArray_DATA((PyArrayObject*)py_firstneigh);
    int* py_types_data = (int*)PyArray_DATA((PyArrayObject*)py_types);
    int nghost = (int)PyLong_AsLong(py_nghost);
    int umax_num_neigh_atoms = (int)PyLong_AsLong(py_umax_num_neigh_atoms);
    double angle_threshold = PyFloat_AsDouble(py_angle_threshold);

    void* long_bonds_ptr;
    void* short_bonds_ptr;
    int num_ls_bonds;
    if (npy_float_dtype == NPY_FLOAT32) {
        float* py_rcs_data = (float*)PyArray_DATA((PyArrayObject*)py_rcs);
        long_bonds_ptr = (void*)(new std::vector<float>());
        short_bonds_ptr = (void*)(new std::vector<float>());
        ai2pot::gst::Altbc<float>::find_long_short_bonds(*((std::vector<float>*)(long_bonds_ptr)),
                                                         *((std::vector<float>*)(short_bonds_ptr)),
                                                         inum,
                                                         py_ilist_data,
                                                         py_numneigh_data,
                                                         py_firstneigh_data,
                                                         py_rcs_data,
                                                         py_types_data,
                                                         nghost,
                                                         umax_num_neigh_atoms,
                                                         angle_threshold);
        num_ls_bonds = ((std::vector<float>*)(long_bonds_ptr))->size();
    } else {    // npy_float_dtype == NPY_FLOAT64
        double* py_rcs_data = (double*)PyArray_DATA((PyArrayObject*)py_rcs);
        long_bonds_ptr = (void*)(new std::vector<double>());
        short_bonds_ptr = (void*)(new std::vector<double>());
        ai2pot::gst::Altbc<double>::find_long_short_bonds(*((std::vector<double>*)(long_bonds_ptr)),
                                                          *((std::vector<double>*)(short_bonds_ptr)),
                                                          inum,
                                                          py_ilist_data,
                                                          py_numneigh_data,
                                                          py_firstneigh_data,
                                                          py_rcs_data,
                                                          py_types_data,
                                                          nghost,
                                                          umax_num_neigh_atoms,
                                                          angle_threshold);
        num_ls_bonds = ((std::vector<double>*)(long_bonds_ptr))->size();
    }

    // Step . Make results (Too much memory!!!)
    npy_intp long_short_bonds_array_shape[2] = {2, num_ls_bonds};
    PyObject* long_short_bonds_array;
    if (npy_float_dtype == NPY_FLOAT32) {
        long_short_bonds_array = PyArray_Zeros(2, long_short_bonds_array_shape, PyArray_DescrFromType(NPY_FLOAT32), 0);
        float* long_short_bonds_array_data = (float*)PyArray_DATA((PyArrayObject*)long_short_bonds_array);
        for (int ii=0; ii<num_ls_bonds; ii++) {
            long_short_bonds_array_data[0*num_ls_bonds + ii] = ( *((std::vector<float>*)long_bonds_ptr) )[ii];
            long_short_bonds_array_data[1*num_ls_bonds + ii] = ( *((std::vector<float>*)short_bonds_ptr) )[ii];
        }
    } else {
        long_short_bonds_array = PyArray_Zeros(2, long_short_bonds_array_shape, PyArray_DescrFromType(NPY_FLOAT64), 0);
        double* long_short_bonds_array_data = (double*)PyArray_DATA((PyArrayObject*)long_short_bonds_array);
        for (int ii=0; ii<num_ls_bonds; ii++) {
            long_short_bonds_array_data[0*num_ls_bonds + ii] = ( *((std::vector<double>*)long_bonds_ptr) )[ii];
            long_short_bonds_array_data[1*num_ls_bonds + ii] = ( *((std::vector<double>*)short_bonds_ptr) )[ii];
        }
    }

    // Step . Free
    if (npy_float_dtype == NPY_FLOAT32) {
        delete (std::vector<float>*)(short_bonds_ptr);
        delete (std::vector<float>*)(long_bonds_ptr);
    } else {
        delete (std::vector<double>*)(short_bonds_ptr);
        delete (std::vector<double>*)(long_bonds_ptr);
    }

    return long_short_bonds_array;
}

static PyMethodDef methods[] = {
    {"find_long_short_bonds", (PyCFunction)&py_find_long_short_bonds, METH_VARARGS, "gst.find_long_short_bonds() from C++."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef gst = {
    PyModuleDef_HEAD_INIT,
    "gst",
    "gst module from C++",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_gst(void) {
    return PyModule_Create(&gst);
}