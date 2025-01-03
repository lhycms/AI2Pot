/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <vector>

#include "../include/altbc.h"
#include "../include/order_parameter.h"


int initialize_npy(void) {
    if (PyArray_API)
        return 0;
    else
        import_array1(-1);
}

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

static PyObject* py_find_batch_long_short_bonds(PyObject* self, PyObject *args) {
    int npy_mark = initialize_npy();

    PyObject* py_binum;
    PyObject* py_bilist;
    PyObject* py_bnumneigh;
    PyObject* py_bfirstneigh;
    PyObject* py_brcs;
    PyObject* py_btypes;
    PyObject* py_bnghost;
    PyObject* py_umax_num_neigh_atoms;
    PyObject* py_angle_threshold;
    if (!PyArg_ParseTuple(args,
                          "OOOOOOOOO",
                          &py_binum,
                          &py_bilist,
                          &py_bnumneigh,
                          &py_bfirstneigh,
                          &py_brcs,
                          &py_btypes,
                          &py_bnghost,
                          &py_umax_num_neigh_atoms,
                          &py_angle_threshold))
        return NULL;
    if (!PyArray_Check(py_binum)
        || !PyArray_Check(py_bilist)
        || !PyArray_Check(py_bnumneigh)
        || !PyArray_Check(py_bfirstneigh)
        || !PyArray_Check(py_brcs)
        || !PyArray_Check(py_btypes)
        || !PyArray_Check(py_bnghost)
        || !PyLong_Check(py_umax_num_neigh_atoms)
        || !PyFloat_Check(py_angle_threshold)
        ) 
    {
        PyErr_SetString(PyExc_TypeError, "Check the parameters of py_find_batch_long_short_bonds.");
        return NULL;
    }

    npy_intp* py_binum_shape = PyArray_SHAPE((PyArrayObject*)py_binum);
    int batch_size = (int)py_binum_shape[0];
    int inum = (int)py_binum_shape[1];
    int npy_int_dtype = PyArray_TYPE((PyArrayObject*)py_binum);
    if (npy_int_dtype != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Please use NPY_INT32 for py_find_batch_long_short_bonds().");
        return NULL;
    }
    int npy_float_dtype = PyArray_TYPE((PyArrayObject*)py_brcs);
    int* py_binum_data = (int*)PyArray_DATA((PyArrayObject*)py_binum);
    int* py_bilist_data = (int*)PyArray_DATA((PyArrayObject*)py_bilist);
    int* py_bnumneigh_data = (int*)PyArray_DATA((PyArrayObject*)py_bnumneigh);
    int* py_bfirstneigh_data = (int*)PyArray_DATA((PyArrayObject*)py_bfirstneigh);
    // rcs
    int* py_btypes_data = (int*)PyArray_DATA((PyArrayObject*)py_btypes);
    int* bnghost = (int*)PyArray_DATA((PyArrayObject*)py_bnghost);
    int umax_num_neigh_atoms = (int)PyLong_AsLong(py_umax_num_neigh_atoms);
    double angle_threshold = (double)PyFloat_AsDouble(py_angle_threshold);

    void* long_bonds_ptr;
    void* short_bonds_ptr;
    int num_ls_bonds;
    if (npy_float_dtype == NPY_FLOAT32) {
        float* py_brcs_data = (float*)PyArray_DATA((PyArrayObject*)py_brcs);
        long_bonds_ptr = (void*)(new std::vector<float>());
        short_bonds_ptr = (void*)(new std::vector<float>());
        for (int ii=0; ii<batch_size; ii++)
            ai2pot::gst::Altbc<float>::find_long_short_bonds(*((std::vector<float>*)(long_bonds_ptr)),
                                                             *((std::vector<float>*)(short_bonds_ptr)),
                                                             py_binum_data[ii],
                                                             &py_bilist_data[ii*inum],
                                                             &py_bnumneigh_data[ii*inum],
                                                             &py_bfirstneigh_data[ii*inum*umax_num_neigh_atoms],
                                                             &py_brcs_data[ii*inum*umax_num_neigh_atoms*3],
                                                             &py_btypes_data[ii*(inum+bnghost[ii])],
                                                             bnghost[ii],
                                                             umax_num_neigh_atoms,
                                                             angle_threshold);
        num_ls_bonds = ((std::vector<float>*)(long_bonds_ptr))->size();
    } else {    // npy_float_dtype == NPY_FLOAT64
        double* py_brcs_data = (double*)PyArray_DATA((PyArrayObject*)py_brcs);
        long_bonds_ptr = (void*)(new std::vector<double>());
        short_bonds_ptr = (void*)(new std::vector<double>());
        for (int ii=0; ii<batch_size; ii++)
            ai2pot::gst::Altbc<double>::find_long_short_bonds(*((std::vector<double>*)(long_bonds_ptr)),
                                                            *((std::vector<double>*)(short_bonds_ptr)),
                                                            py_binum_data[ii],
                                                            &py_bilist_data[ii*inum],
                                                            &py_bnumneigh_data[ii*inum],
                                                            &py_bfirstneigh_data[ii*inum*umax_num_neigh_atoms],
                                                            &py_brcs_data[ii*inum*umax_num_neigh_atoms*3],
                                                            &py_btypes_data[ii*(inum*bnghost[ii])],
                                                            bnghost[ii],
                                                            umax_num_neigh_atoms,
                                                            angle_threshold);
        num_ls_bonds = ((std::vector<double>*)(long_bonds_ptr))->size();
    }
    
    // Step . Make results (Too much memory)
    npy_intp long_short_bonds_array_shape[2] = {2, num_ls_bonds};
    PyObject* long_short_bonds_array;
    if (npy_float_dtype == NPY_FLOAT32) {
        long_short_bonds_array = PyArray_Zeros(2, long_short_bonds_array_shape, PyArray_DescrFromType(NPY_FLOAT32), 0);
        float* long_short_bonds_array_data = (float*)PyArray_DATA((PyArrayObject*)long_short_bonds_array);
        for (int ii=0; ii<num_ls_bonds; ii++) {
            long_short_bonds_array_data[0*num_ls_bonds + ii] = (*((std::vector<float>*)long_bonds_ptr))[ii];
            long_short_bonds_array_data[1*num_ls_bonds + ii] = (*((std::vector<float>*)short_bonds_ptr))[ii];
        }
    } else {    // npy_float_dtype == NPY_FLOAT64
        long_short_bonds_array = PyArray_Zeros(2, long_short_bonds_array_shape, PyArray_DescrFromType(NPY_FLOAT64), 0);
        double* long_short_bonds_array_data = (double*)PyArray_DATA((PyArrayObject*)long_short_bonds_array);
        for (int ii=0; ii<num_ls_bonds; ii++) {
            long_short_bonds_array_data[0*num_ls_bonds + ii] = (*((std::vector<double>*)long_bonds_ptr))[ii];
            long_short_bonds_array_data[1*num_ls_bonds + ii] = (*((std::vector<double>*)short_bonds_ptr))[ii];
        }
    }

    // Step . Free
    if (npy_float_dtype == NPY_FLOAT32) {
        delete (std::vector<float>*)(long_bonds_ptr);
        delete (std::vector<float>*)(short_bonds_ptr);
    } else {
        delete (std::vector<double>*)(long_bonds_ptr);
        delete (std::vector<double>*)(short_bonds_ptr);
    }

    return long_short_bonds_array;
}

static PyObject* py_find_order_parameters(PyObject* self, PyObject* args) {
    int npy_mark = initialize_npy();

    PyObject* py_inum;
    PyObject* py_ilist;
    PyObject* py_numneigh;
    PyObject* py_firstneigh;
    PyObject* py_rcs;
    PyObject* py_types;
    PyObject* py_nghost;
    PyObject* py_umax_num_neigh_atoms;
    if (!PyArg_ParseTuple(args,
                          "OOOOOOOO",
                          &py_inum,
                          &py_ilist,
                          &py_numneigh,
                          &py_firstneigh,
                          &py_rcs,
                          &py_types,
                          &py_nghost,
                          &py_umax_num_neigh_atoms))
        return NULL;
    if (!PyLong_Check(py_inum)
        || !PyArray_Check(py_ilist)
        || !PyArray_Check(py_numneigh)
        || !PyArray_Check(py_firstneigh)
        || !PyArray_Check(py_rcs)
        || !PyArray_Check(py_types)
        || !PyLong_Check(py_nghost)
        || !PyLong_Check(py_umax_num_neigh_atoms)) {
        PyErr_SetString(PyExc_TypeError, "Check the parameters of py_find_order_parameters().");
        return NULL;
    }

    if ( PyArray_TYPE((PyArrayObject*)py_ilist)
         == PyArray_TYPE((PyArrayObject*)py_numneigh)
         == PyArray_TYPE((PyArrayObject*)py_firstneigh)
         == PyArray_TYPE((PyArrayObject*)py_types) )
    {
        PyErr_SetString(PyExc_TypeError, "Please unify the NPY_INT in py_find_order_parameters().");
        return NULL;
    }
    int npy_int_dtype = PyArray_TYPE((PyArrayObject*)py_ilist);
    int npy_float_dtype = PyArray_TYPE((PyArrayObject*)py_rcs);
    if (npy_int_dtype != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Please use NPY_INT32 in py_find_order_parameters()");
        return NULL;
    }

    int inum = (int)PyLong_AsLong(py_inum);
    int* py_ilist_data = (int*)PyArray_DATA((PyArrayObject*)py_ilist);
    int* py_numneigh_data = (int*)PyArray_DATA((PyArrayObject*)py_numneigh);
    int* py_firstneigh_data = (int*)PyArray_DATA((PyArrayObject*)py_firstneigh);
    int* py_types_data = (int*)PyArray_DATA((PyArrayObject*)py_types);
    int nghost = (int)PyLong_AsLong(py_nghost);
    int umax_num_neigh_atoms = (int)PyLong_AsLong(py_umax_num_neigh_atoms);

    // Step . Make results
    PyObject* py_order_parameters;
    npy_intp py_order_parameters_shape[1] = {inum};
    if (npy_float_dtype == NPY_FLOAT32) {        
        py_order_parameters = PyArray_Zeros(1, py_order_parameters_shape, PyArray_DescrFromType(NPY_FLOAT32), 0);
        float* py_order_parameters_data = (float*)PyArray_DATA((PyArrayObject*)py_order_parameters);
        float* py_rcs_data = (float*)PyArray_DATA((PyArrayObject*)py_rcs);
        ai2pot::gst::OrderParameter<float>::find_order_parameters(py_order_parameters_data,
                                                                  inum,
                                                                  py_ilist_data,
                                                                  py_numneigh_data,
                                                                  py_firstneigh_data,
                                                                  py_rcs_data,
                                                                  py_types_data,
                                                                  nghost,
                                                                  umax_num_neigh_atoms);
    } else {    // npy_float_dtype == NPY_FLOAT64
        py_order_parameters = PyArray_Zeros(1, py_order_parameters_shape, PyArray_DescrFromType(NPY_FLOAT64), 0);
        double* py_order_parameters_data = (double*)PyArray_DATA((PyArrayObject*)py_order_parameters);
        double* py_rcs_data = (double*)PyArray_DATA((PyArrayObject*)py_rcs);
        ai2pot::gst::OrderParameter<double>::find_order_parameters(py_order_parameters_data,
                                                                   inum,
                                                                   py_ilist_data,
                                                                   py_numneigh_data,
                                                                   py_firstneigh_data,
                                                                   py_rcs_data,
                                                                   py_types_data,
                                                                   nghost,
                                                                   umax_num_neigh_atoms);
    }
    return py_order_parameters;
}

static PyMethodDef methods[] = {
    {"find_long_short_bonds", (PyCFunction)&py_find_long_short_bonds, METH_VARARGS, "gst.find_long_short_bonds() from C++."},
    {"find_batch_long_short_bonds", (PyCFunction)&py_find_batch_long_short_bonds, METH_VARARGS, "gst.find_batch_long_short_bonds() from C++."},
    {"find_order_parameters", (PyCFunction)&py_find_order_parameters, METH_VARARGS, "gst.find_order_parameters() from C++."},
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
