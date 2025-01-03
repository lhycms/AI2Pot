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
#include <stdlib.h>

#include "./nblist_bind.h"
#include "../include/structure.h"
#include "../include/neighborList.h"



int initialize_npy(void) {
    if (PyArray_API)
        return 0;
    else
        import_array1(-1);
}

static PyObject *py_find_info4mlff(PyObject *self, PyObject *args)
{
    int npy_mark = initialize_npy();
    PyObject *py_lattice;
    PyObject *py_types_pre;
    PyObject *py_types; //
    PyObject *py_coords;
    PyObject *py_rcut;
    PyObject *py_umax_num_neigh_atoms;
    PyObject *py_is_cart_coords;
    PyObject *py_pbc_xyz;
    PyObject *py_sort;
    if (!PyArg_ParseTuple(args, 
                         "OOOOOOOO",
                         &py_lattice,
                         &py_types_pre,
                         &py_coords,
                         &py_rcut,
                         &py_umax_num_neigh_atoms,
                         &py_is_cart_coords,
                         &py_pbc_xyz,
                         &py_sort))
        return NULL;
    if (!PyArray_Check(py_lattice) ||
        !PyArray_Check(py_types_pre) ||
        !PyArray_Check(py_coords) ||
        !PyFloat_Check(py_rcut) ||
        !PyLong_Check(py_umax_num_neigh_atoms) ||
        !PyBool_Check(py_is_cart_coords) ||
        !PyList_Check(py_pbc_xyz) ||
        !PyBool_Check(py_sort))
    {
        PyErr_SetString(PyExc_TypeError, "Please check the parameter in interface of py_find_info4mlff().");
        return NULL;
    }
    PyObject *return_tuple = PyTuple_New(7);

    assert( PyArray_TYPE((PyArrayObject*)py_lattice) == PyArray_TYPE((PyArrayObject*)py_coords) );
    int npy_int_type = PyArray_TYPE((PyArrayObject*)py_types_pre);
    int npy_float_type = PyArray_TYPE((PyArrayObject*)py_coords);

    // Step 1. Get Structure
    void *structure_ptr;
    npy_intp* py_coords_shape = PyArray_SHAPE((PyArrayObject*)py_coords);
    int num_atoms = py_coords_shape[0];
    if ( npy_int_type==NPY_INT64 ) {    // Convert NPY_INT64 to NPY_INT32
        py_types = PyArray_Cast((PyArrayObject*)py_types_pre, NPY_INT32);
    } else {
        py_types = py_types_pre;
        Py_INCREF(py_types);
    }
    if ( npy_float_type == NPY_FLOAT32 ) {
        // 1.C1. Lattice
        float *lattice_data = (float*)PyArray_DATA((PyArrayObject*)py_lattice);
        // 1.C2. types
        int *types_data = (int*)PyArray_DATA((PyArrayObject*)py_types);
        // 1.C3. coords
        float *coords = (float*)PyArray_DATA((PyArrayObject*)py_coords);
        // 1.C4. is_cart_coords
        bool is_cart_coords;
        if (PyObject_IsTrue(py_is_cart_coords))
            is_cart_coords = true;
        else
            is_cart_coords = false;
        structure_ptr = (void*)(new ai2pot::Structure<float>(num_atoms, lattice_data, types_data, coords, is_cart_coords));
    } else {    // npy_float_type==NPY_FLOAT64
        // 1.C1. lattice
        double *lattice_data = (double*)PyArray_DATA((PyArrayObject*)py_lattice);
        // 1.C2. types
        int *types_data = (int*)PyArray_DATA((PyArrayObject*)py_types);
        // 1.C3. coords
        double *coords = (double*)PyArray_DATA((PyArrayObject*)py_coords);
        // 1.C4. is_cart_coords
        bool is_cart_coords;
        if (PyObject_IsTrue(py_is_cart_coords))
            is_cart_coords = true;
        else
            is_cart_coords = false;
        structure_ptr = (void*)(new ai2pot::Structure<double>(num_atoms, lattice_data, types_data, coords, is_cart_coords));
    }

    // Step 2. Get Neighbor List
    void *nblist_ptr;
    double rcut = PyFloat_AsDouble(py_rcut);
    bool pbc_xyz[3];
    bool sort;
    for (int ii=0; ii<3; ii++) {
        if (PyObject_IsTrue( PyList_GetItem(py_pbc_xyz, (Py_ssize_t)ii) ))
            pbc_xyz[ii] = true;
        else
            pbc_xyz[ii] = false;
    }
    if (PyObject_IsTrue(py_sort))
        sort = true;
    else
        sort = false;
    if ( npy_float_type == NPY_FLOAT32 )
        nblist_ptr = (void*)(new ai2pot::NeighborList<float>(*((ai2pot::Structure<float>*)structure_ptr), (float)rcut, pbc_xyz, sort));
    else
        nblist_ptr = (void*)(new ai2pot::NeighborList<double>(*((ai2pot::Structure<double>*)structure_ptr), rcut, pbc_xyz, sort));

    // Step 3. Make return_tuple
    int inum;
    int nghost;
    npy_intp py_ilist_dims[1] = { (npy_intp)num_atoms };
    PyObject *py_ilist = PyArray_Zeros(1, py_ilist_dims, PyArray_DescrFromType(NPY_INT32), 0);
    npy_intp py_numneigh_dims[1] = { (npy_intp)num_atoms };
    PyObject *py_numneigh = PyArray_Zeros(1, py_numneigh_dims, PyArray_DescrFromType(NPY_INT32), 0);
    npy_intp py_firstneigh_dims[2] = { (npy_intp)num_atoms, (npy_intp)PyLong_AsLong(py_umax_num_neigh_atoms) };
    PyObject *py_firstneigh = PyArray_Zeros(2, py_firstneigh_dims, PyArray_DescrFromType(NPY_INT32), 0);
    npy_intp py_rcs_dims[3] = {(npy_intp)num_atoms, (npy_intp)PyLong_AsLong(py_umax_num_neigh_atoms), 3};
    PyObject *py_rcs;
    //PyObject *py_types = PyArray_Zeros(1, {(npy_intp)(num_atoms + nghost)}, PyArray_DescrFromType(NPY_INT64), 0);
    if (npy_float_type == NPY_FLOAT32) {
        // Convert NPY_FLOAT32 TO NPY_FLOAT64.
        py_rcs = PyArray_Zeros(3, py_rcs_dims, PyArray_DescrFromType(NPY_FLOAT32), 0);
        ((ai2pot::NeighborList<float>*)nblist_ptr)->find_info4mlff(inum,
                                                           (int*)PyArray_DATA((PyArrayObject*)py_ilist),
                                                           (int*)PyArray_DATA((PyArrayObject*)py_numneigh),
                                                           (int*)PyArray_DATA((PyArrayObject*)py_firstneigh),
                                                           (float*)PyArray_DATA((PyArrayObject*)py_rcs),
                                                           (int*)PyArray_DATA((PyArrayObject*)py_types),
                                                           nghost,
                                                           (int)PyLong_AsLong(py_umax_num_neigh_atoms));
    } else {
        py_rcs = PyArray_Zeros(3, py_rcs_dims, PyArray_DescrFromType(NPY_FLOAT64), 0);
        ((ai2pot::NeighborList<double>*)nblist_ptr)->find_info4mlff(inum,
                                                           (int*)PyArray_DATA((PyArrayObject*)py_ilist),
                                                           (int*)PyArray_DATA((PyArrayObject*)py_numneigh),
                                                           (int*)PyArray_DATA((PyArrayObject*)py_firstneigh),
                                                           (double*)PyArray_DATA((PyArrayObject*)py_rcs),
                                                           (int*)PyArray_DATA((PyArrayObject*)py_types),
                                                           nghost,
                                                           (int)PyLong_AsLong(py_umax_num_neigh_atoms));
    }
    PyTuple_SetItem(return_tuple, 0, PyLong_FromLong((long)inum));
    PyTuple_SetItem(return_tuple, 1, py_ilist);
    PyTuple_SetItem(return_tuple, 2, py_numneigh);
    PyTuple_SetItem(return_tuple, 3, py_firstneigh);
    PyTuple_SetItem(return_tuple, 4, py_rcs);
    PyTuple_SetItem(return_tuple, 5, py_types);
    PyTuple_SetItem(return_tuple, 6, PyLong_FromLong((long)nghost));
    
    // Step Free. 
    if (npy_float_type == NPY_FLOAT32) {
        delete (ai2pot::Structure<float>*)structure_ptr;
        delete (ai2pot::NeighborList<float>*)nblist_ptr;
    } else {
        delete (ai2pot::Structure<double>*)structure_ptr;
        delete (ai2pot::NeighborList<double>*)nblist_ptr;
    }
    return return_tuple;
}

static PyMethodDef methods[] = {
    {"find_info4mlff", (PyCFunction)&py_find_info4mlff, METH_VARARGS, "nblist.find_info4mlff() from C++"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef nblist = {
    PyModuleDef_HEAD_INIT,
    "nblist",
    "nblist from C++",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_nblist(void) {
    return PyModule_Create(&nblist);
}
