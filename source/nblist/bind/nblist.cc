#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#include "./nblist.h"
#include "../include/neighborList.h"


namespace ai2pot {

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
    PyObject *py_types;
    PyObject *py_coords;
    PyObject *py_rcut;
    PyObject *py_max_num_neigh_atoms;
    PyObject *py_is_cart_coords;
    PyObject *py_pbc_xyz;
    PyObject *py_sort;
    if (PyArg_ParseTuple(args, 
                         "OOOOOOO",
                         &py_lattice,
                         &py_types,
                         &py_coords,
                         &py_rcut,
                         &py_max_num_neigh_atoms,
                         &py_is_cart_coords,
                         &py_pbc_xyz,
                         &py_sort))
        return NULL;
    PyObject *return_tuple = PyTuple_New(7);

    assert( PyArray_TYPE((PyArrayObject*)py_lattice) == PyArray_TYPE((PyArrayObject*)py_coords) );
    int npy_int_type = PyArray_TYPE((PyArrayObject*)py_types);
    int npy_float_type = PyArray_TYPE((PyArrayObject*)py_coords);

    npy_intp* py_coords_shape = PyArray_SHAPE((PyArrayObject*)py_coords);
    int num_atoms = py_coords_shape[0];
    if ( (npy_int_type==NPY_INT32) && (npy_float_type==NPY_FLOAT32) ) {
        // 1.1. Lattice
        float *lattice_data = (float*)PyArray_DATA((PyArrayObject*)py_lattice);
        float basis_vector[3][3];
        for (int ii=0; ii<3; ii++)
            for (int jj=0; jj<3; jj++)
                basis_vector[ii][jj] = lattice_data[ii*3 + jj];
        // 1.2. types
        int *types_data = (int*)PyArray_DATA((PyArrayObject*)py_types);
        int *atomic_number = types_data;
        // 1.3. coords
        float **coords = (float**)malloc(num_atoms * sizeof(float*));
        for (int ii=0; ii<3; ii++)
            coords[ii] = (float*)malloc(3 * sizeof(float));
        for (int ii=0; ii<num_atoms; ii++)
            for (int jj=0; jj<3; jj++)
                coords[ii][jj] = ;

        // 1.Free
        int 
        float
    } else if ( (npy_int_type==NPY_INT32) && (npy_float_type==NPY_FLOAT64) ) {
        int 
        double
    } else if ( (npy_int_type==NPY_INT64) && (npy_float_type==NPY_FLOAT32) ) {
        long 
        int
    } else {
        long
        double
    }



    return return_tuple;
}

}; // namespace : ai2pot