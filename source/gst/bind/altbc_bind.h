#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <vector>

#include "../include/altbc.h"


int initialize_npy(void);

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
static PyObject* py_find_long_short_bonds(PyObject* self, PyObject* args);