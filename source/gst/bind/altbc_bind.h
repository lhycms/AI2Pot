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

#ifndef AI2POT_GST_ALTBC_BIND
#define AI2POT_GST_ALTBC_BIND

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


/**
 * @brief 
 * 
 * @param self 
 * @param args 
 *  @param 
 * @return PyObject* 
 */
static PyObject* py_find_batch_long_short_bonds(PyObject* self, PyObject* args);


#endif