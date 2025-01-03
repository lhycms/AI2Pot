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


namespace ai2pot {

int initialize_npy(void);

/**
 * @brief 
 * 
 * @param self
 * @param args
 *  @param py_lattice 
 *          np.array (3, 3), NPY_FLOAT32
 *  @param py_types 
 *          np.array (num_atoms,), NPY_INT32 or NPY_INT64
 *  @param py_coords 
 *          np.array (num_atoms, 3), NPY_FLOAT32 or NPY_FLOAT64
 *  @param py_rcut 
 *          double
 *  @param py_max_num_neigh_atoms 
 *          long
 *  @param py_is_cart_coords 
 *          bool
 *  @param py_pbc_xyz 
 *          List[bool] [True, True, True]
 *  @param py_sort 
 *          bool
 *  @return PyObject* 
 */
static PyObject* py_find_info4mlff(PyObject *self, PyObject *args);

};  // namespace : ai2pot