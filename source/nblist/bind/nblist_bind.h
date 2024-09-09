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
 *  @param py_types 
 *  @param py_coords 
 *  @param py_rcut 
 *  @param py_max_num_neigh_atoms 
 *  @param py_is_cart_coords 
 *  @param py_pbc_xyz 
 *  @param py_sort 
 *  @return PyObject* 
 */
static PyObject* py_find_info4mlff(PyObject *self, PyObject *args);

};  // namespace : ai2pot