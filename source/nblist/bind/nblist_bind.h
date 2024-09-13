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