#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdexcept>

#include "../include/neighborList.h"


/**
 * @brief 
 * 
 * @param lattice_py 
 *          np.array (3, 3)
 * @param atomic_numbers_py 
 *          np.array (num_atoms,)  -- astype(np.int32)
 * @param frac_coords_py 
 *          np.array (num_atoms, 3)
 * @param rcut_py 
 *          double
 * @param pbc_xyz_py
 *          List[bool] [True, True, True]
 * @param umax_num_neigh_atoms
 *          int (long)
 * @param sort_py
 *          bool 
 * @param is_cart_coord_py
 *          bool 
 * @return PyObject* 
 */
PyObject* find_info4mlff(
        PyObject* lattice_py,
        PyObject* atomic_numbers_py,
        PyObject* frac_coords_py,
        PyObject* rcut_py,
        PyObject* pbc_xyz_py,
        PyObject* umax_num_neigh_atoms_py,
        PyObject* sort_py,
        PyObject* is_cart_coord_py)
{   
    import_array();

    // Step 0. Ensure NPY_INT32 and NPY_FLOAT64
    //PyArray_Descr* descr_lattice_py = PyArray_DESCR((PyArrayObject*)lattice_py);
    //PyArray_Descr* descr_atomic_numbers_py = PyArray_DESCR((PyArrayObject*)atomic_numbers_py);
    //PyArray_Descr* descr_frac_coords_py = PyArray_DESCR((PyArrayObject*)frac_coords_py);
    //PyArray_Descr* descr_umax_num_neigh_atoms_py = PyArray_DESCR((PyArrayObject*)umax_num_neigh_atoms_py);
    //assert(descr_lattice_py->type_num == NPY_FLOAT64);
    //assert(descr_atomic_numbers_py->type_num == NPY_INT32);
    //assert(descr_frac_coords_py->type_num == NPY_FLOAT64);
    //assert(descr_umax_num_neigh_atoms_py->type_num == NPY_INT32);
    

    // Step 1. Init matersdk::Structure<double>
    // Step 1.1. 
    PyArrayObject* lattice_py_array = (PyArrayObject*)lattice_py;
    PyArrayObject* atomic_numbers_py_array = (PyArrayObject*)atomic_numbers_py;
    PyArrayObject* frac_coords_py_array = (PyArrayObject*)frac_coords_py;
    int num_atoms = (int)PyArray_SHAPE(atomic_numbers_py_array)[0];  // npy_intp 占8个字节
    int umax_num_neigh_atoms = (int)PyLong_AsLong(umax_num_neigh_atoms_py);

    // Step 1.2. Declaration and assignment
    double lattice[3][3];
    int atomic_numbers[num_atoms];
    double frac_coords[num_atoms][3];

    double* lattice_ptr = (double*)PyArray_DATA(lattice_py_array);
    for (int ii=0; ii<3; ii++)
        for (int jj=0; jj<3; jj++)
            lattice[ii][jj] = lattice_ptr[ii*3 + jj];
    int* atomic_numbers_ptr = (int*)PyArray_DATA(atomic_numbers_py_array);
    double* frac_coords_ptr = (double*)PyArray_DATA(frac_coords_py_array);
    for (int ii=0; ii<num_atoms; ii++) {
        atomic_numbers[ii] = atomic_numbers_ptr[ii];
        frac_coords[ii][0] = frac_coords_ptr[ii*3 + 0];
        frac_coords[ii][1] = frac_coords_ptr[ii*3 + 1];
        frac_coords[ii][2] = frac_coords_ptr[ii*3 + 2];
    }
    
    // Step 1.3. You must init it with fractional/cartesian coordinates
    bool is_cart_coord = (PyObject_IsTrue(is_cart_coord_py));
    matersdk::Structure<double> structure(num_atoms, lattice, atomic_numbers, frac_coords, is_cart_coord);

    // Step 2. Init matersdk::NeighborList<double>
    // Step 2.1. 
    double rcut = PyFloat_AsDouble(rcut_py);
    bool pbc_xyz[3];
    bool sort;
    for (int ii=0; ii<3; ii++) {
        if ( PyObject_IsTrue(PyList_GetItem(pbc_xyz_py, ii)) )
            pbc_xyz[ii] = true;
        else 
            pbc_xyz[ii] = false;
    }
    if (PyObject_IsTrue(sort_py))
        sort = true;
    else
        sort = false;

    // Step 2.2. Build 
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, sort);

    // Step 3. Call matersdk::NeighborList<double>::find_info4mlff()
    int inum = (int)num_atoms;
    int* ilist = (int*)malloc(sizeof(int) * inum);
    int* numneigh = (int*)malloc(sizeof(int) * inum);
    int* firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
    double* relative_coords = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
    int* types = (int*)malloc(sizeof(int) * inum);
    int nghost;
    neighbor_list.find_info4mlff(
        inum,
        ilist,
        numneigh,
        firstneigh,
        relative_coords,
        types,
        nghost,
        umax_num_neigh_atoms);

    // Step 4. Return 
    PyObject* nblist_info = PyTuple_New(7);
    // Step 4.1. inum
    PyTuple_SetItem(nblist_info, 0, PyLong_FromLong(inum));
    // Step 4.2. ilist
    npy_intp* ilist_dims = (npy_intp*)malloc(sizeof(npy_intp) * 1);
    ilist_dims[0] = inum;
    PyObject* ilist_py = PyArray_SimpleNewFromData(1, ilist_dims, NPY_INT, ilist);
    PyArray_ENABLEFLAGS((PyArrayObject*)ilist_py, NPY_OWNDATA);
    // Step 4.3. numneigh
    npy_intp* numneigh_dims = (npy_intp*)malloc(sizeof(npy_intp) * 1);
    numneigh_dims[0] = inum;
    PyObject* numneigh_py = PyArray_SimpleNewFromData(1, numneigh_dims, NPY_INT, numneigh);
    PyArray_ENABLEFLAGS((PyArrayObject*)numneigh_py, NPY_OWNDATA);
    // Step 4.4. firstneigh
    npy_intp* firstneigh_dims = (npy_intp*)malloc(sizeof(npy_intp) * 2);
    firstneigh_dims[0] = inum;
    firstneigh_dims[1] = umax_num_neigh_atoms;
    PyObject* firstneigh_py = PyArray_SimpleNewFromData(2, firstneigh_dims, NPY_INT, firstneigh);
    PyArray_ENABLEFLAGS((PyArrayObject*)numneigh_py, NPY_OWNDATA);
    // Step 4.5. relative_coords
    npy_intp* relative_coords_dims = (npy_intp*)malloc(sizeof(npy_intp) * 3);
    relative_coords_dims[0] = inum;
    relative_coords_dims[1] = umax_num_neigh_atoms;
    relative_coords_dims[2] = 3;
    PyObject* relative_coords_py = PyArray_SimpleNewFromData(3, relative_coords_dims, NPY_DOUBLE, relative_coords);
    PyArray_ENABLEFLAGS((PyArrayObject*)relative_coords_py, NPY_OWNDATA);
    // Step 4.6. types
    npy_intp* types_dims = (npy_intp*)malloc(sizeof(npy_intp));
    types_dims[0] = inum;
    PyObject* types_py = PyArray_SimpleNewFromData(1, types_dims, NPY_INT32, types);
    PyArray_ENABLEFLAGS((PyArrayObject*)types_py, NPY_OWNDATA);
    // Step 4.7. nghost
    //PyTuple_SetItem(nblist_info, 6, PyLong_FromLong((long)nghost));
    
    // Step 4.7. 
    PyTuple_SetItem(nblist_info, 1, ilist_py);
    PyTuple_SetItem(nblist_info, 2, numneigh_py);
    PyTuple_SetItem(nblist_info, 3, firstneigh_py);
    PyTuple_SetItem(nblist_info, 4, relative_coords_py);
    PyTuple_SetItem(nblist_info, 5, types_py);
    PyTuple_SetItem(nblist_info, 6, PyLong_FromLong((long)nghost));
    
    
    // Step . Free memory
    //free(ilist);
    //free(numneigh);
    //free(firstneigh);
    //free(relative_coords);
    //free(types);
    free(ilist_dims);
    free(numneigh_dims);
    free(firstneigh_dims);
    free(relative_coords_dims);
    free(types_dims);

    return nblist_info;
}
