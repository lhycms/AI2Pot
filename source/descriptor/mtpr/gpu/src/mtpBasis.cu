#include "../include/mtpBasis.cuh"


// Explicitly instantiate template for float
template
void find_mtp_basis_val_der_cuda_launcher<float>();

// Explicitly instantiate template for double
template
void find_mtp_basis_val_der_cuda_launcher<double>();
