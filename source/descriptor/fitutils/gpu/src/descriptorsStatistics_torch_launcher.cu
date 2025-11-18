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

#include "../include/descriptorsStatistics_torch_launcher.cuh"


// 1. find_all_type_descriptors_statistics_torch_launcher()
template void ai2pot::fitutils::find_all_type_descriptors_statistics_torch_launcher<float>(
    int *d_natoms_in_batch_ptr,
    float *d_descriptors_mean,
    float *d_descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *d_binum,
    float *d_bdescriptors);

template void ai2pot::fitutils::find_all_type_descriptors_statistics_torch_launcher<double>(
    int *d_natoms_in_batch_ptr,
    double *d_descriptors_mean,
    double *d_descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *d_binum,
    double *d_bdescriptors);


// 2. find_each_type_descriptors_statistics_torch_launcher()
