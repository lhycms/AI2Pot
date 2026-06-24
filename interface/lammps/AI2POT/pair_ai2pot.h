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

#ifdef PAIR_CLASS
PairStyle(ai2pot, PairAI2Pot)
#else

#ifndef LMP_PAIR_AI2POT_H
#define LMP_PAIR_AI2POT_H

#include "pair.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>

namespace LAMMPS_NS {


class PairAI2Pot : public Pair {
public:
    PairAI2Pot(class LAMMPS*);
    ~PairAI2Pot();

    int get_node_rank();
    std::vector<int> get_lmp2model_type_map(
        const std::vector<int>& model_type_map, 
        const std::vector<int>& lmp_type_map);
    void construct_mlff_input();

    void settings(int, char**) override;
    void coeff(int, char**) override;
    double init_one(int, int) override;
    void init_style() override;
    void compute(int eflag, int vflag) override;

protected:
    virtual void allocate();

private:
    torch::jit::script::Module model;
    torch::Device device = torch::kCPU;
    torch::Dtype torch_float_dtype = torch::kFloat32;

    double cutoff;
    int umax_num_neigh_atoms;
    std::vector<int> model_type_map;
    std::vector<int> lmp_type_map;
    std::vector<int> lmp2model_type_map;
    c10::TensorOptions int_options;
    c10::TensorOptions float_options;

    at::Tensor binum_tensor;
    at::Tensor bilist_tensor;
    at::Tensor bnumneigh_tensor;
    at::Tensor bfirstneigh_tensor;
    at::Tensor brcs_tensor;
    at::Tensor btypes_tensor;
    at::Tensor bnghost_tensor;
};  // class : PairAI2Pot

};  // namespace : LAMMPS_NS

#endif

#endif