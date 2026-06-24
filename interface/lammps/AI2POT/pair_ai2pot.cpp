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

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <mpi.h>

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <string.h>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "domain.h"

#include "pair_ai2pot.h"

using namespace LAMMPS_NS;


static std::string normalize_element_symbol(const std::string& symbol)
{
    if (symbol.empty()) return symbol;

    std::string result = symbol;
    result[0] = static_cast<char>(std::toupper(result[0]));

    for (size_t i = 1; i < result.size(); i++) {
        result[i] = static_cast<char>(std::tolower(result[i]));
    }

    return result;
}


static int element_to_atomic_number(const std::string& symbol)
{
    static const std::unordered_map<std::string, int> table = {
        {"H", 1}, {"He", 2},
        {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
        {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18},
        {"K", 19}, {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25}, {"Fe", 26},
        {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30}, {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34},
        {"Br", 35}, {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, {"Nb", 41}, {"Mo", 42},
        {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50},
        {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58},
        {"Pr", 59}, {"Nd", 60}, {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66},
        {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74},
        {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82},
        {"Bi", 83}, {"Po", 84}, {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90},
        {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {"Am", 95}, {"Cm", 96}, {"Bk", 97}, {"Cf", 98},
        {"Es", 99}, {"Fm", 100}, {"Md", 101}, {"No", 102}, {"Lr", 103}, {"Rf", 104}, {"Db", 105},
        {"Sg", 106}, {"Bh", 107}, {"Hs", 108}, {"Mt", 109}, {"Ds", 110}, {"Rg", 111}, {"Cn", 112},
        {"Nh", 113}, {"Fl", 114}, {"Mc", 115}, {"Lv", 116}, {"Ts", 117}, {"Og", 118}
    };

    std::string key = normalize_element_symbol(symbol);

    auto iter = table.find(key);
    if (iter == table.end()) {
        return -1;
    }

    return iter->second;
}


PairAI2Pot::PairAI2Pot(LAMMPS* lmp) : Pair(lmp)
{
    writedata = 0;
}


PairAI2Pot::~PairAI2Pot()
{
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }
}


void PairAI2Pot::allocate()
{
    allocated = 1;
    
    memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
    for (int ii=1; ii<atom->ntypes+1; ii++)
        for (int jj=1; jj<atom->ntypes+1; jj++)
            setflag[ii][jj] = 0;
    memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}


int PairAI2Pot::get_node_rank()
{
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status comm_status;
    int comm_size;
    int comm_rank;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_len;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Get_processor_name(processor_name, &processor_name_len);

    MPI_Comm node_comm;
    int node_rank;
    MPI_Comm_split_type(
        comm,
        MPI_COMM_TYPE_SHARED,
        comm_rank,
        MPI_INFO_NULL,
        &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);

    MPI_Comm_free(&node_comm);

    return node_rank;
}


std::vector<int> PairAI2Pot::get_lmp2model_type_map(
    const std::vector<int>& model_type_map,
    const std::vector<int>& lmp_type_map)
{
    std::vector<int> lmp2model_type_map(atom->ntypes+1, -1);

    for (int lmp_index=1; lmp_index<atom->ntypes+1; lmp_index++) {
        int atomic_number = lmp_type_map[lmp_index];
        int lmp2model_val = -1;

        for (int model_index=0; model_index<static_cast<int>(model_type_map.size()); model_index++) {
            if (model_type_map[model_index] == atomic_number) {
                lmp2model_val = model_index;
                break;
            }
        }

        if (lmp2model_val == -1) {
            error->all(FLERR, "LAMMPS atom type is not included in AI2Pot model type_map");
        }

        lmp2model_type_map[lmp_index] = lmp2model_val;
    }

    return lmp2model_type_map;
}


void PairAI2Pot::construct_mlff_input()
{
    int nlocal = atom->nlocal;
    int nghost = atom->nghost;
    int nall = atom->nlocal + atom->nghost;
    // int* list->ilist
    // int* list->numneigh
    // int** list->firstneigh
    // int* atom->type
    // double** x = atom->x

    // 1. Malloc tensor
    c10::TensorOptions local_int_options = c10::TensorOptions().dtype(torch::kInt32).device(c10::kCPU);
    c10::TensorOptions local_float_options = c10::TensorOptions().dtype(torch::kFloat32).device(c10::kCPU);
    binum_tensor = at::zeros({1}, local_int_options);
    bilist_tensor = at::zeros({1, nlocal}, local_int_options);
    bnumneigh_tensor = at::zeros({1, nlocal}, local_int_options);
    bfirstneigh_tensor = at::zeros({1, nlocal, umax_num_neigh_atoms}, local_int_options);
    brcs_tensor = at::zeros({1, nlocal, umax_num_neigh_atoms, 3}, local_float_options);
    btypes_tensor = at::zeros({1, nall}, local_int_options);
    bnghost_tensor = at::zeros({1}, local_int_options);

    // 2. Assign tensor
    int *inum_ptr = binum_tensor.data_ptr<int>();
    int *ilist = bilist_tensor.data_ptr<int>();
    int *numneigh = bnumneigh_tensor.data_ptr<int>();
    int *firstneigh = bfirstneigh_tensor.data_ptr<int>();
    float *rcs = brcs_tensor.data_ptr<float>();
    int *types = btypes_tensor.data_ptr<int>();
    int *nghost_ptr = bnghost_tensor.data_ptr<int>();

    inum_ptr[0] = nlocal;
    nghost_ptr[0] = nghost;
    for (int ii=0; ii<list->inum; ii++) {
        int center_idx = list->ilist[ii];
        ilist[ii] = list->ilist[ii];
        numneigh[ii] = list->numneigh[center_idx];

        for (int jj=0; jj<list->numneigh[center_idx]; jj++) {
            int neigh_idx = list->firstneigh[center_idx][jj] & NEIGHMASK;
            firstneigh[ii*umax_num_neigh_atoms + jj] = neigh_idx;

            for (int aa=0; aa<3; aa++)
                rcs[ii*umax_num_neigh_atoms*3 + jj*3 + aa] = atom->x[neigh_idx][aa] - atom->x[center_idx][aa];
        }
    }

    for (int ii=0; ii<nall; ii++)
        types[ii] = lmp2model_type_map[atom->type[ii]];

    // 3. Communication
    binum_tensor = binum_tensor.to(device);
    bilist_tensor = bilist_tensor.to(device);
    bnumneigh_tensor = bnumneigh_tensor.to(device);
    bfirstneigh_tensor = bfirstneigh_tensor.to(device);
    brcs_tensor = brcs_tensor.to(device);
    btypes_tensor = btypes_tensor.to(device);
    bnghost_tensor = bnghost_tensor.to(device);
}


void PairAI2Pot::settings(int argc, char **argv) {
    if (argc != 1)
        error->all(FLERR, "Illegal pair_style command");

    std::string pt_path = argv[0];
    try {
        // 1. Model
        torch::jit::getExecutorMode() = false;
        if (torch::cuda::is_available())
            device = torch::Device(torch::kCUDA, get_node_rank());
        
        model = torch::jit::load(pt_path);
        model.to(device);
        model.to(torch_float_dtype);
        model.eval();

        std::string potential_type = "AI2Pot-NEP";
        if (model.hasattr("mtp_level"))
            potential_type = "AI2Pot-MTP";

        // 2. Log
        if (comm->me == 0) {
            std::string message;
            message += "AI2Pot: TorchScript model loaded successfully\n";
            message += "  model file      : " + pt_path + "\n";
            message += "  potential type  : " + potential_type + "\n";
            message += "  device          : " + device.str() + "\n";
            message += "  torch dtype     : float32\n";
            utils::logmesg(lmp, message);
        }
    } catch (const c10::Error &e) {
        utils::logmesg(
            lmp,
            std::string("AI2Pot: failed to load TorchScript model from %s\n")
            + std::string("PyTorch error message:\n%s\n")
            + pt_path
            + std::string(e.what()));
        error->all(FLERR, "Failed to load TorchScript model");
    }
}


void PairAI2Pot::coeff(int argc, char **argv) {
    if (!allocated)
        allocate();
    
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, argv[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, argv[1], 1, atom->ntypes, jlo, jhi, error);

    // 1. cutoff
    if (model.hasattr("mtp_level"))
        cutoff = model.attr("rmax").toDouble();
    else
        cutoff = model.attr("rmax_radial").toDouble();

    // 2. setflag
    int count = 0;
    for (int i=ilo; i<=ihi; i++) {
        for (int j=std::max(jlo, i); j<=jhi; j++) {
            setflag[i][j] = 1;
            cutsq[i][j] = cutoff * cutoff;
            count++;
        }
    }
    if (count == 0)
        error->all(FLERR, "Incorrect args for pair coefficients");

    // 3. lmp2model_type_map
    torch::Tensor model_type_map_tensor = model.attr("type_map_tensor").toTensor().to(torch::kCPU).to(torch::kInt32);
    model_type_map.clear();
    for (int i=0; i<model_type_map_tensor.numel(); i++)
        model_type_map.push_back(model_type_map_tensor[i].item<int>());
    
    lmp_type_map.clear();
    lmp_type_map.resize(atom->ntypes+1, -1);
    for (int lmp_index=1; lmp_index<atom->ntypes+1; lmp_index++) {
        int atomic_numer = element_to_atomic_number(std::string(argv[lmp_index + 1]));
        lmp_type_map[lmp_index] = atomic_numer;
    }

    lmp2model_type_map = get_lmp2model_type_map(model_type_map, lmp_type_map);


    // 4. Other attributions of model
    umax_num_neigh_atoms = model.attr("umax_num_neigh_atoms").toInt();

    // 5. TensorOptions
    int_options = c10::TensorOptions().dtype(torch::kInt32).device(device);
    float_options = c10::TensorOptions().dtype(torch_float_dtype).device(device);

    // 6. Log
    if (comm->me == 0) {
        std::string message;
        message += "AI2Pot: pair_coeff initialized\n";
        message += "  cutoff                 : " + std::to_string(cutoff) + "\n";
        message += "  umax_num_neigh_atoms   : " + std::to_string(umax_num_neigh_atoms) + "\n";

        message += "  model_type_map  : ";
        for (int z : model_type_map) {
            message += std::to_string(z) + " ";
        }
        message += "\n";

        message += "  lmp_type_map    : ";
        for (int i=1; i<atom->ntypes+1; i++) {
            message += std::to_string(lmp_type_map[i]) + " ";
        }
        message += "\n";

        message += "  lmp2model_map   : ";
        for (int i=1; i<atom->ntypes+1; i++) {
            message += std::to_string(lmp2model_type_map[i]) + " ";
        }
        message += "\n";

        utils::logmesg(lmp, message);
    }
}


double PairAI2Pot::init_one(int i, int j) {
    return cutoff;
}


void PairAI2Pot::init_style() {
    neighbor->add_request(this, NeighConst::REQ_FULL);
}


void PairAI2Pot::compute(int eflag, int vflag) {
    if (eflag || vflag)
        ev_setup(eflag, vflag);

    // 1. Construct mlff_input
    double **f = atom->f;
    int nlocal = atom->nlocal;
    int nall = atom->nlocal + atom->nghost;
    construct_mlff_input();

if (comm->me == 0) {
    std::cout << "binum dtype       = " << binum_tensor.dtype() << std::endl;
    std::cout << "bilist dtype      = " << bilist_tensor.dtype() << std::endl;
    std::cout << "bnumneigh dtype   = " << bnumneigh_tensor.dtype() << std::endl;
    std::cout << "bfirstneigh dtype = " << bfirstneigh_tensor.dtype() << std::endl;
    std::cout << "brcs dtype        = " << brcs_tensor.dtype() << std::endl;
    std::cout << "btypes dtype      = " << btypes_tensor.dtype() << std::endl;
    std::cout << "bnghost dtype     = " << bnghost_tensor.dtype() << std::endl;

    std::cout << "binum sizes       = " << binum_tensor.sizes() << std::endl;
    std::cout << "bilist sizes      = " << bilist_tensor.sizes() << std::endl;
    std::cout << "bnumneigh sizes   = " << bnumneigh_tensor.sizes() << std::endl;
    std::cout << "bfirstneigh sizes = " << bfirstneigh_tensor.sizes() << std::endl;
    std::cout << "brcs sizes        = " << brcs_tensor.sizes() << std::endl;
    std::cout << "btypes sizes      = " << btypes_tensor.sizes() << std::endl;
    std::cout << "bnghost sizes     = " << bnghost_tensor.sizes() << std::endl;
}

    // 2. Forward
    if (model.attr("fit_virial").toBool() == true) {
        auto output = model.get_method("predict_efv")(
            {
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                bnghost_tensor
            }
        ).toTuple();
    } else {
        auto output = model.get_method("predict_ef")(
            {
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                bnghost_tensor
            }
        ).toTuple();
    }
    
    // 3. Assign lmp variable

}
