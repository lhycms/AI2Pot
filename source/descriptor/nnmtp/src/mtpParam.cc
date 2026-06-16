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

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <cstdlib>
#include <set>
#include <cstring>
#include "../include/mtpParam.h"


static std::string g_ai2pot_path;

namespace ai2pot {
namespace nnmtp {

void set_ai2pot_path(const std::string& path) {
    g_ai2pot_path = path;
}


std::string get_mtp_template_dir() {
    const char* ai2pot_path = std::getenv("AI2POT_PATH");
    if (ai2pot_path)
        return std::string(ai2pot_path) + "/source/descriptor/mtpr/MTP_templates/";
    else
        return g_ai2pot_path + "/_data/MTP_templates";
}


MtpParam::MtpParam()
{
    this->_alpha_moments_count = 0;
    this->_alpha_index_basic_count = 0;
    this->_alpha_index_basic = nullptr;
    this->_alpha_index_times_count = 0;
    this->_alpha_index_times = nullptr;
    this->_alpha_scalar_moments = 0;
    this->_alpha_moment_mapping = nullptr;
    this->_max_num_mus4mom = 0;
    this->_num_mus4moms = nullptr;
    this->_mus4moms_ptr = nullptr;
}

MtpParam::MtpParam(const std::string& filename)
{
    this->_load(filename);
}


MtpParam::MtpParam(const int mtp_level) {
    if ((mtp_level < 2) || (mtp_level > 28) || (mtp_level % 2 != 0))
        MtpError("mtp level must be odd number (2 ~ 28)");
    std::string dir = get_mtp_template_dir();
    std::string filename;
    if (mtp_level <= 4) {
        filename = dir + (std::string)"depreciated-" + "0" + std::to_string(mtp_level) + ".almtp";
    
    } else {
        if (mtp_level < 10)
            filename = dir + (std::string)"0" + std::to_string(mtp_level) + ".almtp";
        else
            filename = dir + std::to_string(mtp_level) + ".almtp";
    }
    this->_load(filename);
}

void MtpParam::_load(const std::string& filename)
{
    std::ifstream ifs;
    ifs.open(filename);
    if ( !ifs.is_open() ) {
        MtpError((std::string)"Cannot open file " + filename);
    }
    char tmpline[1000];
    std::string tmpstr;

    for (int ii=0; ii<10; ii++) 
        ifs.ignore(5000, '\n');
    
    // alpha_moments_count
    ifs >> tmpstr;
    if (tmpstr != "alpha_moments_count")
        MtpError("Cannot read alpha_moment_counts.");
    ifs.ignore(2);
    ifs >> this->_alpha_moments_count;
    if (ifs.fail()) 
        MtpError("Cannot read alpha_moment_counts.");
    
    // alpha_index_basic_count
    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic_count")
        MtpError("Cannot read alpha_index_basic_count.");
    ifs.ignore(2);
    ifs >> this->_alpha_index_basic_count;
    if (ifs.fail())
        MtpError("Cannot read alpha_index_basic_count.");
    
    // alpha_index_basic
    this->_alpha_index_basic = (int (*)[4])malloc(4*sizeof(int)*this->_alpha_index_basic_count);
    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic")
        MtpError("Cannot read alpha_index_basic.");
    ifs.ignore(4);
    for (int ii=0; ii<this->_alpha_index_basic_count; ii++) {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> this->_alpha_index_basic[ii][0] >> tmpch
            >> this->_alpha_index_basic[ii][1] >> tmpch
            >> this->_alpha_index_basic[ii][2] >> tmpch
            >> this->_alpha_index_basic[ii][3];
        if (ifs.fail())
            MtpError("Cannot read alpha_index_basic.");
    }
    ifs.ignore(1000, '\n');

    // alpha_index_times_count
    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times_count") 
        MtpError("Cannot read alpha_index_times_count.");
    ifs.ignore(2);
    ifs >> this->_alpha_index_times_count;
    if (ifs.fail())
        MtpError("Cannot read alpha_index_times_count.");

    // alpha_index_times
    if (this->_alpha_index_times_count != 0)
        this->_alpha_index_times = (int (*)[4])malloc(4*sizeof(int) * this->_alpha_index_times_count);
    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times")
        MtpError("Cannot read alpha_index_times.");
    ifs.ignore(4);
    if (this->_alpha_index_times_count != 0)
        for (int ii=0; ii<this->_alpha_index_times_count; ii++) {
            char tmpch;
            ifs.ignore(1000, '{');
            ifs >> this->_alpha_index_times[ii][0] >> tmpch
                >> this->_alpha_index_times[ii][1] >> tmpch
                >> this->_alpha_index_times[ii][2] >> tmpch
                >> this->_alpha_index_times[ii][3];
            if (ifs.fail())
                MtpError("Cannot read alpha_index_times.");
        }
    ifs.ignore(1000, '\n');

    // alpha_scalar_moments
    ifs >> tmpstr;
    if (tmpstr != "alpha_scalar_moments")
        MtpError("Cannot read alpha_scalar_moments.");
    ifs.ignore(2);
    ifs >> this->_alpha_scalar_moments;
    if (ifs.fail())
        MtpError("Cannot read alpha_scalar_moments.");
    
    // alpha_moment_mapping
    this->_alpha_moment_mapping = (int*)malloc(sizeof(int) * this->_alpha_scalar_moments);
    ifs >> tmpstr;
    if (tmpstr != "alpha_moment_mapping")
        MtpError("Cannot read alpha_moment_mapping.")
    ifs.ignore(4);
    for (int ii=0; ii<this->_alpha_scalar_moments; ii++) {
        char tmpch;
        ifs >> this->_alpha_moment_mapping[ii] >> tmpch;
        if (ifs.fail())
            MtpError("Cannot read alpha_moment_mapping.");
    }

    ifs.close();

    std::vector<std::set<int>> mus4moms_lst = this->_get_mus4all_mom_dp(this->_alpha_moments_count);
    this->_max_num_mus4mom = 0;
    for (int ii=0; ii<this->_alpha_moments_count; ii++)
        if (mus4moms_lst[ii].size() > this->_max_num_mus4mom)
            this->_max_num_mus4mom = mus4moms_lst[ii].size();

    this->_mus4moms_ptr = (int*)malloc(sizeof(int) * this->_alpha_moments_count * this->_max_num_mus4mom);
    this->_num_mus4moms = (int*)malloc(sizeof(int) * this->_alpha_moments_count);
    memset(this->_mus4moms_ptr, -1, sizeof(int) * this->_alpha_moments_count * this->_max_num_mus4mom);
    for (int ii=0; ii<this->_alpha_moments_count; ii++) {
        // Error:
        //memcpy(
        //    &(this->_mus4moms_ptr[this->_max_num_mus4mom * ii]),
        //    &(mus4moms_lst[ii]),
        //    sizeof(int) * mus4moms_lst[ii].size());
        this->_num_mus4moms[ii] = mus4moms_lst[ii].size();
        int count = 0;
        for (auto &v : mus4moms_lst[ii])
            this->_mus4moms_ptr[ii*this->_max_num_mus4mom+(count++)] = v;
    }
}

MtpParam::MtpParam(const MtpParam& rhs)
{
    this->_alpha_moments_count = rhs._alpha_moments_count;
    this->_alpha_index_basic_count = rhs._alpha_index_basic_count;
    if (this->_alpha_index_basic_count != 0) {
        this->_alpha_index_basic = (int (*)[4])malloc(4*sizeof(int)*this->_alpha_index_basic_count);
        for (int ii=0; ii<this->_alpha_index_basic_count; ii++)
        {
            this->_alpha_index_basic[ii][0] = rhs._alpha_index_basic[ii][0];
            this->_alpha_index_basic[ii][1] = rhs._alpha_index_basic[ii][1];
            this->_alpha_index_basic[ii][2] = rhs._alpha_index_basic[ii][2];
            this->_alpha_index_basic[ii][3] = rhs._alpha_index_basic[ii][3];
        }
    }
    this->_alpha_index_times_count = rhs._alpha_index_times_count;
    if (this->_alpha_index_times != 0) {
        this->_alpha_index_times = (int (*)[4])malloc(4*sizeof(int)*this->_alpha_index_times_count);
        for (int ii=0; ii<this->_alpha_index_times_count; ii++)
        {
            this->_alpha_index_times[ii][0] = rhs._alpha_index_times[ii][0];
            this->_alpha_index_times[ii][1] = rhs._alpha_index_times[ii][1];
            this->_alpha_index_times[ii][2] = rhs._alpha_index_times[ii][2];
            this->_alpha_index_times[ii][3] = rhs._alpha_index_times[ii][3];
        }
    }
    this->_alpha_scalar_moments = rhs._alpha_scalar_moments;
    if (this->_alpha_moments_count != 0) {
        this->_alpha_moment_mapping = (int*)malloc(sizeof(int) * this->_alpha_scalar_moments);
        for (int ii=0; ii<this->_alpha_scalar_moments; ii++)
            this->_alpha_moment_mapping[ii] = rhs._alpha_moment_mapping[ii];
    }
    this->_max_num_mus4mom = rhs._max_num_mus4mom;
    if (this->_max_num_mus4mom != 0) {
        this->_num_mus4moms = (int*)malloc(sizeof(int) * this->_alpha_moments_count);
        this->_mus4moms_ptr = (int*)malloc(sizeof(int) * this->_alpha_moments_count * this->_max_num_mus4mom);
        for (int ii=0; ii<this->_alpha_moments_count; ii++) {
            this->_num_mus4moms[ii] = rhs._num_mus4moms[ii];
            for (int jj=0; jj<this->_max_num_mus4mom; jj++)
                this->_mus4moms_ptr[ii*this->_max_num_mus4mom + jj] = rhs._mus4moms_ptr[ii*this->_max_num_mus4mom + jj];
        }
    }
}

MtpParam::MtpParam(MtpParam&& rhs)
{
    if (this != &rhs) {
        this->_alpha_moments_count = rhs._alpha_moments_count;
        this->_alpha_index_basic_count = rhs._alpha_index_basic_count;
        this->_alpha_index_basic = rhs._alpha_index_basic;
        this->_alpha_index_times_count = rhs._alpha_index_times_count;
        this->_alpha_index_times = rhs._alpha_index_times;
        this->_alpha_scalar_moments = rhs._alpha_scalar_moments;
        this->_alpha_moment_mapping = rhs._alpha_moment_mapping;
        this->_max_num_mus4mom = rhs._max_num_mus4mom;
        this->_num_mus4moms = rhs._num_mus4moms;
        this->_mus4moms_ptr = rhs._mus4moms_ptr;


        rhs._alpha_moments_count = 0;
        rhs._alpha_index_basic_count = 0;
        rhs._alpha_index_basic = nullptr;
        rhs._alpha_index_times_count = 0;
        rhs._alpha_index_times = nullptr;
        rhs._alpha_scalar_moments = 0;
        rhs._alpha_moment_mapping = nullptr;
        rhs._max_num_mus4mom = 0;
        rhs._num_mus4moms = nullptr;
        rhs._mus4moms_ptr = nullptr;
    }
}

MtpParam& MtpParam::operator=(const MtpParam &rhs)
{
    this->_alpha_moments_count = 0;
    if (this->_alpha_index_basic_count != 0) {
        free(this->_alpha_index_basic);
        this->_alpha_index_basic_count = 0;
    }
    if (this->_alpha_index_times_count != 0) {
        free(this->_alpha_index_times);
        this->_alpha_index_times_count = 0;
    }
    if (this->_alpha_scalar_moments != 0) {
        free(this->_alpha_moment_mapping);
        this->_alpha_scalar_moments = 0;
    }
    if (this->_max_num_mus4mom != 0) {
        free(this->_num_mus4moms);
        free(this->_mus4moms_ptr);
        this->_max_num_mus4mom = 0;
    }

    this->_alpha_moments_count = rhs._alpha_moments_count;
    this->_alpha_index_basic_count = rhs._alpha_index_basic_count;
    if (this->_alpha_index_basic_count != 0) {
        this->_alpha_index_basic = (int (*)[4])malloc(4*sizeof(int)*this->_alpha_index_basic_count);
        for (int ii=0; ii<this->_alpha_index_basic_count; ii++) {
            this->_alpha_index_basic[ii][0] = rhs._alpha_index_basic[ii][0];
            this->_alpha_index_basic[ii][1] = rhs._alpha_index_basic[ii][1];
            this->_alpha_index_basic[ii][2] = rhs._alpha_index_basic[ii][2];
            this->_alpha_index_basic[ii][3] = rhs._alpha_index_basic[ii][3];
        }
    }
    this->_alpha_index_times_count = rhs._alpha_index_times_count;
    if (this->_alpha_index_times_count != 0) {
        this->_alpha_index_times = (int (*)[4])malloc(4*sizeof(int)*this->_alpha_index_times_count);
        for (int ii=0; ii<this->_alpha_index_times_count; ii++) {
            this->_alpha_index_times[ii][0] = rhs._alpha_index_times[ii][0];
            this->_alpha_index_times[ii][1] = rhs._alpha_index_times[ii][1];
            this->_alpha_index_times[ii][2] = rhs._alpha_index_times[ii][2];
            this->_alpha_index_times[ii][3] = rhs._alpha_index_times[ii][3];
        }
    }
    this->_alpha_scalar_moments = rhs._alpha_scalar_moments;
    if (this->_alpha_scalar_moments != 0) {
        this->_alpha_moment_mapping = (int*)malloc(sizeof(int) * this->_alpha_scalar_moments);
        for (int ii=0; ii<this->_alpha_scalar_moments; ii++)
            this->_alpha_moment_mapping[ii] = rhs._alpha_moment_mapping[ii];
    }
    this->_max_num_mus4mom = rhs._max_num_mus4mom;
    if ((this->_alpha_scalar_moments != 0) && (this->_max_num_mus4mom != 0)) {
        this->_num_mus4moms = (int*)malloc(sizeof(int) * this->_alpha_moments_count);
        this->_mus4moms_ptr = (int*)malloc(sizeof(int) * this->_alpha_moments_count * this->_max_num_mus4mom);
        memset(this->_mus4moms_ptr, -1, sizeof(int) * this->_alpha_scalar_moments * this->_max_num_mus4mom);
        for (int ii=0; ii<this->_alpha_scalar_moments; ii++) {
            for (int jj=0; jj<this->_max_num_mus4mom; jj++) {
                this->_num_mus4moms[ii] = rhs._num_mus4moms[ii];
                this->_mus4moms_ptr[ii*this->_max_num_mus4mom+jj] = rhs._mus4moms_ptr[ii*this->_max_num_mus4mom+jj];
            }
        }
    }

    return *this;
}

MtpParam& MtpParam::operator=(MtpParam&& rhs)
{
    if (this != &rhs) {
        this->_alpha_moments_count = 0;
        if (this->_alpha_index_basic_count != 0) {
            free(this->_alpha_index_basic);
            this->_alpha_index_basic_count = 0;
        }
        if (this->_alpha_index_times_count != 0) {
            free(this->_alpha_index_times);
            this->_alpha_index_times_count = 0;
        }
        if (this->_alpha_scalar_moments != 0) {
            free(this->_alpha_moment_mapping);
            this->_alpha_scalar_moments = 0;
        }
        if (this->_max_num_mus4mom != 0) {
            free(this->_num_mus4moms);
            free(this->_mus4moms_ptr);
            this->_max_num_mus4mom = 0;
        }

        this->_alpha_moments_count = rhs._alpha_moments_count;
        this->_alpha_index_basic_count = rhs._alpha_index_basic_count;
        this->_alpha_index_basic = rhs._alpha_index_basic;
        this->_alpha_index_times_count = rhs._alpha_index_times_count;
        this->_alpha_index_times = rhs._alpha_index_times;
        this->_alpha_scalar_moments = rhs._alpha_scalar_moments;
        this->_alpha_moment_mapping = rhs._alpha_moment_mapping;
        this->_max_num_mus4mom = rhs._max_num_mus4mom;
        this->_num_mus4moms = rhs._num_mus4moms;
        this->_mus4moms_ptr = rhs._mus4moms_ptr;


        rhs._alpha_moments_count = 0;
        rhs._alpha_index_basic_count = 0;
        rhs._alpha_index_basic = nullptr;
        rhs._alpha_index_times_count = 0;
        rhs._alpha_index_times = nullptr;
        rhs._alpha_scalar_moments = 0;
        rhs._alpha_moment_mapping = nullptr;
        rhs._max_num_mus4mom = 0;
        rhs._num_mus4moms = nullptr;
        rhs._mus4moms_ptr = nullptr;
    }
    return *this;
}

MtpParam::~MtpParam()
{
    if (this->_alpha_index_basic_count != 0)
        free(this->_alpha_index_basic);
    if (this->_alpha_index_times_count != 0)
        free(this->_alpha_index_times);
    if (this->_alpha_scalar_moments != 0)
        free(this->_alpha_moment_mapping);
    if (this->_max_num_mus4mom != 0) {
        free(this->_num_mus4moms);
        free(this->_mus4moms_ptr);
    }
    this->_alpha_moments_count = 0;
    this->_alpha_index_basic_count = 0;
    this->_alpha_index_times_count = 0;
    this->_alpha_scalar_moments = 0;
    this->_max_num_mus4mom = 0;
}

std::set<int> MtpParam::_get_mus4all_mom(int mom_idx)
{
    std::set<int> mus_lst;
    mus_lst.clear();

    if (mom_idx < this->_alpha_index_basic_count) {
        mus_lst.insert(this->_alpha_index_basic[mom_idx][0]);
    } else {
        for (int ii=0; ii<this->_alpha_index_times_count; ii++)
        {
            if (this->_alpha_index_times[ii][3] == mom_idx) {
                int sub_mom1_idx = this->_alpha_index_times[ii][0];
                int sub_mom2_idx = this->_alpha_index_times[ii][1];
//printf("*** %d, %d\n", sub_mom1_idx, sub_mom2_idx);
                std::set<int> sub_mus1_lst = this->_get_mus4all_mom(sub_mom1_idx);
                std::set<int> sub_mus2_lst = this->_get_mus4all_mom(sub_mom2_idx);

                for (auto& tmp_mu : sub_mus1_lst)
                    mus_lst.insert(tmp_mu);
                for (auto& tmp_mu : sub_mus2_lst)
                    mus_lst.insert(tmp_mu);
            }
        }
    }
    return mus_lst;
}

std::vector<std::set<int>> MtpParam::_get_mus4all_mom_dp(int num_moms)
{
    if (num_moms > this->_alpha_moments_count)
        MtpError("num_moms is larger then alpha_moments_count.");
    std::vector<std::set<int>> mus4moms_lst;
    mus4moms_lst.clear();
    std::set<int> mus_lst;

    for (int ii=0; ii<num_moms; ii++)
    {
//printf("mom_idx = %d:\n", ii);
        mus_lst.clear();
        if (ii < this->_alpha_index_basic_count) {
            mus_lst.insert(this->_alpha_index_basic[ii][0]);
        } else {
            for (int jj=0; jj<this->_alpha_index_times_count; jj++)
            {
                if (ii == this->_alpha_index_times[jj][3]) {
                    int sub_mom1_idx = this->_alpha_index_times[jj][0]; //
                    int sub_mom2_idx = this->_alpha_index_times[jj][1];
//printf("\t*** %d, %d\n", sub_mom1_idx, sub_mom2_idx);
                    std::set<int> sub_mus1_lst = mus4moms_lst[sub_mom1_idx];
                    std::set<int> sub_mus2_lst = mus4moms_lst[sub_mom2_idx];

                    for (auto& v : sub_mus1_lst)
                        mus_lst.insert(v);
                    for (auto& v : sub_mus2_lst)
                        mus_lst.insert(v);
                }
            }
        }
        mus4moms_lst.push_back(mus_lst);
    }
    return mus4moms_lst;
}

void MtpParam::show() const
{
    printf("1. alpha_moments_count = %10d\n", this->_alpha_moments_count);
    printf("2. alpha_index_basic_count = %10d\n", this->_alpha_index_basic_count);
    printf("3. alpha_index_basic:\n");
    for (int ii=0; ii<this->_alpha_index_basic_count; ii++)
        printf("\t%5d: [%5d, %5d, %5d, %5d]\n", ii, this->_alpha_index_basic[ii][0], this->_alpha_index_basic[ii][1], this->_alpha_index_basic[ii][2], this->_alpha_index_basic[ii][3]);
    printf("4. alpha_index_times_count = %10d\n", this->_alpha_index_times_count);
    printf("5. alpha_index_times:\n");
    for (int ii=0; ii<this->_alpha_index_times_count; ii++)
        printf("\t[%5d, %5d, %5d, %5d]\n", this->_alpha_index_times[ii][0], this->_alpha_index_times[ii][1], this->_alpha_index_times[ii][2], this->_alpha_index_times[ii][3]);
    printf("6. alpha_scalar_moments = %10d\n", this->_alpha_scalar_moments);
    printf("7. alpha_moment_mapping:\n\t");
    for (int ii=0; ii<this->_alpha_scalar_moments; ii++)
        printf("%5d, ", this->_alpha_moment_mapping[ii]);
    printf("\n");
    printf("8. mu4fmoms_ptr:\n");
    for (int ii=0; ii<this->_alpha_moments_count; ii++) {
        printf("\t mom_idx#%5d has %5d distinct mus:\t", ii, this->_num_mus4moms[ii]);
        printf("[");
        for (int jj=0; jj<this->_max_num_mus4mom; jj++)
            printf("%4d, ", this->_mus4moms_ptr[ii*this->_max_num_mus4mom+jj]);
        printf("]\n");
    }
    printf("\n");
}

int MtpParam::alpha_moments_count() const
{
    return this->_alpha_moments_count;
}

int MtpParam::alpha_index_basic_count() const
{
    return this->_alpha_index_basic_count;
}

const int (*MtpParam::alpha_index_basic() const)[4]
{
    return this->_alpha_index_basic;
}

int MtpParam::alpha_index_times_count() const
{
    return this->_alpha_index_times_count;
}

const int (*MtpParam::alpha_index_times() const)[4]
{
    return this->_alpha_index_times;
}

int MtpParam::alpha_scalar_moments() const 
{
    return this->_alpha_scalar_moments;
}

const int *MtpParam::alpha_moment_mapping() const
{
    return this->_alpha_moment_mapping;
}

int MtpParam::max_num_mus4mom() const
{
    return this->_max_num_mus4mom;
}

const int *MtpParam::num_mus4moms() const
{
    return this->_num_mus4moms;
}

const int *MtpParam::mus4moms_ptr() const
{
    return this->_mus4moms_ptr;
}


int MtpParam::nmus() const {
    int max_mu = 0;
    for (int ii=0; ii<this->_alpha_index_basic_count; ii++) {
        if (this->_alpha_index_basic[ii][0] > max_mu)
            max_mu = this->_alpha_index_basic[ii][0];
    }
    return max_mu + 1;
}




AlphaIndexTimes::AlphaIndexTimes()
{
    this->_alpha_index_times_count = 0;
    this->_alpha_index_times = nullptr;
}

AlphaIndexTimes::AlphaIndexTimes(
    int alpha_index_times_count,
    int (*alpha_index_times)[4])
{
    this->_alpha_index_times_count = alpha_index_times_count;
    this->_alpha_index_times = (int (*)[4])malloc(sizeof(int) * this->_alpha_index_times_count * 4);
    for (int ii=0; ii<this->_alpha_index_times_count; ii++)
        for (int jj=0; jj<4; jj++)
            this->_alpha_index_times[ii][jj] = alpha_index_times[ii][jj];
}

AlphaIndexTimes::AlphaIndexTimes(const AlphaIndexTimes &rhs)
{
    this->_alpha_index_times_count = rhs._alpha_index_times_count;
    this->_alpha_index_times = (int (*)[4])malloc(sizeof(int) * this->_alpha_index_times_count * 4);
    for (int ii=0; ii<this->_alpha_index_times_count; ii++)
        for (int jj=0; jj<4; jj++)
            this->_alpha_index_times[ii][jj] = rhs._alpha_index_times[ii][jj];
}

AlphaIndexTimes::AlphaIndexTimes(AlphaIndexTimes &&rhs)
{
    if (this != &rhs) {
        this->_alpha_index_times_count = rhs._alpha_index_times_count;
        this->_alpha_index_times = rhs._alpha_index_times;

        rhs._alpha_index_times_count = 0;
        rhs._alpha_index_times = nullptr;
    }
}

AlphaIndexTimes& AlphaIndexTimes::operator=(const AlphaIndexTimes &rhs)
{
    if (rhs._alpha_index_times_count != 0) {
        free(this->_alpha_index_times);
        this->_alpha_index_times_count = 0;
    }

    this->_alpha_index_times_count = rhs._alpha_index_times_count;
    this->_alpha_index_times = (int (*)[4])malloc(sizeof(int) * this->_alpha_index_times_count * 4);
    for (int ii=0; ii<this->_alpha_index_times_count; ii++)
        for (int jj=0; jj<4; jj++)
            this->_alpha_index_times[ii][jj] = rhs._alpha_index_times[ii][jj];
    
    return *this;
}

AlphaIndexTimes& AlphaIndexTimes::operator=(AlphaIndexTimes &&rhs)
{
    if (this != &rhs)
    {
        if (rhs._alpha_index_times_count != 0) {
            free(this->_alpha_index_times);
            this->_alpha_index_times_count = 0;
        }
        this->_alpha_index_times_count = rhs._alpha_index_times_count;
        this->_alpha_index_times = rhs._alpha_index_times;

        rhs._alpha_index_times_count = 0;
        rhs._alpha_index_times = nullptr;
    }
    return *this;
}

AlphaIndexTimes::~AlphaIndexTimes()
{
    free(this->_alpha_index_times);
    this->_alpha_index_times_count = 0;
}

void AlphaIndexTimes::show() const
{
    printf("1. alpha_index_basic_count = %5d\n", this->_alpha_index_times_count);
    printf("2. alpha_index_basic:\n");
    for (int ii=0; ii<this->_alpha_index_times_count; ii++)
        printf(
            "\t%4d : {%4d, %4d, %4d, %4d}\n", 
            ii,
            this->_alpha_index_times[ii][0],
            this->_alpha_index_times[ii][1],
            this->_alpha_index_times[ii][2],
            this->_alpha_index_times[ii][3]);
}

int AlphaIndexTimes::alpha_index_times_count() const
{
    return this->_alpha_index_times_count;
}

const int (*AlphaIndexTimes::alpha_index_times() const)[4]
{
    return this->_alpha_index_times;
}

};  // namespace : nnmtp
};  // namespace : ai2pot
