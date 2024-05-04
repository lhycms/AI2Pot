#ifndef MATERSDK_LEVEL_H
#define MATERSDK_LEVEL_H


#include <vector>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <cmath>

namespace matersdk {
namespace mtp {


class Combinations {
public:
    Combinations(std::vector<std::vector<std::pair<int, int>>> mjus_njus_lst, bool mark_tiny=true);

    // Remove duplicated combinations
    void remove_duplicates();

    // Remove the combinations which cannot be tensor contracted
    void remove_cannot_contract();

    void show() const;

    const std::vector<std::vector<std::pair<int, int>>> get_combinations() const;

    // Get the number of combinations
    const int get_num_combinations() const;

    static int get_level(const int mju, const int nju);

private:
    std::vector<std::vector<std::pair<int, int>>> mjus_njus_lst;
};  // class: Combinations



/**
 * @brief The Sort Basis for Combinations
 * 
 */
class CombinationsSortBasis {
public:
    CombinationsSortBasis(const Combinations rhs) : combinations(rhs)
    {}


    bool operator()(int index_i, int index_j) const {
        // Step 1. 计算 index_i 的 mtp level
        int level_index_i = 0;
        for (int ii=0; ii<combinations.get_combinations()[index_i].size(); ii++) {
            level_index_i += Combinations::get_level(
                                    combinations.get_combinations()[index_i][ii].first, 
                                    combinations.get_combinations()[index_i][ii].second);
        }

        // Step 2. 计算 index_j 的 mtp level
        int level_index_j = 0;
        for (int ii=0; ii<combinations.get_combinations()[index_j].size(); ii++) {
            level_index_j += Combinations::get_level(
                                    combinations.get_combinations()[index_j][ii].first,
                                    combinations.get_combinations()[index_j][ii].second);
        }

        return (level_index_i < level_index_j);
    }

private:
    Combinations combinations;
};  // class : CombinationsSortBasis



/**
 * @brief The Arrangement for Combinations
 * 
 */
class CombinationsArrangement {
public:
    CombinationsArrangement(const Combinations combinations, int* new_indices) : combinations(combinations), new_indices(new_indices)
    {}

    Combinations arrange() const {
        std::vector<std::vector<std::pair<int, int>>> sorted_mjus_njus_lst;
        sorted_mjus_njus_lst.clear();
        for (int ii=0; ii<this->combinations.get_combinations().size(); ii++) {
            sorted_mjus_njus_lst.push_back(combinations.get_combinations()[this->new_indices[ii]]);
        }

        Combinations sorted_combinations(sorted_mjus_njus_lst);
        return sorted_combinations;
    };

private:
    Combinations combinations;
    int* new_indices;
};


/**
 * @brief MTPLevel
 * 
 */
class MTPLevel {
public:
    MTPLevel(int max_level);

    int get_max_num_M();

    std::vector<std::vector<std::pair<int, int>>> get_redundant_combinaions();

    // find all Combination: mju, nju (which is 冗余)
    void calc_redundant_combination(
        int num_M, 
        int max_level,
        int level,  // 0
        std::vector<std::pair<int, int>>& combinations // None
    );

    void calc_redundant_combinations(
        int max_level,
        std::vector<std::pair<int, int>>& combinations // None
    );

private:
    int max_level = 0;
    int max_num_M = 0;
    std::vector<std::vector<std::pair<int, int>>> redundant_combinations;
};  // class: MTPLevel



/**
 * @brief Construct a new Combinations:: Combinations object
 * 
 * @param mjus_njus_lst : std::vector<std::vector<std::pair<int, int>>>
 * @param sort_unique_mark : 
 */
Combinations::Combinations(std::vector<std::vector<std::pair<int, int>>> mjus_njus_lst, bool mark_tiny) {
    // Step 1. Init the Combinations with `std::vector<std::vector<std::pair<int, int>>>`
    this->mjus_njus_lst.resize(mjus_njus_lst.size());
    for (int ii=0; ii<mjus_njus_lst.size(); ii++) 
        this->mjus_njus_lst[ii].resize(mjus_njus_lst[ii].size());
    
    for (int ii=0; ii<mjus_njus_lst.size(); ii++) {
        for (int jj=0; jj<mjus_njus_lst[ii].size(); jj++) {
            this->mjus_njus_lst[ii][jj] = mjus_njus_lst[ii][jj];
        }
    }

    // Step 2. 
    if (mark_tiny) {
        // Step 2.1. remove duplicates combinations.
        this->remove_duplicates();

        // Step 2.2. remove combinations which cannot be contracted.
        this->remove_cannot_contract();
    }
}


/**
 * @brief Remove the duplicated combination, e.g. [(mju0, nju0), (mju1, nju1), (mju2, nju2), ...]
 * 
 */
void Combinations::remove_duplicates() {
    std::vector<std::vector<std::pair<int, int>>> new_mjus_njus_lst;
    new_mjus_njus_lst.clear();

    for (int ii=0; ii<this->get_num_combinations(); ii++) {
        for (int jj=0; jj<this->mjus_njus_lst[ii].size()-1; jj++) {
            int level_front = Combinations::get_level(
                                    this->mjus_njus_lst[ii][jj].first, 
                                    this->mjus_njus_lst[ii][jj].second);
            int level_behind = Combinations::get_level(
                                    this->mjus_njus_lst[ii][jj+1].first,
                                    this->mjus_njus_lst[ii][jj+1].second);
                
                if (level_front <= level_behind) {
                    new_mjus_njus_lst.push_back(this->mjus_njus_lst[ii]);
                }
        }
    }

    // 重新为 `this->mjus_njus_lst` 赋值
    this->mjus_njus_lst = new_mjus_njus_lst;
}


/**
 * @brief Remove the combination which cannot be tensor contracted, e.g. [(mju0, nju0), (mju1, nju1), (mju2, nju2), ...]
 * 
 */
void Combinations::remove_cannot_contract() {
    // Step 1. 初始化一个新的 `new_mjus_njus_lst`
    std::vector<std::vector<std::pair<int, int>>> new_mjus_njus_lst;
    new_mjus_njus_lst.clear();

    // Step 2. Populate `new_mjus_njus_lst`
    for (int ii=0; ii<this->mjus_njus_lst.size(); ii++) {   // 循环每一组 [(mju0, nju0), (mju1, nju1), ...]
        std::vector<int> njus_not_zero_lst;                 // 每一组 [(mju0, nju0), (mju1, nju1), ...] 中的 nju0, nju1, ... (nju值非零)
        njus_not_zero_lst.clear();

        // Step 2.1. Populate `njus_not_zero_lst`
        for (int jj=0; jj<this->mjus_njus_lst[ii].size(); jj++)    // 循环 [(mju0, nju0), (mju1, nju1), ...] 中的 (mjux, njux)
            njus_not_zero_lst.push_back(this->mjus_njus_lst[ii][jj].second);

/*        
        for (auto value: njus_not_zero_lst)
            printf("%3d,\t", value);
        printf("\n");

        Output
        ------
                0,      0,
                0,      1,
                0,      2,
                0,      3,
                0,      4,
                0,      0,
                1,      1,
                1,      2,
                1,      3,
                2,      2,
         */


        // Step 2.2. Count the occurrences of each nju
        /*
                0: 2,
                1: 1,   0: 1,
                2: 1,   0: 1,
                3: 1,   0: 1,
                4: 1,   0: 1,
                0: 2,
                1: 2,
                2: 1,   1: 1,
                3: 1,   1: 1,
                2: 2,
         */
        std::unordered_map<int, int> nju2num;   // 针对每个 combination (mju, nju)
        for (const int& tmp_nju: njus_not_zero_lst)
            nju2num[tmp_nju]++;

        // Step 2.3. Populate `mark_contraction` #Note!!!
        bool mark_contraction = true;
        for (const std::pair<int, int>& tmp_nju2num: nju2num) {
            // Step 2.3.1. case 1: 各种 nju 均有偶数个
            //printf("%d: %d,\t", tmp_nju2num.first, tmp_nju2num.second);   // Note!!!
            if ((tmp_nju2num.first != 0) && (tmp_nju2num.second % 2 != 0))
                mark_contraction = false;
        }
        //printf("\n");  // Note!!!

        // Step 2.4. 根据 `mark_contraction` 判断是否需要将 `this->mjus_njus_lst[ii]` 放入 `new_mjus_njus_lst`
        if (mark_contraction)
            new_mjus_njus_lst.push_back(this->mjus_njus_lst[ii]);
    }


    this->mjus_njus_lst = new_mjus_njus_lst;
}


void Combinations::show() const {
    for (int ii=0; ii<this->mjus_njus_lst.size(); ii++) {
        printf("[mju, nju] :\t");
        int level = 0;
        for (int jj=0; jj<this->mjus_njus_lst[ii].size(); jj++) {
            printf(
                "[%4d, %4d],  ", 
                this->mjus_njus_lst[ii][jj].first, 
                this->mjus_njus_lst[ii][jj].second
            );
            level += Combinations::get_level(this->mjus_njus_lst[ii][jj].first, this->mjus_njus_lst[ii][jj].second);
        }
        printf("level = %4d,\tno.%3d\n", level, ii);
    }
}


const std::vector<std::vector<std::pair<int, int>>> Combinations::get_combinations() const {
    return (const std::vector<std::vector<std::pair<int, int>>>)this->mjus_njus_lst;
}


const int Combinations::get_num_combinations() const {
    return this->mjus_njus_lst.size();
}


int Combinations::get_level(const int mju, const int nju) {
    return (2 + 4*mju + nju);
}



MTPLevel::MTPLevel(int max_level) {
    this->max_level = max_level;
    this->max_num_M = this->get_max_num_M();
    this->redundant_combinations.clear();
}


int MTPLevel::get_max_num_M() {
    int ii;
    int current_level = 0;
    for (ii=0; current_level < this->max_level; ii++) {
        current_level += 2;
    }

    return ii;
}


std::vector<std::vector<std::pair<int, int>>> MTPLevel::get_redundant_combinaions() {
    return this->redundant_combinations;
}


void MTPLevel::calc_redundant_combination(
    int num_M,
    int max_level,
    int level,
    std::vector<std::pair<int, int>>& combination)
{
    if (num_M == 0) {
        if (level <= max_level)
            this->redundant_combinations.push_back(combination);
        return ;
    }

    for (int tmp_mju=0; tmp_mju<=max_level+1; tmp_mju++) {
        for (int tmp_nju=0; tmp_nju<=max_level+1; tmp_nju++) {
            int new_level = level + Combinations::get_level(tmp_mju, tmp_nju);
            if (new_level <= max_level) {
                std::vector<std::pair<int, int>> new_combination = combination;
                new_combination.push_back(std::pair<int, int>(tmp_mju, tmp_nju));
                MTPLevel::calc_redundant_combination(num_M-1, max_level, new_level, new_combination);
            }
        }
    }
}


void MTPLevel::calc_redundant_combinations(
    int max_level,
    std::vector<std::pair<int, int>>& combination)
{
    for (int ii=0; ii<this->max_num_M; ii++) {
        int level = 0;
        std::vector<std::pair<int, int>> combination;
        combination.clear();

        this->calc_redundant_combination(ii, max_level, level, combination);
    }
}


}; // namespace : mtp
}; // namespace : matersdk
#endif