#include <iostream>
#include <vector>
#include <stdio.h>
#include <utility>


class MlipCoeffPairCombs 
{
public:
    MlipCoeffPairCombs(
        int alpha_moment_count,
        std::vector<std::vector<int>> alpha_index_basic,
        std::vector<std::vector<int>> alpha_index_times,
        std::vector<int> alpha_moment_mapping)
            :   _alpha_moment_count(alpha_moment_count),
                _alpha_index_basic(alpha_index_basic), 
                _alpha_index_times(alpha_index_times), 
                _alpha_moment_mapping(alpha_moment_mapping)
    {
        this->_build();
    }

    void _build() {
        this->_coeff_pair_combs.resize(this->_alpha_moment_count);
        for (int ii=0; ii<this->_alpha_index_basic.size(); ii++) {
            this->_coeff_pair_combs[ii] = std::vector<std::pair<int, int>>(
                {{this->_alpha_index_basic[ii][0],
                this->_alpha_index_basic[ii][1] + this->_alpha_index_basic[ii][2] + this->_alpha_index_basic[ii][3]}}
            );
        }

        for (int ii=0; ii<this->_alpha_index_times.size(); ii++) {
            int idx0 = this->_alpha_index_times[ii][0];
            int idx1 = this->_alpha_index_times[ii][1];
            int idx3 = this->_alpha_index_times[ii][3];
            std::vector<std::pair<int, int>> coeff_pair_comb;
            coeff_pair_comb.clear();
            for (auto& tmp_coeff_pair : this->_coeff_pair_combs[idx0])
                coeff_pair_comb.push_back(tmp_coeff_pair);
            for (auto& tmp_coeff_pair : this->_coeff_pair_combs[idx1])
                coeff_pair_comb.push_back(tmp_coeff_pair);
            this->_coeff_pair_combs[idx3] = coeff_pair_comb;
        }

        std::vector<std::vector<std::pair<int, int>>> contracted_coeff_pair_combs;
        contracted_coeff_pair_combs.clear();
        for (int ii=0; ii<this->_alpha_moment_mapping.size(); ii++) {
            int idx = this->_alpha_moment_mapping[ii];
            contracted_coeff_pair_combs.push_back(this->_coeff_pair_combs[idx]);
        }
        this->_coeff_pair_combs = contracted_coeff_pair_combs;

        this->_num_combs = this->_coeff_pair_combs.size();
    }

    const int num_combs()
    {
        return this->_num_combs;
    }

    std::vector<std::vector<std::pair<int, int>>> coeff_pair_combs()
    {
        return this->_coeff_pair_combs;
    }

private:
    int _num_combs;
    int _alpha_moment_count;
    std::vector<std::vector<int>> _alpha_index_basic;
    std::vector<std::vector<int>> _alpha_index_times;
    std::vector<int> _alpha_moment_mapping;
    std::vector<std::vector<std::pair<int, int>>> _coeff_pair_combs;
};  // class : MlipCoeffPairCombs




std::ostream& operator<<(std::ostream& COUT, MlipCoeffPairCombs rhs)
{
    for (int ii=0; ii<rhs.num_combs(); ii++) {
        std::vector<std::pair<int, int>> tmp_comb = rhs.coeff_pair_combs()[ii];
        printf("Combs #%5d:\n\t", ii);
        for (int jj=0; jj<tmp_comb.size(); jj++) {
            printf("[%3d, %3d], ", tmp_comb[jj].first, tmp_comb[jj].second);
        }
        printf("\n");
    }
    return COUT;
}


int main() { 
    int alpha_moments_count = 41;
    std::vector<std::vector<int>> alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 2, 0, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {0, 0, 2, 0}, {0, 0, 1, 1}, {0, 0, 0, 2}, {0, 3, 0, 0}, {0, 2, 1, 0}, {0, 2, 0, 1}, {0, 1, 2, 0}, {0, 1, 1, 1}, {0, 1, 0, 2}, {0, 0, 3, 0}, {0, 0, 2, 1}, {0, 0, 1, 2}, {0, 0, 0, 3}, {1, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}, {2, 0, 0, 0}};
    std::vector<std::vector<int>> alpha_index_times = {{0, 0, 1, 25}, {1, 1, 1, 26}, {2, 2, 1, 26}, {3, 3, 1, 26}, {1, 4, 1, 27}, {2, 5, 1, 27}, {3, 6, 1, 27}, {1, 5, 1, 28}, {2, 7, 1, 28}, {3, 8, 1, 28}, {1, 6, 1, 29}, {2, 8, 1, 29}, {3, 9, 1, 29}, {4, 4, 1, 30}, {5, 5, 2, 30}, {6, 6, 2, 30}, {7, 7, 1, 30}, {8, 8, 2, 30}, {9, 9, 1, 30}, {10, 10, 1, 31}, {11, 11, 3, 31}, {12, 12, 3, 31}, {13, 13, 3, 31}, {14, 14, 6, 31}, {15, 15, 3, 31}, {16, 16, 1, 31}, {17, 17, 3, 31}, {18, 18, 3, 31}, {19, 19, 1, 31}, {0, 20, 1, 32}, {1, 21, 1, 33}, {2, 22, 1, 33}, {3, 23, 1, 33}, {0, 25, 1, 34}, {0, 26, 1, 35}, {0, 30, 1, 36}, {1, 27, 1, 37}, {2, 28, 1, 37}, {3, 29, 1, 37}, {0, 32, 1, 38}, {0, 34, 1, 39}, {0, 35, 1, 40}};
    std::vector<int> alpha_moment_mapping = {0, 20, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};

    MlipCoeffPairCombs mlip_coeff_pair_combs(alpha_moments_count, alpha_index_basic, alpha_index_times, alpha_moment_mapping);
    std::cout << mlip_coeff_pair_combs << std::endl;
    return 0;
}
