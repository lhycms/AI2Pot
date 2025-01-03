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

#ifndef AI2POT_MTP_LEVEL_H
#define AI2POT_MTP_LEVEL_H
#include <utility>
#include <vector>


namespace ai2pot {
namespace mtp {

class MtpMCoeffPair {
public:
    MtpMCoeffPair();

    MtpMCoeffPair(int mu, int nu);

    MtpMCoeffPair(std::pair<int, int> coeff_pair);

    MtpMCoeffPair(const MtpMCoeffPair& rhs);

    ~MtpMCoeffPair();

    MtpMCoeffPair& operator=(const MtpMCoeffPair& rhs);

    void _calc_level();

    const int level() const;

    std::pair<int, int> coeff_pair() const;

private:
    std::pair<int, int> _coeff_pair = std::pair<int, int>(0, 0);
    int _level = 0;
};  // class : MtpMCoeffPair


class CoeffPairSortFunction {
public:
    bool operator()(const MtpMCoeffPair& cp1, const MtpMCoeffPair& cp2);
};  // class : CoeffPairSortFunction


class MtpMCoeffPairCombs {
public:
    MtpMCoeffPairCombs();

    MtpMCoeffPairCombs(int max_level);

    MtpMCoeffPairCombs(const MtpMCoeffPairCombs& rhs);

    ~MtpMCoeffPairCombs();

    MtpMCoeffPairCombs& operator=(const MtpMCoeffPairCombs& rhs);

    static std::vector<std::vector<MtpMCoeffPair>> get_all_schemes4lev(
        int aim_level,
        int start_idx_mu,
        int start_idx_nu);
    
    static std::vector<std::vector<MtpMCoeffPair>> get_contracted_combs(
        const std::vector<std::vector<MtpMCoeffPair>>& coeff_pair_combs);

    void _build();

    const int max_level() const;

    const std::vector<std::vector<MtpMCoeffPair>>& coeff_pair_combs() const;

    const int size() const;

    const int nmus() const;

    const int nmus_check() const;

    template <typename Arg>
    decltype(auto) operator[](Arg&& arg)
    {
        return this->_coeff_pair_combs[std::forward(arg)];
    }

    void show() const;

private:
    int _max_level = 0;
    std::vector<std::vector<MtpMCoeffPair>> _coeff_pair_combs = std::vector<std::vector<MtpMCoeffPair>>();
};  // class : MtpMCoeff


};  // namespace : mtp
};  // namespace : ai2pot

#endif