#include <stdio.h>
#include <math.h>
#include <vector>

#include "../../nblist/include/structure.h"
#include "../../nblist/include/neighborList.h"


namespace ai2pot {
namespace gst {

template <typename CoordType>
class Altbc {
public:
    static void find_long_short_bonds(std::vector<CoordType>& long_bond_lengths,
                                      std::vector<CoordType>& short_bond_lengths,
                                      int inum,
                                      int* ilist,
                                      int* numneigh,
                                      int* firstneigh,
                                      CoordType* rcs,
                                      int* types,
                                      int nghost,
                                      int umax_num_neigh_atoms,
                                      CoordType angle_threshold);
};  // class : Altbc

template <typename CoordType>
void Altbc<CoordType>::find_long_short_bonds(std::vector<CoordType>& long_bond_lengths,
                                             std::vector<CoordType>& short_bond_lengths,
                                             int inum,
                                             int* ilist,
                                             int* numneigh,
                                             int* firstneigh,
                                             CoordType* rcs,
                                             int* types,
                                             int nghost,
                                             int umax_num_neigh_atoms,
                                             CoordType angle_threshold)
{
    CoordType* neigh_vec1;
    CoordType* neigh_vec2;
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<numneigh[ii]; jj++) {
            for (int kk=jj; kk<numneigh[jj]; kk++) {
                neigh_vec1 = &rcs[ii*umax_num_neigh_atoms*3 + jj*3];
                neigh_vec2 = &rcs[ii*umax_num_neigh_atoms*3 + kk*3];
                CoordType distance1 = std::sqrt(std::pow(neigh_vec1[0], 2) + std::pow(neigh_vec1[1], 2) + std::pow(neigh_vec1[2], 2));
                CoordType distance2 = std::sqrt(std::pow(neigh_vec2[0], 2) + std::pow(neigh_vec2[1], 2) + std::pow(neigh_vec2[2], 2));
                CoordType cos_val = (neigh_vec1[0]*neigh_vec2[0] + neigh_vec1[1]*neigh_vec2[1] + neigh_vec1[2]*neigh_vec2[2])
                                    / ( distance1 * distance2 );
                if (cos_val < std::cos(angle_threshold * M_PI / 180.0)) {
                    if (distance1 < distance2) {
                        short_bond_lengths.push_back(distance1);
                        long_bond_lengths.push_back(distance2);
                    } else {
                        short_bond_lengths.push_back(distance2);
                        long_bond_lengths.push_back(distance1);
                    }
                }

            }
        }
    }
}

};  // namespace : gst
};  // namespace : ai2pot