#include <iostream>

#include "../include/CpuNeighborList.h"


namespace matersdk {

/**
 * @brief Construct a new Voxels object
 * 
 * @param blockSize 
 * @param vsy           Voxel Size along Y axis
 * @param vsz           Voxel Size along Z axis
 * @param miny          min y
 * @param maxy          max y
 * @param minz          min z
 * @param maxz          max z
 * @param boxVectors    box Vectors (3*3)
 * @param usePeriodic   Is periodic or not
 */
CpuNeighborList::Voxels::Voxels(
        int blockSize, 
        float vsy, float vsz, 
        float miny, float maxy, float minz, float maxz,
        const Vec3* boxVectors, bool usePeriodic) :
    blockSize(blockSize), 
    voxelSizeY(vsy), voxelSizeZ(vsz),
    miny(miny), maxy(maxy), minz(minz), maxz(maxz),
    usePeriodic(usePeriodic)
{
    // Step 1. Initialize `this->periodicBoxVectors`
    for (int ii=0; ii<3; ii++) {
        for (int jj=0; jj<3; jj++) {
            float temp = (float)boxVectors[ii][jj];
            this->periodicBoxVectors[ii][jj] = temp;
        }
    }
    this->periodicBoxSize[0] = (float)boxVectors[0][0];
    this->periodicBoxSize[1] = (float)boxVectors[1][1];
    this->periodicBoxSize[2] = (float)boxVectors[2][2];
    
    // Step 2. Initialize `this->recipBoxVectors`
    this->recipBoxSize[0] = (float)(1.0 / boxVectors[0][0]);
    this->recipBoxSize[1] = (float)(1.0 / boxVectors[1][1]);
    this->recipBoxSize[2] = (float)(1.0 / boxVectors[2][2]);

    // Step 3. 判断Box是否是三斜的 (triclinic)
    this->triclinic = (
            boxVectors[0][1] != 0.0 || boxVectors[0][2] != 0.0 ||
            boxVectors[1][0] != 0.0 || boxVectors[1][2] != 0.0 ||
            boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0.0
        );

    // Step 4. 
    if (this->usePeriodic) {
        // Note: 四舍五入，让 Voxel 分的更均匀
        this->ny = (int) floorf(boxVectors[1][1] / this->voxelSizeY + 0.5f);
        this->nz = (int) floorf(boxVectors[2][2] / this->voxelSizeZ + 0.5f);
        this->voxelSizeY = boxVectors[1][1] / this->ny;
        this->voxelSizeY = boxVectors[2][2] / this->nz;
    } else {
        
    }
    
    this->bins.resize(ny);
    for (int ii=0; ii<ny; ii++) {
        this->bins[ii].resize(nz);
    }
} 


/**
 * @brief Get the voxel index containing a particular location.
 * 
 * @param location The cart coordinates of atom.
 * @return VoxelIndex 
 */
VoxelIndex CpuNeighborList::Voxels::getVoxelIndex(const float* location) const 
{
    float y_periodic, z_periodic;
    if (!this->usePeriodic) {
        y_periodic = location[1] - miny;
        z_periodic = location[2] - minz;
    } else {
        float shiftAlongZ = floorf(location[2] * this->recipBoxSize[2]);
        z_periodic = location[2] - this->periodicBoxVectors[2][2] * shiftAlongZ;
        y_periodic = location[1] - this->periodicBoxVectors[2][1] * shiftAlongZ;
        float shiftAlongY = floorf(y_periodic * this->recipBoxSize[1]);
        y_periodic -= this->periodicBoxVectors[1][1] * shiftAlongY;
    }
    int voxel_index_y = std::max(0, std::min(this->ny - 1, int(floorf(y_periodic/this->voxelSizeY))));
    int voxel_index_z = std::max(0, std::min(this->nz - 1, int(floorf(z_periodic/this->voxelSizeZ))));

    return VoxelIndex(voxel_index_y, voxel_index_z);
}


/**
 * @brief Insert a particle into voxel data structure
 * 
 * @param atom      The index of atom
 * @param location  The cart coordinate of atom
 */
void CpuNeighborList::Voxels::insert(const int &atom, const float *location) {
    VoxelIndex voxelIndex = this->getVoxelIndex(location);
    //std::cout << voxelIndex.y << ", " << voxelIndex.z << std::endl;
    this->bins[voxelIndex.y][voxelIndex.z].push_back(std::make_pair(location[0], atom));
}


/**
 * @brief Sort the atom in each voxel by X coordinate
 * 
 * @example std::cout << item.first << ", " << item.second << std::endl;
 *      0.8, 1
 *      0.9, 2
 *      1, 0
 */
void CpuNeighborList::Voxels::sortItems() {
    for (int ii=0; ii<this->ny; ii++) {
        for (int jj=0; jj<this->nz; jj++) {
            std::sort(this->bins[ii][jj].begin(), this->bins[ii][jj].end());
        }
    }
}


/**
 * @brief Find the index of the first particle in voxel (y, z)
 * whose x coordinate is 
 * 
 * @param y         The index of voxel along Y axis
 * @param z         The index of voxel along Z axis
 * @param x
 * @param lower
 * @param upper
 */
int CpuNeighborList::Voxels::findLowerBound(
                    int y, int z, double x, int lower, int upper) const {
    const std::vector<std::pair<float, int>> &bin = this->bins[y][z];
    while (lower < upper) {
        int middle = (lower + upper) / 2;
        if (bin[middle].first < x) {
            lower = middle+1;
        } else {
            upper = middle;
        }
    }
    return lower;
}


/**
 * @brief Find the index of the first particle in voxel (y, z)
 * whose x coordinate is >= specified value 
 * 
 * @param y         The index of voxel along Y axis
 * @param z         The index of voxel along Z axis
 * @param x
 * @param lower
 * @param upper 
 */
int CpuNeighborList::Voxels::findUpperBound(
                    int y, int z, double x, int lower, int upper) const {
    const std::vector<std::pair<float, int>> &bin = this->bins[y][z];
    while (lower < upper) {
        int middle = (lower + upper) / 2;
        if (bin[middle].first > x) {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    return upper;
}


} // namespace: matersdk