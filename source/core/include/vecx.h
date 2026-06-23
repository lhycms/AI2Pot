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

#ifndef CORE_VECX_H
#define CORE_VECX_H

#include <iostream>
#include <cassert>


namespace ai2pot {

/**
 * @brief This Class represents a three component vector. It is used for storing positions, 
 * velocities, and forces 
 */
class Vec3 {
public:
    /**
     * @brief Create a Vec3 whose elements are all 0.
     * 
     */
    Vec3() {
        this->data[0] = this->data[1] = this->data[2] = 0.0;
    }

    /**
     * @brief Create a Vec3 with specified x, y, and z components
     * 
     */
    Vec3(double x, double y, double z) {
        this->data[0] = x;
        this->data[1] = y;
        this->data[2] = z;
    }

    /**
     * @brief Take value using index
     * @param index int. 
     */
    double operator[](int index) const {
        assert(index>=0 && index<3);
        return this->data[index];
    }

    double& operator[](int index) {
        assert(index>=0 && index<3);
        return this->data[index];
    }

    /**
     * @brief Overload the `==` operator
     * 
     */
    bool operator==(const Vec3 &rhs) const {
        const Vec3 &lhs = *this;
        return (
                lhs[0] == rhs[0] && \
                lhs[1] == rhs[1] && \
                lhs[2] == rhs[2]
        );
    }

    /**
     * @brief Overload the `!=` operator 
     * 
     * @param rhs
     * @return 
     */
    bool operator!=(const Vec3 &rhs) const {
        const Vec3 &lhs = *this;
        return (
                lhs[0] != rhs[0] || \
                lhs[1] != rhs[1] || \
                lhs[2] != rhs[2]
        );
    }

    /**
     * @brief Overload the `+`(unary plus) operator
     * 
     */
    Vec3 operator+() const {
        const Vec3 &lhs = *this;
        return Vec3(lhs[0], lhs[1], lhs[2]);
    }


    /**
     * @brief Overload the `+`(binary plus) operator
     * 
     */
    Vec3 operator+(const Vec3 &rhs) const {
        const Vec3 &lhs = *this;
        return Vec3(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
    }

    /**
     * @brief Overload the `+=` operator
     * 
     */
    Vec3& operator+=(const Vec3 &rhs) {
        this->data[0] += rhs[0];
        this->data[1] += rhs[1];
        this->data[2] += rhs[2];
        return *this;
    }

    /**
     * @brief Overload the `-` (unary minus) oprator
     * 
     */
    Vec3 operator-() const {
        const Vec3 &lhs = *this;
        return Vec3(-lhs[0], -lhs[1], -lhs[2]);
    }

    /**
     * @brief Overload the `-` (binary minus) operator
     *  
     */
    Vec3 operator-(const Vec3 &rhs) const {
        const Vec3 &lhs = *this;
        return Vec3(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
    }

    /**
     * @brief Overload the `-=` operator
     * 
     */
    Vec3& operator-=(const Vec3 &rhs) {
        this->data[0] -= rhs[0];
        this->data[1] -= rhs[1];
        this->data[2] -= rhs[2];
        return *this;
    }

    /**
     * @brief scalar product
     * 
     */
    Vec3 operator*(double rhs) const {
        const Vec3 &lhs = *this;
        return Vec3(lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs);
    }

    /**
     * @brief self scalar product
     * 
     */
    Vec3& operator*=(double rhs) {
        this->data[0] *= rhs;
        this->data[1] *= rhs;
        this->data[2] *= rhs;
        return *this;
    }

    /**
     * @brief scalar division
     * 
     */
    Vec3 operator/(double rhs) const {
        const Vec3 &lhs = *this;
        double scaling_factor = 1.0 / rhs;
        return Vec3(
                lhs[0]*scaling_factor,
                lhs[1]*scaling_factor,
                lhs[2]*scaling_factor
        );
    }

    /**
     * @brief self scalar division
     * 
     */
    Vec3& operator/=(double rhs) {
        double scaling_factor = 1.0 / rhs;
        this->data[0] *= scaling_factor;
        this->data[1] *= scaling_factor;
        this->data[2] *= scaling_factor;
        return *this;
    }

    /**
     * @brief dot
     * 
     */
    double dot(const Vec3 &rhs) const {
        const Vec3 &lhs = *this;
        return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
    }

    /**
     * @brief cross
     * 
     */
    Vec3 cross(const Vec3 &rhs) const {
        const Vec3 &lhs = *this;
        return Vec3(
            lhs[1]*rhs[2] - lhs[2]*rhs[1],
            lhs[2]*rhs[0] - lhs[0]*rhs[2],
            lhs[0]*rhs[1] - lhs[1]*rhs[0]
        );
    }


private:
    double data[3];
};


/**
 * @brief 
 * 
 * @param lhs 
 * @param rhs 
 * @return Vec3 
 */
Vec3 operator*(double lhs, const Vec3 &rhs) {
    return Vec3(rhs[0]*lhs, rhs[1]*lhs, rhs[2]*lhs);
}


std::ostream& operator<<(std::ostream &COUT, const Vec3 &rhs) {
    COUT << "[" << rhs[0] << ", " << rhs[1] << ", " << rhs[2] << "]";
    return COUT;
}


}   // namespace `ai2pot`




#endif /* CORE_VECX_H */