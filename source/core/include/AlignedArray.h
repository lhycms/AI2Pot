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

#ifndef CORE_ALIGNED_ARRAY_H
#define CORE_ALIGNED_ARRAY_H


namespace ai2pot {


/**
 * @brief This class represents an array in memory whose starting 
 * point is guaranteed to be aligned with a 16 byte boundary. 
 * This can improve the performance of vectorized code, since 
 * loads and stores are more efficient.
 * 
 * @tparam T 
 */
template <typename T>
class AlignedArray {

public:
    /**
     * @brief Default constructor, to allow AlignedArrays to be 
     * used inside collections
     */
    AlignedArray() : data_size(0), base_ptr(0), data(0)
    {};


    /**
     * @brief Create an Aligned Array that contains a specified
     * number of elements
     */
    AlignedArray(int size) {
        this->allocate(size);
    }

    /**
     * @brief Destructor
     */
    ~AlignedArray() {
        delete [] this->base_ptr;
    }

    /**
     * @brief Get the number of elements in the array 
     */
    const int size() const {
        return this->data_size;
    }

    /**
     * @brief Change the size of the array. 
     * 
     * @note This may cause all contents to be lost.
     * 
     */
    void resize(int size) {
        if (this->data_size == size)
            return;
        if (this->base_ptr != 0)
            delete [] this->base_ptr;
        this->allocate(size);
    }

    /**
     * @brief Get the reference to an element of the array.
     */
    T& operator[](int index) {
        return this->data[index];
    }

    /**
     * @brief Get a const reference to an element of the array
     */
    const T& operator[](int index) const {
        return this->data[index];
    }


private:
    int data_size;
    char* base_ptr;
    T* data;

    /**
     * @brief Get `this->size`, `this->base_ptr`, `this->data` and allocate memory
     * 
     * @param size The memory's size of data(pointer[T]) to point
     */
    void allocate(int size) {
        this->data_size = size;
        this->base_ptr = new char[sizeof(T) * size + 15];
        char* expanded_ptr = this->base_ptr + 15;
        char* aligned_ptr = (char*)( ( (long long)expanded_ptr ) & (~0x0F) );
        this->data = (T*)aligned_ptr;
    }
};  /* class: AlignedArray */


} /* namespace: · */

#endif