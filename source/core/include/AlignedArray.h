/**
 * @file AlignedArray.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-04
 * 
 * @copyright Copyright (c) 2023
 * 
 * @ref 1. https://github.com/openmm/openmm/blob/644dc1ecc9e95b3c8e831803bb3e2ac925999f74/platforms/cpu/include/AlignedArray.h#L4
 *      2. Array align at 16-byte boundary: https://philippegroarke.com/blog/2017/02/19/quicktip-understanding-16-byte-memory-alignment-detection/
 *      3. Array align at 16-byte boundary: http://www.songho.ca/misc/alignment/dataalign.html
 */    
#ifndef CORE_ALIGNED_ARRAY_H
#define CORE_ALIGNED_ARRAY_H


namespace matersdk {


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


} /* namespace: matersdk */

#endif