/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief  3D Hilbert encoding/decoding in 32- and 64-bit
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * This code is based on the implementation of the Hilbert curve presented in:
 *
 * Yohei Miki, Masayuki Umemura
 * GOTHIC: Gravitational oct-tree code accelerated by hierarchical time step controlling
 * https://doi.org/10.1016/j.newast.2016.10.007
 *
 * The 2D Hilbert curve  code is based on the book by Henry S. Warren
 * https://learning.oreilly.com/library/view/hackers-delight-second
 */

#pragma once

#include "morton.hpp"

namespace cstone
{

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ static unsigned mortonToHilbertDevice[8] = {0, 1, 3, 2, 7, 6, 4, 5};
#endif

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px,py,pz  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert(unsigned px, unsigned py, unsigned pz, int order = maxTreeLevel<KeyType>{}) noexcept
{
    assert(px < (1u << order));
    assert(py < (1u << order));
    assert(pz < (1u << order));

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    constexpr unsigned mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};
#endif

    KeyType key = 0;

    for (int level = order - 1; level >= 0; --level)
    {
        unsigned xi = (px >> level) & 1u;
        unsigned yi = (py >> level) & 1u;
        unsigned zi = (pz >> level) & 1u;

        // append 3 bits to the key
        unsigned octant = (xi << 2) | (yi << 1) | zi;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        key = (key << 3) + mortonToHilbertDevice[octant];
#else
        key = (key << 3) + mortonToHilbert[octant];
#endif

        // turn px, py and pz
        px ^= -(xi & ((!yi) | zi));
        py ^= -((xi & (yi | zi)) | (yi & (!zi)));
        pz ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

        if (zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = py;
            py          = pz;
            pz          = pt;
        }
        else if (!yi)
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }
    }

    return key;
}

template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert2D(unsigned px, unsigned py, int order = maxTreeLevel<KeyType>{}) noexcept;

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px        input x integer coordinate, in [0:2^bx]
 * @param[in]  py        input y integer coordinate, in [0:2^by]
 * @param[in]  pz        input z integer coordinate, in [0:2^bz]
 * @param[in]  bx        number of bits to encode in x dimension, in [0:maxTreelevel<KeyType>{}]
 * @param[in]  by        number of bits to encode in y dimension, in [0:maxTreelevel<KeyType>{}]
 * @param[in]  bz        number of bits to encode in z dimension, in [0:maxTreelevel<KeyType>{}]
 * @return               the Hilbert key
 *
 * Example box with (Lx, Ly, Lz) = (8,4,1):
 *  The longest dimension will get the max number of bits per dimension maxTreelevel<KeyType>{},
 *  i.e 10 bits if KeyType is 32-bit. The bits in the other dimensions are reduced by 1 for each
 *  factor of 2 that the box is shorter in that dimension than the longest. For the example box,
 *  (bx, by, bz) will be (10, 9, 7)
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbertMixD(unsigned px, unsigned py, unsigned pz, unsigned bx, unsigned by, unsigned bz) noexcept
{
    assert(px < (1u << bx));
    assert(py < (1u << by));
    assert(pz < (1u << bz));
    assert(bx <= maxTreeLevel<KeyType>{} && by <= maxTreeLevel<KeyType>{} && bz <= maxTreeLevel<KeyType>{});

    KeyType key = 0;

    std::array<unsigned, 3> bits{bx, by, bz};
    std::array<int, 3> permutation{0, 1, 2};
    std::sort(permutation.begin(), permutation.end(), [&bits](int i, int j) { return bits[i] > bits[j]; });
    std::array<unsigned, 3> coordinates{px, py, pz};
    std::array<unsigned, 3> sorted_coordinates{coordinates[permutation[0]], coordinates[permutation[1]],
                                               coordinates[permutation[2]]};
    std::sort(bits.begin(), bits.end(), std::greater<unsigned>{});

    std::cout << "px: " << std::bitset<10>(px) << " py: " << std::bitset<10>(py) << " pz: " << std::bitset<10>(pz)
              << std::endl;
    std::cout << "sorted_coordinates: " << std::bitset<10>(sorted_coordinates[0]) << " "
              << std::bitset<10>(sorted_coordinates[1]) << " " << std::bitset<10>(sorted_coordinates[2]) << std::endl;
    std::cout << "bits: " << bits[0] << " " << bits[1] << " " << bits[2] << std::endl;

    if (bits[0] > bits[1]) // 1 dim has more bits than the other 2 dims, add 1D levels
    {
        const int n = bits[0] - bits[1];
        // add n 1D levels and add to key (trivial)
        for (int i{0}; i < n; ++i)
        {
            const auto processes_bit_index = bits[0] - 1 - i;
            std::cout << "processed bit: " << ((sorted_coordinates[0] >> processes_bit_index) & 1) << std::endl;
            key |= ((sorted_coordinates[0] >> processes_bit_index) & 1) << (3 * processes_bit_index);
        }
        const unsigned mask = (1u << bits[1]) - 1;
        sorted_coordinates[0] &= mask;
        bits[0] -= n;
        // now we have bits[0] == bits[1]
    }
    std::cout << "After 1D levels" << std::endl;
    std::cout << "key:    " << std::bitset<32>(key) << std::endl;
    std::cout << "coordinate[0]: " << std::bitset<10>(sorted_coordinates[0]) << std::endl;
    if (bits[1] > bits[2]) // 2 dims have more bits than the 3rd, add 2D levels
    {
        const int n = bits[1] - bits[2];
        // encode n 2D levels with 2D-Hilbert and add to key
        const KeyType key_2D = iHilbert2D<KeyType>(sorted_coordinates[0], sorted_coordinates[1], n);
        std::cout << "key_2D: " << std::bitset<32>(key_2D) << std::endl;
        // IM: Check if we want to the 2D key together or break it from 2 bits per level to 3 bits per level
        key |= key_2D << (3 * bits[2]);
        // remove n bits from sorted_coordinates[0] and sorted_coordinates[1]
        const unsigned mask = (1u << bits[2]) - 1;
        sorted_coordinates[0] &= mask;
        sorted_coordinates[1] &= mask;
        bits[0] -= n;
        bits[1] -= n;
        // now we have bits[0] == bits[1] == bits[2]
    }
    std::cout << "After 2D levels" << std::endl;
    std::cout << "sorted_coordinates: " << std::bitset<10>(sorted_coordinates[0]) << " "
              << std::bitset<10>(sorted_coordinates[1]) << " " << std::bitset<10>(sorted_coordinates[2]) << std::endl;
    std::cout << "key:    " << std::bitset<32>(key) << std::endl;

    // encode remaining bits[0] == min(bx,by,bz) 3D levels or octal digits with 3D-Hilbert and add to key
    const KeyType key_3D =
        iHilbert<KeyType>(sorted_coordinates[0], sorted_coordinates[1], sorted_coordinates[2], bits[0]);
    std::cout << "key_3D: " << std::bitset<32>(key_3D) << std::endl;
    key |= key_3D;
    std::cout << "After 3D levels" << std::endl;
    std::cout << "key:    " << std::bitset<32>(key) << std::endl;
    // Example for (bx,by,bz) = (10,9,7): 1D,2D,2D,3D*7

    return key;
}

//! @brief inverse function of iHilbertMixD
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned, unsigned>
decodeHilbertMixD(KeyType key, unsigned bx, unsigned by, unsigned bz) noexcept
{
    return {0, 0, 0};
}

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px,py,pz  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert1DMixed(unsigned px, unsigned py, unsigned pz, unsigned levels_1D, axis long_dimension) noexcept
{
    assert(px < (1u << maxTreeLevel<KeyType>{}));
    assert(py < (1u << maxTreeLevel<KeyType>{}));
    assert(pz < (1u << maxTreeLevel<KeyType>{}));
    assert(levels_1D < maxTreeLevel<KeyType>{});
    assert(levels_1D > 0);
    assert(levels_1D % 2 == 0);

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    constexpr unsigned mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};
#endif

    KeyType key = 0;

    unsigned p_long_dimension = 0;
    if (long_dimension == axis::x)
    {
        assert(py < (1u << (maxTreeLevel<KeyType>{} - levels_1D)));
        assert(pz < (1u << (maxTreeLevel<KeyType>{} - levels_1D)));
        p_long_dimension = px;
    }
    else if (long_dimension == axis::y)
    {
        assert(px < (1u << (maxTreeLevel<KeyType>{} - levels_1D)));
        assert(pz < (1u << (maxTreeLevel<KeyType>{} - levels_1D)));
        p_long_dimension = py;
    }
    else
    {
        assert(px < (1u << (maxTreeLevel<KeyType>{} - levels_1D)));
        assert(py < (1u << (maxTreeLevel<KeyType>{} - levels_1D)));
        p_long_dimension = pz;
    }
    for (int level = maxTreeLevel<KeyType>{} - 2; level >= static_cast<int>(maxTreeLevel<KeyType>{} - levels_1D);
         level -= 2)
    {
        key = (key << 3) | ((p_long_dimension >> level) & 3);
    }

    for (int level = maxTreeLevel<KeyType>{} - levels_1D - 1; level >= 0; --level)
    {
        unsigned xi = (px >> level) & 1u;
        unsigned yi = (py >> level) & 1u;
        unsigned zi = (pz >> level) & 1u;

        // append 3 bits to the key
        unsigned octant = (xi << 2) | (yi << 1) | zi;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        key = (key << 3) + mortonToHilbertDevice[octant];
#else
        key = (key << 3) + mortonToHilbert[octant];
#endif

        // turn px, py and pz
        px ^= -(xi & ((!yi) | zi));
        py ^= -((xi & (yi | zi)) | (yi & (!zi)));
        pz ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

        if (zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = py;
            py          = pz;
            pz          = pt;
        }
        else if (!yi)
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }
    }
    return key;
}

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px,py,pz  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert2DMixed(unsigned px, unsigned py, unsigned pz, unsigned levels_2D, axis short_dimension) noexcept
{
    assert(px < (1u << maxTreeLevel<KeyType>{}));
    assert(py < (1u << maxTreeLevel<KeyType>{}));
    assert(pz < (1u << maxTreeLevel<KeyType>{}));
    assert(levels_2D < maxTreeLevel<KeyType>{});
    assert(levels_2D > 0);

    unsigned px_2D, py_2D;
    if (short_dimension == axis::x)
    {
        assert(px < (1u << (maxTreeLevel<KeyType>{} - levels_2D)));
        px_2D = py >> (maxTreeLevel<KeyType>{} - levels_2D);
        py_2D = pz >> (maxTreeLevel<KeyType>{} - levels_2D);
    }
    else if (short_dimension == axis::y)
    {
        assert(py < (1u << (maxTreeLevel<KeyType>{} - levels_2D)));
        px_2D = px >> (maxTreeLevel<KeyType>{} - levels_2D);
        py_2D = pz >> (maxTreeLevel<KeyType>{} - levels_2D);
    }
    else
    {
        assert(pz < (1u << (maxTreeLevel<KeyType>{} - levels_2D)));
        px_2D = px >> (maxTreeLevel<KeyType>{} - levels_2D);
        py_2D = py >> (maxTreeLevel<KeyType>{} - levels_2D);
    }

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    constexpr unsigned mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};
#endif

    KeyType key = 0;

    auto key_2d = iHilbert2D<KeyType>(px_2D, py_2D);
    for (int level{static_cast<int>(levels_2D) - 1}; level >= 0; --level)
    {
        key = (key << 3) + ((key_2d >> (2 * level)) & 3);
    }
    key_2d = key;
    key    = 0;
    for (int level = maxTreeLevel<KeyType>{} - levels_2D - 1; level >= 0; --level)
    {
        unsigned xi = (px >> level) & 1u;
        unsigned yi = (py >> level) & 1u;
        unsigned zi = (pz >> level) & 1u;

        // append 3 bits to the key
        unsigned octant = (xi << 2) | (yi << 1) | zi;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        key = (key << 3) + mortonToHilbertDevice[octant];
#else
        key = (key << 3) + mortonToHilbert[octant];
#endif

        // turn px, py and pz
        px ^= -(xi & ((!yi) | zi));
        py ^= -((xi & (yi | zi)) | (yi & (!zi)));
        pz ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

        if (zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = py;
            py          = pz;
            pz          = pt;
        }
        else if (!yi)
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }
    }
    const auto key_2d_shifted = key_2d << (3 * (maxTreeLevel<KeyType>{} - levels_2D));
    key                       = key_2d_shifted | key;
    return key;
}

/*! @brief compute the Hilbert key for a 2D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px,py  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */

template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert2D(unsigned px, unsigned py, int order) noexcept
{
    assert(px < (1u << maxTreeLevel<KeyType>{}));
    assert(py < (1u << maxTreeLevel<KeyType>{}));

    unsigned xi, yi;
    unsigned temp;
    KeyType key = 0;

    for (int level = order - 1; level >= 0; level--)
    {
        xi = (px >> level) & 1u; // Get bit level of x.
        yi = (py >> level) & 1u; // Get bit level of y.

        if (yi == 0)
        {
            temp = px;           // Swap x and y and,
            px   = py ^ (-xi);   // if xi = 1,
            py   = temp ^ (-xi); // complement them.
        }
        key = 4 * key + 2 * xi + (xi ^ yi); // Append two bits to key.
    }
    return key;
}

//! @brief inverse function of iHilbert
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned, unsigned>
decodeHilbert(KeyType key, unsigned order = maxTreeLevel<KeyType>{}) noexcept
{
    unsigned px = 0;
    unsigned py = 0;
    unsigned pz = 0;

    for (unsigned level = 0; level < order; ++level)
    {
        unsigned octant   = (key >> (3 * level)) & 7u;
        const unsigned xi = octant >> 2u;
        const unsigned yi = (octant >> 1u) & 1u;
        const unsigned zi = octant & 1u;

        if (yi ^ zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = pz;
            pz          = py;
            py          = pt;
        }
        else if ((!xi & !yi & !zi) || (xi & yi & zi))
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }

        // turn px, py and pz
        unsigned mask = (1 << level) - 1;
        px ^= mask & (-(xi & (yi | zi)));
        py ^= mask & (-((xi & ((!yi) | (!zi))) | ((!xi) & yi & zi)));
        pz ^= mask & (-((xi & (!yi) & (!zi)) | (yi & zi)));

        // append 1 bit to the positions
        px |= (xi << level);
        py |= ((xi ^ yi) << level);
        pz |= ((yi ^ zi) << level);
    }

    return {px, py, pz};
}

//! @brief inverse function of iHilbert
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned, unsigned>
decodeHilbert1DMixed(KeyType key, int levels_1D, axis long_dimension) noexcept
{
    unsigned px = 0;
    unsigned py = 0;
    unsigned pz = 0;

    for (unsigned level = 0; level < maxTreeLevel<KeyType>{} - levels_1D; ++level)
    {
        unsigned octant   = (key >> (3 * level)) & 7u;
        const unsigned xi = octant >> 2u;
        const unsigned yi = (octant >> 1u) & 1u;
        const unsigned zi = octant & 1u;

        if (yi ^ zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = pz;
            pz          = py;
            py          = pt;
        }
        else if ((!xi & !yi & !zi) || (xi & yi & zi))
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }

        // turn px, py and pz
        unsigned mask = (1 << level) - 1;
        px ^= mask & (-(xi & (yi | zi)));
        py ^= mask & (-((xi & ((!yi) | (!zi))) | ((!xi) & yi & zi)));
        pz ^= mask & (-((xi & (!yi) & (!zi)) | (yi & zi)));

        // append 1 bit to the positions
        px |= (xi << level);
        py |= ((xi ^ yi) << level);
        pz |= ((yi ^ zi) << level);
    }

    unsigned masked_1D_key{};

    for (int level{levels_1D - 1}; level >= 0; --level)
    {
        masked_1D_key = (masked_1D_key << 2) | ((key >> (3 * (maxTreeLevel<KeyType>{} - levels_1D + level))) & 3);
    }

    masked_1D_key = masked_1D_key << (maxTreeLevel<KeyType>{} - levels_1D);

    if (long_dimension == axis::x) { px = px | masked_1D_key; }
    else if (long_dimension == axis::y) { py = py | masked_1D_key; }
    else { pz = pz | masked_1D_key; }

    return {px, py, pz};
}

template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned>
decodeHilbert2D(KeyType key, unsigned order = maxTreeLevel<KeyType>{}) noexcept;

//! @brief inverse function of iHilbert
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned, unsigned>
decodeHilbert2DMixed(KeyType key, int levels_2D, axis short_dimension) noexcept
{
    unsigned px = 0;
    unsigned py = 0;
    unsigned pz = 0;

    for (unsigned level = 0; level < maxTreeLevel<KeyType>{} - levels_2D; ++level)
    {
        unsigned octant   = (key >> (3 * level)) & 7u;
        const unsigned xi = octant >> 2u;
        const unsigned yi = (octant >> 1u) & 1u;
        const unsigned zi = octant & 1u;

        if (yi ^ zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = pz;
            pz          = py;
            py          = pt;
        }
        else if ((!xi & !yi & !zi) || (xi & yi & zi))
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }

        // turn px, py and pz
        unsigned mask = (1 << level) - 1;
        px ^= mask & (-(xi & (yi | zi)));
        py ^= mask & (-((xi & ((!yi) | (!zi))) | ((!xi) & yi & zi)));
        pz ^= mask & (-((xi & (!yi) & (!zi)) | (yi & zi)));

        // append 1 bit to the positions
        px |= (xi << level);
        py |= ((xi ^ yi) << level);
        pz |= ((yi ^ zi) << level);
    }

    unsigned masked_2D_key{};

    for (int level{levels_2D - 1}; level >= 0; --level)
    {
        masked_2D_key = (masked_2D_key << 2) | ((key >> (3 * (maxTreeLevel<KeyType>{} - levels_2D + level))) & 3);
    }

    auto xy_2d     = decodeHilbert2D<KeyType>(masked_2D_key);
    unsigned px_2D = get<0>(xy_2d);
    unsigned py_2D = get<1>(xy_2d);

    if (short_dimension == axis::x)
    {
        py = py | (px_2D << (maxTreeLevel<KeyType>{} - levels_2D));
        pz = pz | (py_2D << (maxTreeLevel<KeyType>{} - levels_2D));
    }
    else if (short_dimension == axis::y)
    {
        px = px | (px_2D << (maxTreeLevel<KeyType>{} - levels_2D));
        pz = pz | (py_2D << (maxTreeLevel<KeyType>{} - levels_2D));
    }
    else
    {
        px = px | (px_2D << (maxTreeLevel<KeyType>{} - levels_2D));
        py = py | (py_2D << (maxTreeLevel<KeyType>{} - levels_2D));
    }

    return {px, py, pz};
}

// Lam and Shapiro inverse function of hilbert
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned> decodeHilbert2D(KeyType key, unsigned order) noexcept
{
    unsigned sa, sb;
    unsigned x = 0, y = 0, temp = 0;

    for (unsigned level = 0; level < 2 * order; level += 2)
    {
        // Get bit level+1 of key.
        sa = (key >> (level + 1)) & 1;
        // Get bit level of key.
        sb = (key >> level) & 1;
        if ((sa ^ sb) == 0)
        {
            // If sa,sb = 00 or 11,
            temp = x;
            // swap x and y,
            x = y ^ (-sa);
            // and if sa = 1,
            y = temp ^ (-sa);
            // complement them.
        }
        x = (x >> 1) | (sa << 31);        // Prepend sa to x and
        y = (y >> 1) | ((sa ^ sb) << 31); // (sa ^ sb) to y.
    }
    unsigned px = x >> (32 - order);
    // Right-adjust x and y
    unsigned py = y >> (32 - order);
    // and return them to
    return {px, py};
}

//! @brief inverse function of iHilbert 32 bit only up to oder 16 but works at constant time.
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned> decodeHilbert2DConstant(KeyType key) noexcept
{
    unsigned order = maxTreeLevel<KeyType>{};

    key = key | (0x55555555 << 2 * order); // Pad key on left with 01

    const unsigned sr = (key >> 1) & 0x55555555;                // (no change) groups.
    unsigned cs       = ((key & 0x55555555) + sr) ^ 0x55555555; // Compute complement & swap info in two-bit groups.
    // Parallel prefix xor op to propagate both complement
    // and swap info together from left to right (there is
    // no step "cs ^= cs >> 1", so in effect it computes
    // two independent parallel prefix operations on two
    // interleaved sets of sixteen bits).
    cs                  = cs ^ (cs >> 2);
    cs                  = cs ^ (cs >> 4);
    cs                  = cs ^ (cs >> 8);
    cs                  = cs ^ (cs >> 16);
    const unsigned swap = cs & 0x55555555;        // Separate the swap and
    const unsigned comp = (cs >> 1) & 0x55555555; // complement bits.

    unsigned t = (key & swap) ^ comp;          // Calculate x and y in
    key        = key ^ sr ^ t ^ (t << 1);      // the odd & even bit positions, resp.
    key        = key & ((1 << 2 * order) - 1); // Clear out any junk on the left (unpad).

    // Now "unshuffle" to separate the x and y bits.

    t   = (key ^ (key >> 1)) & 0x22222222;
    key = key ^ t ^ (t << 1);
    t   = (key ^ (key >> 2)) & 0x0C0C0C0C;
    key = key ^ t ^ (t << 2);
    t   = (key ^ (key >> 4)) & 0x00F000F0;
    key = key ^ t ^ (t << 4);
    t   = (key ^ (key >> 8)) & 0x0000FF00;
    key = key ^ t ^ (t << 8);

    unsigned px = key >> 16;    // Assign the two halves
    unsigned py = key & 0xFFFF; // of t to x and y.

    return {px, py};
}

/*! @brief compute the 3D integer coordinate box that contains the key range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param  keyStart  lower Hilbert key
 * @param  keyEnd    upper Hilbert key
 * @return           the integer box that contains the given key range
 */
template<class KeyType>
HOST_DEVICE_FUN IBox hilbertIBox(KeyType keyStart, unsigned level) noexcept
{
    assert(level <= maxTreeLevel<KeyType>{});
    constexpr unsigned maxCoord = 1u << maxTreeLevel<KeyType>{};
    unsigned cubeLength         = maxCoord >> level;
    unsigned mask               = ~(cubeLength - 1);

    auto [ix, iy, iz] = decodeHilbert(keyStart);

    // round integer coordinates down to corner closest to origin
    ix &= mask;
    iy &= mask;
    iz &= mask;

    return IBox(ix, ix + cubeLength, iy, iy + cubeLength, iz, iz + cubeLength);
}

//! @brief convenience wrapper
template<class KeyType>
HOST_DEVICE_FUN IBox hilbertIBoxKeys(KeyType keyStart, KeyType keyEnd) noexcept
{
    assert(keyStart <= keyEnd);
    return hilbertIBox(keyStart, treeLevel(keyEnd - keyStart));
}

} // namespace cstone
