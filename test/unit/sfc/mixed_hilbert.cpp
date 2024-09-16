#include <gtest/gtest.h>

#include "coord_samples/random.hpp"

using namespace cstone;

// TEST(MixedHilbert, hilbert3D)
// {
//     unsigned x = 46; // 0000101110
//     unsigned y = 28; // 0000011100
//     unsigned z = 54; // 0000110110
//     auto hilbertKey = iHilbert<unsigned>(x, y, z);
//     std::cout << "hilbertKey = " << std::bitset<32>(hilbertKey) << std::endl; // 00000000000000110010000001010010
//     auto [a, b, c] = decodeHilbert<unsigned>(hilbertKey);
//     EXPECT_EQ(x, a);
//     EXPECT_EQ(y, b);
//     EXPECT_EQ(z, c);
// }

TEST(MixedHilbertBox, x10y9z9)
{
    unsigned bx = 10, by = 9, bz = 9;
    int numKeys{10};
    std::mt19937 gen;

    std::uniform_int_distribution<unsigned> distribution_x_le_511(0, (1 << (bx - 1)) - 1); // 0 to 511
    std::uniform_int_distribution<unsigned> distribution_x_ge_512(512, (1 << bx) - 1);     // 512 to 1023
    std::uniform_int_distribution<unsigned> distribution_y(0, (1 << by) - 1);
    std::uniform_int_distribution<unsigned> distribution_z(0, (1 << bz) - 1);

    auto getRandXle511 = [&distribution_x_le_511, &gen]() { return distribution_x_le_511(gen); };
    auto getRandXge512 = [&distribution_x_ge_512, &gen]() { return distribution_x_ge_512(gen); };
    auto getRandY      = [&distribution_y, &gen]() { return distribution_y(gen); };
    auto getRandZ      = [&distribution_z, &gen]() { return distribution_z(gen); };

    std::vector<unsigned> x_le_511(numKeys);
    std::vector<unsigned> x_ge_512(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x_le_511), end(x_le_511), getRandXle511);
    std::generate(begin(x_ge_512), end(x_ge_512), getRandXge512);
    std::generate(begin(y), end(y), getRandY);
    std::generate(begin(z), end(z), getRandZ);

    for (int i = 0; i < numKeys; ++i)
    {
        std::cout << "x: " << std::bitset<10>(x_le_511[i]) << " (" << x_le_511[i] << ")" << std::endl;
        std::cout << "y: " << std::bitset<10>(y[i]) << " (" << y[i] << ")" << std::endl;
        std::cout << "z: " << std::bitset<10>(z[i]) << " (" << z[i] << ")" << std::endl;
        auto hilbertMixDKey = iHilbertMixD<unsigned>(x_le_511[i], y[i], z[i], bx, by, bz);
        std::cout << "hilbertMixDKey = " << std::bitset<32>(hilbertMixDKey) << std::endl;
        auto hilbertKey = iHilbert<unsigned>(x_le_511[i], y[i], z[i]);
        std::cout << "hilbertKey =     " << std::bitset<32>(hilbertKey) << std::endl;

        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        std::cout << "x: " << std::bitset<10>(x_ge_512[i]) << " (" << x_ge_512[i] << ")" << std::endl;
        std::cout << "y: " << std::bitset<10>(y[i]) << " (" << y[i] << ")" << std::endl;
        std::cout << "z: " << std::bitset<10>(z[i]) << " (" << z[i] << ")" << std::endl;
        auto hilbertMixDKey = iHilbertMixD<unsigned>(x_ge_512[i], y[i], z[i], bx, by, bz);
        std::cout << "hilbertMixDKey = " << std::bitset<32>(hilbertMixDKey) << std::endl;
        auto hilbertKey_px_m_512 = iHilbert<unsigned>(x_ge_512[i] - 512, y[i], z[i]);
        std::cout << "hilbertKey_px_m_512 =     " << std::bitset<32>(hilbertKey_px_m_512) << std::endl;

        EXPECT_EQ(hilbertMixDKey, (1 << 27) + hilbertKey_px_m_512);
    };

    // clang-format off
    // iHilbertMixD(px in [0:512], py, pz, bx, by, bz)    == iHilbert(px, py, pz) >> 3
    // iHilbertMixD(px in [512:1024], py, pz, bx, by, bz) == 01000000000 (=8^9) + (iHilbert3d(px - 512, py, pz) >> 3)
    // IM: I believe that iHilbert >> 3 above is incorrect because the LSBs of the 3D Hilbert key are !=0
    // clang-format on
}

TEST(MixedHilbertBox, x10y10z9)
{
    unsigned bx = 10, by = 10, bz = 9;
    int numKeys{10};
    std::mt19937 gen;

    std::uniform_int_distribution<unsigned> distribution_x_le_511(0, (1 << (bx - 1)) - 1); // 0 to 511
    std::uniform_int_distribution<unsigned> distribution_x_ge_512(512, (1 << bx) - 1);     // 512 to 1023
    std::uniform_int_distribution<unsigned> distribution_y_le_511(0, (1 << (by - 1)) - 1); // 0 to 511
    std::uniform_int_distribution<unsigned> distribution_y_ge_512(512, (1 << by) - 1);     // 512 to 1023
    std::uniform_int_distribution<unsigned> distribution_z(0, (1 << bz) - 1);

    auto getRandXle511 = [&distribution_x_le_511, &gen]() { return distribution_x_le_511(gen); };
    auto getRandXge512 = [&distribution_x_ge_512, &gen]() { return distribution_x_ge_512(gen); };
    auto getRandYle511 = [&distribution_y_le_511, &gen]() { return distribution_y_le_511(gen); };
    auto getRandYge512 = [&distribution_y_ge_512, &gen]() { return distribution_y_ge_512(gen); };
    auto getRandZ      = [&distribution_z, &gen]() { return distribution_z(gen); };

    std::vector<unsigned> x_le_511(numKeys);
    std::vector<unsigned> x_ge_512(numKeys);
    std::vector<unsigned> y_le_511(numKeys);
    std::vector<unsigned> y_ge_512(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x_le_511), end(x_le_511), getRandXle511);
    std::generate(begin(x_ge_512), end(x_ge_512), getRandXge512);
    std::generate(begin(y_le_511), end(y_le_511), getRandYle511);
    std::generate(begin(y_ge_512), end(y_ge_512), getRandYge512);
    std::generate(begin(z), end(z), getRandZ);

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey = iHilbertMixD<unsigned>(x_le_511[i], y_le_511[i], z[i], bx, by, bz);
        auto hilbertKey     = iHilbert<unsigned>(x_le_511[i], y_le_511[i], z[i]);

        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbertMixD<unsigned>(x_le_511[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert<unsigned>(x_le_511[i], y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (1 << 27) + hilbertKey_py_m_512);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbertMixD<unsigned>(x_ge_512[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert<unsigned>(x_ge_512[i] - 512, y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (2 << 27) + hilbertKey_py_m_512);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbertMixD<unsigned>(x_ge_512[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert<unsigned>(x_ge_512[i] - 512, y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (2 << 27) + hilbertKey_py_m_512);
    };

    // clang-format off
    // iHilbertMixD(px in [0:512], py in [0:512], pz, bx, by, bz)       == iHilbert(px, py, pz) >> 3
    // iHilbertMixD(px in [0:512], py in [512:1024], pz, bx, by, bz)    == 01000000000 (=8^9) + (iHilbert(px, py - 512, pz) >> 3)
    // iHilbertMixD(px in [512:1024], py in [512:1024], pz, bx, by, bz) == 02000000000 (=8^9) + (iHilbert(px - 512, py - 512, pz) >> 3)
    // IM: Can't understand below
    // iHilbertMixD(px in [0:512], py in [512:1024], pz, bx, by, bz)    == 03000000000 (=8^9) + (iHilbert(px, py - 512, pz) >> 3)
    // clang-format on
}

TEST(MixedHilbertBox, Long1DDomain)
{
    using real        = double;
    using IntegerType = unsigned;
    int n             = 10;

    Box<real> box{0, 10000000, 0, 50, 0, 50};
    RandomCoordinates<real, Sfc1DMixedKind<IntegerType>> c(n, box);

    unsigned levels_1D = 2;

    std::vector<IntegerType> testCodes(n);
    computeSfc1D3DKeys(c.x().data(), c.y().data(), c.z().data(), Sfc1DMixedKindPointer(testCodes.data()), n, box,
                       levels_1D);

    EXPECT_TRUE(std::is_sorted(testCodes.begin(), testCodes.end()));
}

TEST(MixedHilbertBox, Short1DDomain)
{
    using real        = double;
    using IntegerType = unsigned;
    int n             = 10;

    Box<real> box{0, 0.1, -10000, 100000, -10000, 10000};
    RandomCoordinates<real, Sfc2DMixedKind<IntegerType>> c(n, box);

    unsigned levels_2D = 2;

    std::vector<IntegerType> testCodes(n);
    computeSfc2D3DKeys(c.x().data(), c.y().data(), c.z().data(), Sfc2DMixedKindPointer(testCodes.data()), n, box,
                       levels_2D);

    EXPECT_TRUE(std::is_sorted(testCodes.begin(), testCodes.end()));
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest1D3D()
{
    int numKeys                           = 10;
    int maxCoordLong                      = (1 << maxTreeLevel<KeyType>{}) - 1;
    std::vector<unsigned> levels_1D_sweep = {2, 4};
    for (const auto levels_1D : levels_1D_sweep)
    {
        int maxCoordShort = (1 << (maxTreeLevel<KeyType>{} - levels_1D)) - 1;

        std::mt19937 gen;
        std::uniform_int_distribution<unsigned> distribution_long(0, maxCoordLong);
        std::uniform_int_distribution<unsigned> distribution_short(0, maxCoordShort);

        auto getRandLong  = [&distribution_long, &gen]() { return distribution_long(gen); };
        auto getRandShort = [&distribution_short, &gen]() { return distribution_short(gen); };

        std::vector<unsigned> x(numKeys);
        std::vector<unsigned> y(numKeys);
        std::vector<unsigned> z(numKeys);

        std::vector<axis> axes{axis::x, axis::y, axis::z};
        for (const auto long_axis : axes)
        {
            if (long_axis == axis::x)
            {
                std::generate(begin(x), end(x), getRandLong);
                std::generate(begin(y), end(y), getRandShort);
                std::generate(begin(z), end(z), getRandShort);
            }
            else if (long_axis == axis::y)
            {
                std::generate(begin(x), end(x), getRandShort);
                std::generate(begin(y), end(y), getRandLong);
                std::generate(begin(z), end(z), getRandShort);
            }
            else
            {
                std::generate(begin(x), end(x), getRandShort);
                std::generate(begin(y), end(y), getRandShort);
                std::generate(begin(z), end(z), getRandLong);
            }

            for (int i = 0; i < numKeys; ++i)
            {
                KeyType hilbertKey = iHilbert1DMixed<KeyType>(x[i], y[i], z[i], levels_1D, long_axis);

                auto [a, b, c] = decodeHilbert1DMixed(hilbertKey, levels_1D, long_axis);
                EXPECT_EQ(x[i], a);
                EXPECT_EQ(y[i], b);
                EXPECT_EQ(z[i], c);
            }
        }
    }
}

TEST(MixedHilbertEncoding, InversionTest1D3D)
{
    inversionTest1D3D<unsigned>();
    inversionTest1D3D<uint64_t>();
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest2D3D()
{
    int numKeys                           = 10;
    int maxCoordLong                      = (1 << maxTreeLevel<KeyType>{}) - 1;
    std::vector<unsigned> levels_2D_sweep = {2, 3};
    for (const auto levels_2D : levels_2D_sweep)
    {
        int maxCoordShort = (1 << (maxTreeLevel<KeyType>{} - levels_2D)) - 1;

        std::mt19937 gen;
        std::uniform_int_distribution<unsigned> distribution_long(0, maxCoordLong);
        std::uniform_int_distribution<unsigned> distribution_short(0, maxCoordShort);

        auto getRandLong  = [&distribution_long, &gen]() { return distribution_long(gen); };
        auto getRandShort = [&distribution_short, &gen]() { return distribution_short(gen); };

        std::vector<unsigned> x(numKeys);
        std::vector<unsigned> y(numKeys);
        std::vector<unsigned> z(numKeys);

        std::vector<axis> axes{axis::x, axis::y, axis::z};
        for (const auto short_axis : axes)
        {
            if (short_axis == axis::x)
            {
                std::generate(begin(x), end(x), getRandShort);
                std::generate(begin(y), end(y), getRandLong);
                std::generate(begin(z), end(z), getRandLong);
            }
            else if (short_axis == axis::y)
            {
                std::generate(begin(x), end(x), getRandLong);
                std::generate(begin(y), end(y), getRandShort);
                std::generate(begin(z), end(z), getRandLong);
            }
            else
            {
                std::generate(begin(x), end(x), getRandLong);
                std::generate(begin(y), end(y), getRandLong);
                std::generate(begin(z), end(z), getRandShort);
            }

            for (int i = 0; i < numKeys; ++i)
            {
                KeyType hilbertKey = iHilbert2DMixed<KeyType>(x[i], y[i], z[i], levels_2D, short_axis);

                auto [a, b, c] = decodeHilbert2DMixed(hilbertKey, levels_2D, short_axis);
                EXPECT_EQ(x[i], a);
                EXPECT_EQ(y[i], b);
                EXPECT_EQ(z[i], c);
            }
        }
    }
}

TEST(MixedHilbertEncoding, InversionTest2D3D)
{
    inversionTest2D3D<unsigned>();
    inversionTest2D3D<uint64_t>();
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTestMixD()
{
    int numKeys{10};
    std::vector<std::vector<unsigned>> n_encoding_bits_sweep = {{8, 6, 10}, {10, 9, 9}, {10, 10, 10}, {10, 10, 9}};
    std::mt19937 gen;
    for (const auto& n_encoding_bits : n_encoding_bits_sweep)
    {
        std::uniform_int_distribution<unsigned> distribution_x(0, (1 << n_encoding_bits[0]) - 1);
        std::uniform_int_distribution<unsigned> distribution_y(0, (1 << n_encoding_bits[1]) - 1);
        std::uniform_int_distribution<unsigned> distribution_z(0, (1 << n_encoding_bits[2]) - 1);

        auto getRandX = [&distribution_x, &gen]() { return distribution_x(gen); };
        auto getRandY = [&distribution_y, &gen]() { return distribution_y(gen); };
        auto getRandZ = [&distribution_z, &gen]() { return distribution_z(gen); };

        std::vector<unsigned> x(numKeys);
        std::vector<unsigned> y(numKeys);
        std::vector<unsigned> z(numKeys);

        std::generate(begin(x), end(x), getRandX);
        std::generate(begin(y), end(y), getRandY);
        std::generate(begin(z), end(z), getRandZ);

        for (int i = 0; i < numKeys; ++i)
        {
            KeyType hilbertKey =
                iHilbertMixD<KeyType>(x[i], y[i], z[i], n_encoding_bits[0], n_encoding_bits[1], n_encoding_bits[2]);

            auto [a, b, c] = decodeHilbertMixD(hilbertKey, n_encoding_bits[0], n_encoding_bits[1], n_encoding_bits[2]);
            EXPECT_EQ(x[i], a);
            EXPECT_EQ(y[i], b);
            EXPECT_EQ(z[i], c);
        };
    }
}

TEST(MixedHilbertEncoding, InversionTestMixD)
{
    inversionTestMixD<unsigned>();
    inversionTestMixD<uint64_t>();
}
