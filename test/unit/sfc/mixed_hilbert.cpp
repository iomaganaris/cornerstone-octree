#include <gtest/gtest.h>

#include "coord_samples/random.hpp"

using namespace cstone;

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

//! @brief test the curve on the first eight octants
template<class KeyType>
void firstOrderCurve1D3D()
{
    constexpr unsigned hilbertToMorton[8] = {0, 1, 3, 2, 6, 7, 5, 4};

    for (unsigned xi = 0; xi < 2; ++xi)
    {
        for (unsigned yi = 0; yi < 2; ++yi)
        {
            for (unsigned zi = 0; zi < 2; ++zi)
            {
                unsigned L1RangeLong  = (1 << maxTreeLevel<KeyType>{}) / 2;
                unsigned L1RangeShort = (1 << (maxTreeLevel<KeyType>{} - 2)) / 2;
                unsigned mortonOctant = 4 * xi + 2 * yi + zi;

                {
                    KeyType hilbertKey =
                        iHilbert1DMixed<KeyType>(L1RangeLong * xi, L1RangeShort * yi, L1RangeShort * zi, 2, axis::x);
                    unsigned hilbertOctant = octalDigit(hilbertKey, 1);
                    EXPECT_EQ(mortonOctant, hilbertToMorton[hilbertOctant]);
                }
                {
                    KeyType hilbertKey     = iHilbert1DMixed<KeyType>(L1RangeLong * xi + L1RangeLong - 1,
                                                                  L1RangeShort * yi + L1RangeShort - 1,
                                                                  L1RangeShort * zi + L1RangeShort - 1, 2, axis::x);
                    unsigned hilbertOctant = octalDigit(hilbertKey, 1);
                    EXPECT_EQ(mortonOctant, hilbertToMorton[hilbertOctant]);
                }
            }
        }
    }
}

TEST(MixedHilbertFirstOrderCurve, 1D3D)
{
    firstOrderCurve1D3D<unsigned>();
    firstOrderCurve1D3D<uint64_t>();
}
