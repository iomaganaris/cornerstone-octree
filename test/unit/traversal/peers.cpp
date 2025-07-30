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
 * @brief Test functions used to find peer ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/peers.hpp"
#include "cstone/tree/cs_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

//! @brief reference peer search, all-all leaf comparison
template<class KeyType, class T>
static std::vector<int> findPeersAll2All(int myRank,
                                         const SfcAssignment<KeyType>& assignment,
                                         gsl::span<const KeyType> tree,
                                         const Box<T>& box,
                                         float invThetaEff,
                                         const bool disableMixD = false)
{
    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    const bool mixD = !disableMixD && (mixDBits.bx != maxTreeLevel<KeyType>{} ||
        mixDBits.by != maxTreeLevel<KeyType>{} ||
        mixDBits.bz != maxTreeLevel<KeyType>{});

    TreeNodeIndex firstIdx = findNodeAbove(tree.data(), nNodes(tree), assignment[myRank]);
    TreeNodeIndex lastIdx  = findNodeAbove(tree.data(), nNodes(tree), assignment[myRank + 1]);
    // std::cout << "myRank: " << myRank << ", firstIdx: " << firstIdx << ", lastIdx: " << lastIdx << std::endl;

    std::vector<Vec3<T>> boxCenter(nNodes(tree));
    std::vector<Vec3<T>> boxSize(nNodes(tree));
    for (TreeNodeIndex i = 0; i < TreeNodeIndex(nNodes(tree)); ++i)
    {
        IBox ibox                          = mixD ? sfcIBox(sfcMixDKey(tree[i]), sfcMixDKey(tree[i + 1]), mixDBits.bx, mixDBits.by, mixDBits.bz) : sfcIBox(sfcKey(tree[i]), sfcKey(tree[i + 1]));
        std::tie(boxCenter[i], boxSize[i]) = centerAndSize<KeyType>(ibox, box, disableMixD);
        // std::cout << "boxCenter[" << i << "]: " << boxCenter[i][0] << " " << boxCenter[i][1] << " " << boxCenter[i][2]
        //           << " boxSize: " << boxSize[i][0] << " " << boxSize[i][1] << " " << boxSize[i][2] << std::endl;
    }

    std::vector<int> peers(assignment.numRanks());
    for (TreeNodeIndex i = firstIdx; i < lastIdx; ++i) {
        if (mixD && (boxSize[i][0] == 0 && boxSize[i][1] == 0 && boxSize[i][2] == 0)) {
            continue; // skip empty boxes
        }
        for (TreeNodeIndex j = 0; j < TreeNodeIndex(nNodes(tree)); ++j) {
            // std::cout << "Checking box " << i << " against " << j << std::endl;
            if (mixD && (boxSize[j][0] == 0 && boxSize[j][1] == 0 && boxSize[j][2] == 0)) {
                // std::cout << "Skipping empty box " << j << std::endl;
                continue; // skip empty boxes
            }
            // std::cout << "boxCenter[i]: " << boxCenter[i][0] << " " << boxCenter[i][1] << " " << boxCenter[i][2]
            //           << ", boxSize[i]: " << boxSize[i][0] << " " << boxSize[i][1] << " " << boxSize[i][2] << std::endl;
            // std::cout << "boxCenter[j]: " << boxCenter[j][0] << " " << boxCenter[j][1] << " " << boxCenter[j][2]
            //           << ", boxSize[j]: " << boxSize[j][0] << " " << boxSize[j][1] << " " << boxSize[j][2] << std::endl;
            // std::cout << "minVecMacMutual: " << minVecMacMutual(boxCenter[i], boxSize[i], boxCenter[j], boxSize[j], box, invThetaEff) << std::endl;
            if (!minVecMacMutual(boxCenter[i], boxSize[i], boxCenter[j], boxSize[j], box, invThetaEff)) {
                peers[assignment.findRank(tree[j])] = 1;
            }
        }
    }

    std::vector<int> ret;
    for (int i = 0; i < int(peers.size()); ++i)
        if (peers[i] && i != myRank) { ret.push_back(i); }

    return ret;
}

template<class KeyType>
static void findMacPeers64grid(int rank, Box<double> box, float theta, int /*refNumPeers*/)
{
    Octree<KeyType> octree;
    auto leaves = makeUniformNLevelTree<KeyType>(64, 1);
    octree.update(leaves.data(), nNodes(leaves));

    SfcAssignment<KeyType> assignment(octree.numLeafNodes());
    for (int i = 0; i < octree.numLeafNodes() + 1; ++i)
    {
        assignment.set(i, leaves[i], 1);
    }

    std::vector<int> peers     = findPeersMac(rank, assignment, octree, box, invThetaVecMac(theta));
    std::vector<int> reference = findPeersAll2All(rank, assignment, octree.treeLeaves(), box, invThetaVecMac(theta));

    // EXPECT_EQ(refNumPeers, peers.size());
    EXPECT_EQ(peers, reference);
}

TEST(Peers, findMacGrid64)
{
    // just the surface
    findMacPeers64grid<unsigned>(0, Box<double>{-1, 1, BoundaryType::open}, 1.1, 7);
    findMacPeers64grid<uint64_t>(0, Box<double>{-1, 1, BoundaryType::open}, 1.1, 7);
    findMacPeers64grid<unsigned>(0, Box<double>{0, 1, 0, 0.015625, 0, 0.00390625, BoundaryType::open}, 1.1, 7);
    findMacPeers64grid<uint64_t>(0, Box<double>{0, 1, 0, 0.015625, 0, 0.00390625, BoundaryType::open}, 1.1, 7);
}

TEST(Peers, findMacGrid64Narrow)
{
    findMacPeers64grid<unsigned>(0, Box<double>{-1, 1, BoundaryType::open}, 1.0, 19);
    findMacPeers64grid<uint64_t>(0, Box<double>{-1, 1, BoundaryType::open}, 1.0, 19);
    findMacPeers64grid<unsigned>(0, Box<double>{0, 1, 0, 0.015625, 0, 0.00390625, BoundaryType::open}, 1.0, 19);
    findMacPeers64grid<uint64_t>(0, Box<double>{0, 1, 0, 0.015625, 0, 0.00390625, BoundaryType::open}, 1.0, 19);
}

TEST(Peers, findMacGrid64PBC)
{
    // just the surface + PBC, 26 six peers at the surface
    findMacPeers64grid<unsigned>(0, Box<double>{-1, 1, BoundaryType::periodic}, 1.1, 26);
    findMacPeers64grid<uint64_t>(0, Box<double>{-1, 1, BoundaryType::periodic}, 1.1, 26);
    findMacPeers64grid<unsigned>(0, Box<double>{0, 1, 0, 0.015625, 0, 0.00390625, BoundaryType::periodic}, 1.1, 26);
    findMacPeers64grid<uint64_t>(0, Box<double>{0, 1, 0, 0.015625, 0, 0.00390625, BoundaryType::periodic}, 1.1, 26);
}

template<class KeyType>
static void findPeers(Box<double> box, const bool disableMixD = false)
{
    int nParticles    = 100000;
    int bucketSize    = 64;
    int numRanks      = 50;
    float invThetaEff = invThetaVecMac(0.5f);

    const auto mixDBits = getBoxMixDimensionBits<double, KeyType, Box<double>>(box);
    const bool useMixD = !disableMixD && (mixDBits.bx != maxTreeLevel<KeyType>{} ||
                          mixDBits.by != maxTreeLevel<KeyType>{} ||
                          mixDBits.bz != maxTreeLevel<KeyType>{});

    auto particleKeys   = useMixD ? RandomCoordinates<double, SfcMixDKind<KeyType>>(nParticles, box, 42, mixDBits.bx, mixDBits.by, mixDBits.bz).particleKeys() : makeRandomGaussianKeys<KeyType>(nParticles);
    auto [tree, counts] = computeOctree(particleKeys.data(), particleKeys.data() + nParticles, bucketSize);

    Octree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    int probeRank             = numRanks / 2;
    std::vector<int> peersDtt = findPeersMac(probeRank, assignment, octree, box, invThetaEff);
    std::vector<int> peersStt = findPeersMacStt(probeRank, assignment, octree, box, invThetaEff);
    std::vector<int> peersA2A = findPeersAll2All<KeyType>(probeRank, assignment, tree, box, invThetaEff);
    EXPECT_EQ(peersDtt, peersStt);
    EXPECT_EQ(peersDtt, peersA2A);

    // check for mutuality
    for (int peerRank : peersDtt)
    {
        std::vector<int> peersOfPeerDtt = findPeersMac(peerRank, assignment, octree, box, invThetaEff);

        // std::vector<int> peersOfPeerStt = findPeersMacStt(peerRank, assignment, octree, box, invThetaEff);
        // EXPECT_EQ(peersDtt, peersStt);
        std::vector<int> peersOfPeerA2A = findPeersAll2All<KeyType>(peerRank, assignment, tree, box, invThetaEff);
        EXPECT_EQ(peersOfPeerDtt, peersOfPeerA2A);

        // the peers of the peers of the probeRank have to have probeRank as peer
        EXPECT_TRUE(std::find(begin(peersOfPeerDtt), end(peersOfPeerDtt), probeRank) != end(peersOfPeerDtt));
    }
}

TEST(Peers, find)
{
    findPeers<unsigned>(Box<double>{-1, 1});
    findPeers<uint64_t>(Box<double>{-1, 1});
    findPeers<unsigned>(Box<double>{0, 1, 0, 0.015625, 0, 0.00390625});
    findPeers<uint64_t>(Box<double>{0, 1, 0, 0.015625, 0, 0.00390625});
}
