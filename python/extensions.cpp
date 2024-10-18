#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "cstone/sfc/common.hpp"
#include "cstone/sfc/hilbert.hpp"

constexpr unsigned iHilbert_wrapper(int px, int py, int pz, int order = cstone::maxTreeLevel<unsigned>{}) noexcept
{
    return cstone::iHilbert<unsigned>(static_cast<unsigned>(px), static_cast<unsigned>(py), static_cast<unsigned>(pz),
                                      static_cast<unsigned>(order));
};

constexpr util::tuple<int, int, int> decodeHilbert_wrapper(int key,
                                                           int order = cstone::maxTreeLevel<unsigned>{}) noexcept
{
    return cstone::decodeHilbert<unsigned>(static_cast<unsigned>(key), static_cast<unsigned>(order));
};

constexpr unsigned iHilbertMixD_wrapper(int px, int py, int pz, int bx, int by, int bz) noexcept
{
    return cstone::iHilbertMixD<unsigned>(static_cast<unsigned>(px), static_cast<unsigned>(py),
                                          static_cast<unsigned>(pz), static_cast<unsigned>(bx),
                                          static_cast<unsigned>(by), static_cast<unsigned>(bz));
}

constexpr util::tuple<int, int, int> decodeHilbertMixD_wrapper(int key, int bx, int by, int bz) noexcept
{
    return cstone::decodeHilbertMixD<unsigned>(static_cast<unsigned>(key), static_cast<unsigned>(bx),
                                               static_cast<unsigned>(by), static_cast<unsigned>(bz));
};

std::pair<unsigned, std::vector<unsigned>> spanSfcRange(int x, int y, int output_size)
{
    auto output  = std::make_pair<unsigned, std::vector<unsigned>>(0, std::vector<unsigned>(output_size));
    output.first = cstone::spanSfcRange(static_cast<unsigned>(x), static_cast<unsigned>(y), output.second.data());
    return output;
}

cstone::IBox hilbertIBoxKeys_wrapper(int keyStart, unsigned keyEnd) noexcept
{
    return cstone::hilbertIBoxKeys<unsigned>(static_cast<unsigned>(keyStart), static_cast<unsigned>(keyEnd));
}

cstone::IBox hilbertMixDIBoxKeys_wrapper(int keyStart, int keyEnd, int bx, int by, int bz) noexcept
{
    return cstone::hilbertMixDIBoxKeys<unsigned>(static_cast<unsigned>(keyStart), static_cast<unsigned>(keyEnd),
                                                 static_cast<unsigned>(bx), static_cast<unsigned>(by),
                                                 static_cast<unsigned>(bz));
}

NB_MODULE(cornerstone, m)
{
    nanobind::class_<cstone::IBox>(m, "IBox")
        .def("xmin", &cstone::IBox::xmin)
        .def("xmax", &cstone::IBox::xmax)
        .def("ymin", &cstone::IBox::ymin)
        .def("ymax", &cstone::IBox::ymax)
        .def("zmin", &cstone::IBox::zmin)
        .def("zmax", &cstone::IBox::zmax);
    m.def("iHilbert", &iHilbert_wrapper);
    m.def("decodeHilbert", &decodeHilbert_wrapper);
    m.def("iHilbertMixD", &iHilbertMixD_wrapper);
    m.def("decodeHilbertMixD", &decodeHilbertMixD_wrapper);
    m.def("spanSfcRange", &spanSfcRange);
    m.def("hilbertIBoxKeys", &hilbertIBoxKeys_wrapper);
    m.def("hilbertMixDIBoxKeys", &hilbertMixDIBoxKeys_wrapper);
}
