/**
 * Distributed under the terms of the GNU General Public License v3.0.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "foamalgo/miscellaneous.hpp"

namespace py = pybind11;


PYBIND11_MODULE(miscellaneous, m) {
  m.doc() = "Miscellaneous functions in cpp";

  m.def("intersection", &foam::intersection);
}