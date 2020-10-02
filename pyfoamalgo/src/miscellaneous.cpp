/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file BSD_LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "foamalgo/miscellaneous.hpp"

namespace py = pybind11;


PYBIND11_MODULE(miscellaneous, m) {
  m.doc() = "Miscellaneous functions in cpp";

  m.def("intersection", &foam::intersection);
}