"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) 2020, Jun Zhu. All rights reserved.
"""
import functools

import numpy as np


def stack_detector_modules(train_data, device, ppt, *,
                           modules=None,
                           module_numbers=None,
                           memory_cell_last=False):
    """Stack detector modules.

    :param dict train_data: data from a pulse-train.
    :param str device: device or output channel name with a '*' replacing
        the module number. For example, SCS_DET_DSSC1M-1/DET/*CH0:xtdf.
    :param str ppt: property name.
    :param int modules: Number of modules. Ignored if 'module_numbers' is
        given. For AGIPD-1M, LPD-1M and DSSC-1M, number of modules must be
        16 so it is not recommended to use 'module_numbers' to specify
        modules. Note that the default module number starts from 0, i.e.,
        the default module numbers for 4 modules are 0, 1, 2 and 3.
    :param list module_numbers: A list of module numbers. The numbers
        do not need to be continuous or be monotonically increasing. One
        should use it for detectors like JungFrau, ePix100, etc. They have
        a different naming convention from 1M detectors and the module
        number starts from 1.
    :param bool memory_cell_last: Whether memory cell is the last dimension.

    :return: A limited array-like wrapper around the modules data. It is
        sufficient for assembling modules using detector geometry. It can
        be converted to a real numpy array by calling the `asarray` method.
    """
    from extra_data.stacking import StackView

    if not train_data:
        raise ValueError("Empty data!")

    if '*' not in device:
        raise ValueError("Device name must contain a '*' which replaces "
                         "the module number!")
    prefix, suffix = device.split("*")

    if module_numbers is None:
        if modules is None:
            raise ValueError("At least one of module_numbers and modules "
                             "must be given!")
        module_numbers = np.arange(modules)
    modules = len(module_numbers)
    MAX_MODULES = 16
    if modules > MAX_MODULES:
        raise ValueError(f"Number of modules cannot exceed {MAX_MODULES}!")

    dtypes, shapes = set(), set()
    modno_arrays = {}
    for i, modno in enumerate(module_numbers):
        try:
            array = train_data[f"{prefix}{modno}{suffix}"][ppt]
        except KeyError:
            continue

        if memory_cell_last and array.ndim > 2:
            # (y, x, memory cells) -> (memory cells, y x)
            array = np.moveaxis(array, -1, 0)

        dtypes.add(array.dtype)
        shapes.add(array.shape)
        modno_arrays[i] = array

    if len(dtypes) > 1:
        raise ValueError(f"Modules have mismatched dtypes: {dtypes}!")

    if len(shapes) > 1:
        raise ValueError(f"Modules have mismatched shapes: {shapes}!")

    dtype = dtypes.pop()
    shape = shapes.pop()
    return StackView(modno_arrays, modules, shape, dtype,
                     fillvalue=np.nan, stack_axis=-3)


def use_doc(kls):
    def wrapper(method):
        @functools.wraps(method)
        def doc_method(self, *args, **kwargs):
            return method(self, *args, **kwargs)
        doc_method.__doc__ = getattr(kls, method.__name__).__doc__
        return doc_method
    return wrapper
