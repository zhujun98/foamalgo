"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu
"""
import functools

import numpy as np


class StackView:
    """Limited array-like object holding detector data from several modules.

    Access is limited to either a single module at a time or all modules
    together, but this is enough to assemble detector images.

    Copied from and credit to EXtra-data
    (https://github.com/European-XFEL/EXtra-data)
    """
    def __init__(self, data, nmodules, mod_shape, dtype, fillvalue,
                 stack_axis=-3):
        self._nmodules = nmodules
        self._data = data  # {modno: array}
        self.dtype = dtype
        self._fillvalue = fillvalue
        self._mod_shape = mod_shape
        self.ndim = len(mod_shape) + 1
        self._stack_axis = stack_axis
        if self._stack_axis < 0:
            self._stack_axis += self.ndim
        sax = self._stack_axis
        self.shape = mod_shape[:sax] + (nmodules,) + mod_shape[sax:]

    def __repr__(self):
        return "<VirtualStack (shape={}, {}/{} modules, dtype={})>".format(
            self.shape, len(self._data), self._nmodules, self.dtype,
        )

    # Multidimensional slicing
    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)

        missing_dims = self.ndim - len(slices)
        if Ellipsis in slices:
            ix = slices.index(Ellipsis)
            missing_dims += 1
            slices = slices[:ix] + (slice(None, None),) * missing_dims + slices[ix + 1:]
        else:
            slices = slices + (slice(None, None),) * missing_dims

        modno = slices[self._stack_axis]
        mod_slices = slices[:self._stack_axis] + slices[self._stack_axis + 1:]

        if isinstance(modno, int):
            if modno < 0:
                modno += self._nmodules
            return self._get_single_mod(modno, mod_slices)
        elif modno == slice(None, None):
            return self._get_all_mods(mod_slices)
        else:
            raise Exception(
                "VirtualStack can only slice a single module or all modules"
            )

    def _get_single_mod(self, modno, mod_slices):
        try:
            mod_data = self._data[modno]
        except KeyError:
            if modno >= self._nmodules:
                raise IndexError(modno)
            mod_data = np.full(self._mod_shape, self._fillvalue, self.dtype)
            self._data[modno] = mod_data

        # Now slice the module data as requested
        return mod_data[mod_slices]

    def _get_all_mods(self, mod_slices):
        new_data = {modno: self._get_single_mod(modno, mod_slices)
                    for modno in self._data}
        new_mod_shape = list(new_data.values())[0].shape
        return StackView(new_data, self._nmodules, new_mod_shape, self.dtype,
                         self._fillvalue)

    def asarray(self):
        """Copy this data into a real numpy array

        Don't do this until necessary - the point of using VirtualStack is to
        avoid copying the data unnecessarily.
        """
        start_shape = (self._nmodules,) + self._mod_shape
        arr = np.full(start_shape, self._fillvalue, dtype=self.dtype)
        for modno, data in self._data.items():
            arr[modno] = data
        return np.moveaxis(arr, 0, self._stack_axis)

    def squeeze(self, axis=None):
        """Drop axes of length 1 - see numpy.squeeze()"""
        if axis is None:
            slices = [0 if d == 1 else slice(None, None) for d in self.shape]
        elif isinstance(axis, (int, tuple)):
            if isinstance(axis, int):
                axis = (axis,)

            slices = [slice(None, None)] * self.ndim

            for ax in axis:
                try:
                    slices[ax] = 0
                except IndexError:
                    raise np.AxisError(
                        "axis {} is out of bounds for array of dimension {}"
                            .format(ax, self.ndim)
                    )
                if self.shape[ax] != 1:
                    raise ValueError("cannot squeeze out an axis with size != 1")
        else:
            raise TypeError("axis={!r} not supported".format(axis))

        return self[tuple(slices)]


def stack_detector_modules(train_data, device, ppt, *,
                           modules=None,
                           module_numbers=None,
                           memory_cell_last=False):
    """Stack detector modules.

    :param dict train_data: Data from a pulse-train.
    :param str device: Device or output channel name with a `*` replacing
        the module number. For example, SCS_DET_DSSC1M-1/DET/\*CH0:xtdf.
    :param str ppt: Property name.
    :param int modules: Number of modules. Ignored if `module_numbers` is
        given. For AGIPD-1M, LPD-1M and DSSC-1M, number of modules must be
        16 so it is not recommended to use `module_numbers` to specify
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
