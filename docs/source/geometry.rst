Detector Geometry
=================

.. currentmodule:: pyfoamalgo.geometry

1M-detector geometry
--------------------

.. autoclass:: AGIPD_1MGeometry

    .. automethod:: __init__
    .. automethod:: from_crystfel_geom
    .. automethod:: output_array_for_position_fast
    .. automethod:: position_all_modules
    .. automethod:: output_array_for_dismantle_fast
    .. automethod:: dismantle_all_modules


.. autoclass:: LPD_1MGeometry

    .. automethod:: __init__
    .. automethod:: from_h5_file_and_quad_positions
    .. automethod:: output_array_for_position_fast
    .. automethod:: position_all_modules
    .. automethod:: output_array_for_dismantle_fast
    .. automethod:: dismantle_all_modules


.. autoclass:: DSSC_1MGeometry

    .. automethod:: __init__
    .. automethod:: from_h5_file_and_quad_positions
    .. automethod:: output_array_for_position_fast
    .. automethod:: position_all_modules
    .. automethod:: output_array_for_dismantle_fast
    .. automethod:: dismantle_all_modules

Generalized geometry
--------------------

.. autoclass:: JungFrauGeometry

    .. automethod:: __init__
    .. automethod:: from_crystfel_geom
    .. automethod:: output_array_for_position_fast
    .. automethod:: position_all_modules
    .. automethod:: output_array_for_dismantle_fast
    .. automethod:: dismantle_all_modules


.. autoclass:: EPix100Geometry

    .. automethod:: __init__
    .. automethod:: output_array_for_position_fast
    .. automethod:: position_all_modules
    .. automethod:: output_array_for_dismantle_fast
    .. automethod:: dismantle_all_modules
