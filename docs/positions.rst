****************************
Positions (`skypy.position`)
****************************


Uniform distributions
=====================

The simplest distribution of points on the sky is the uniform distribution over
a spherical region with a given centre. To sample `n` points from this
distribution, use the `~skypy.position.uniform_around` function:

.. code-block:: yaml

    # survey information
    survey_centre: !astropy.coordinates.SkyCoord [ 9h50m59.75s, +11d39m22.15s ]
    survey_area: !astropy.units.Quantity [ '1000 deg2' ]

    # sample 1000 galaxies in survey
    positions: !skypy.position.uniform_around
      centre: $survey_centre
      area: $survey_area
      size: 1000


Reference/API
=============

.. automodapi:: skypy.position
