****************************
Positions (`skypy.position`)
****************************


Uniform distributions
=====================

.. _skypy.position.uniform_around:

The simplest distribution of positions on the sky is the uniform distribution
over a circular region with given centre and area. To sample this distribution,
use the `~skypy.position.uniform_around` function:

.. literalinclude:: examples/uniform_around.yml
   :language: yaml

.. plot::
   :include-source: false

    from skypy.pipeline import Pipeline

    pipeline = Pipeline.read('examples/uniform_around.yml')
    pipeline.execute()

    coords = pipeline['positions']
    ra, dec = coords.ra.wrap_at('180d').radian, coords.dec.radian

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4.2))
    plt.subplot(111, projection="aitoff")
    plt.plot(ra, dec, '.', alpha=0.2)
    plt.grid()


Reference/API
=============

.. automodapi:: skypy.position
