
ADR 3: Position sampling and patches of the sky
===============================================

Author: Nicolas Tessore  
Date: 5 February 2021  
Status: Accepted


Context
-------

This ADR addresses two related open problems for SkyPy simulations:

- How can we describe the regions in which to sample the positions of e.g.
  galaxies or supernovae?
- How can we break up the sampling over large fractions of the sky into smaller
  chunks?

The second point is particularly relevant when the simulation becomes so large
that it no longer fits into memory, as well as for parallelisation.


Extension to pipelines
----------------------

This ADR proposes to introduce a new top-level keyword `regions` [alt: `sky`,
`geometry`, `patches`] into pipelines that describes the geometry of the
simulation. For example, to include a "rectangular" and a "circular" region:

```yaml
regions:
- !rectangular [ ... ]
- !circular [ ... ]
```

The `regions` list can be traversed by the pipeline runner to create what are
effectively independent parallel simulations. The list items are objects with
the following interface.


Region interface
----------------

The regions need to support two sets of operations:

- Information queries: For example, there should be a `.area` [alt:
  `.solid_angle`] attribute that returns the solid angle of the region.
- Random sampling: There needs to be at least a `random_point()` [alt:
  `random_position()`, `uniform_in()`] function that can uniformly sample a
  random position from within the region.

When the pipeline runner traverses the list of regions, it can keep track of
the current region in a `$region` reference that can be used where necessary.
For example, to sample from a luminosity function with positions:

```yaml
tables:
  galaxies:
    z, M: !schechter_lf
      ...
      sky_area: $region.area
    ra, dec: !random_point [ $region ]
```


Support for HEALPix maps
------------------------

The above proposal is powerful enough to support advanced features such as
regions that are described by HEALPix maps. There may be a `healpix()` function
that generates a list of regions from HEALPix pixels:

```yaml
regions: !healpix
  nside: 8
```

The resulting list would contain `12 * nside**2 = 768` regions corresponding
to the HEALPix pixels of a map with `nside = 8`.

The function is easily extensible. For example, instead of using all HEALPix
pixels, there might be a footprint that describes a specific survey:

```yaml
regions: !healpix
  mask: survey-footprint-n512.fits
```

The `mask` keyword can be combined with the `nside` parameter to change the
resolution of the mask if requested.

If the HEALPix maps become finely resolved, it may be desirable to combine
several pixels into a single region. There may be a `batch` [alt: `combine`]
keyword for this purpose:

```yaml
regions: !healpix
  nside: 8
  batch: 16
```

The resulting list of regions will contain `768/16 = 48` regions. The `batch`
keyword may also take a quantity of solid angle and automatically choose the
number of pixels to combine accordingly.


Map making
----------

This ADR does not address the problem of how maps will be generated from the
list of regions. For example, a very real use case would be to generate
populations of galaxies and simply count the total number in each HEALPix pixel
to generate a density map. This will be addressed in a separate ADR.


Consequences
------------
The existing `Pipeline` class must be extended to support iterating regions. No
existing interfaces are affected.
