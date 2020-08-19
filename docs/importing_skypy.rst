*******************
Importing ``SkyPy``
*******************

In order to encourage consistency among users in importing and using SkyPy
functionality, we have put together the following guidelines.

Since most of the functionality in Astropy resides in sub-packages, importing
``astropy`` as::

    >>> import skypy

is not very useful. Instead, it's best to import the desired sub-package
with the syntax::

    >>> from skypy import subpackage  # doctest: +SKIP

For example, to access the galaxy-related functionality, you can import
`skypy.galaxy` with::

    >>> from skypy import galaxy
    >>> redshift = galaxy.redshift.smail(1.2, 1.5, 2.0, size=10)  # doctest: +SKIP

Note that for clarity, and to avoid any issues, we recommend **never**
importing any SkyPy functionality using ``*``, for example::

    >>> from skypy.galaxy import *  # NOT recommended # doctest: +SKIP