from astropy.table import Table
from astropy.table.np_utils import TableMergeError
import numpy as np
import pytest
from skypy.halo.abundance_matching import vale_ostriker


def test_vale_ostriker():
    """ Test Vale & Ostriker abundance matching algorithm"""

    # Mock catalogs for testing
    nh = 20
    ng = 10
    h = Table(data=np.random.uniform(size=(nh, 2)), names=['M', 'mass'])
    g = Table(data=np.random.uniform(size=(ng, 2)), names=['L', 'luminosity'])

    # Test default keys
    matched_default = vale_ostriker(h, g)
    argsort_m = np.argsort(matched_default['mass'])
    argsort_l = np.argsort(matched_default['luminosity'])
    assert np.all(argsort_m == argsort_l)

    # Test custom keys
    matched_custom = vale_ostriker(h, g, mass='M', luminosity='L')
    argsort_m = np.argsort(matched_custom['M'])
    argsort_l = np.argsort(matched_custom['L'])
    assert np.all(argsort_m == argsort_l)

    # Test failure with bad keys
    with pytest.raises(ValueError):
        vale_ostriker(h, g, mass='bad_key')
    with pytest.raises(ValueError):
        vale_ostriker(h, g, luminosity='bad_key')

    # Test correct table lengths for inner and outer joins
    assert len(vale_ostriker(h, g, join_type='inner')) == min(nh, ng)
    assert len(vale_ostriker(h, g, join_type='outer')) == max(nh, ng)

    # Test failure for exact join when nh != ng
    with pytest.raises(TableMergeError):
        vale_ostriker(h, g, join_type='exact')

    # Test failure if halos or subhalos are missing mass column
    s = Table(data=np.random.uniform(size=(5, 1)), names=['not mass'])
    with pytest.raises(ValueError):
        vale_ostriker(h, g, subhalos=s)
    with pytest.raises(ValueError):
        vale_ostriker(s, g, subhalos=h)

    # Test joining halos and subhalos.
    nh, ng, ns = 10, 15, 20
    h = Table(data=np.random.uniform(1, 2, size=(nh, 1)), names=['mass'])
    g = Table(data=np.random.uniform(size=(ng, 1)), names=['luminosity'])
    s = Table(data=np.random.uniform(size=(ns, 1)), names=['mass'])
    matched_subhalos = vale_ostriker(h, g, subhalos=s)
    assert sorted(matched_subhalos['mass'])[:-nh] == sorted(s['mass'])[nh-ng:]
    assert sorted(matched_subhalos['mass'])[-nh:] == sorted(h['mass'])
