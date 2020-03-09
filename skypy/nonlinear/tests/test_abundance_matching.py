from astropy.table import Table
from astropy.table.np_utils import TableMergeError
import numpy as np
import pytest
from skypy.nonlinear.abundance_matching import vale_ostriker


def test_vale_ostriker():
    """ Test Vale & Ostriker abundance matching algorithm"""

    ng = 10
    nh = 20
    h = Table(data=np.random.uniform(size=(nh, 2)), names=['M', 'mass'])
    g = Table(data=np.random.uniform(size=(ng, 2)), names=['L', 'luminosity'])

    matched_default = vale_ostriker(h, g)
    argsort_m = np.argsort(matched_default['mass'])
    argsort_l = np.argsort(matched_default['luminosity'])
    assert np.all(argsort_m == argsort_l)

    matched_custom = vale_ostriker(h, g, mass='M', luminosity='L')
    argsort_m = np.argsort(matched_custom['M'])
    argsort_l = np.argsort(matched_custom['L'])
    assert np.all(argsort_m == argsort_l)

    with pytest.raises(ValueError):
        vale_ostriker(h, g, mass='bad_key')

    with pytest.raises(ValueError):
        vale_ostriker(h, g, luminosity='bad_key')

    assert len(vale_ostriker(h, g, join_type='inner')) == min(nh, ng)

    assert len(vale_ostriker(h, g, join_type='outer')) == max(nh, ng)

    with pytest.raises(TableMergeError):
        vale_ostriker(h, g, join_type='exact')
