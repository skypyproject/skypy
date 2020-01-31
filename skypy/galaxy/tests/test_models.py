import skypy.galaxy.models as model
import numpy as np
import pytest


# check blue
def test_herbel_params_blue():
    a_m, b_m, a_phi, b_phi, alpha = model._herbel_params('blue')
    result = [a_m, b_m, a_phi, b_phi, alpha]
    check = [-0.9408582,
             -20.40492365,
             -0.10268436,
             0.00370253,
             -1.3]
    np.testing.assert_equal(result, check)


# check red
def test_herbel_params_red():
    a_m, b_m, a_phi, b_phi, alpha = model._herbel_params('red')
    result = [a_m, b_m, a_phi, b_phi, alpha]
    check = [-0.70798041,
             -20.37196157,
             -0.70596888,
             0.0035097,
             -0.5]
    np.testing.assert_equal(result, check)


# check error
def test_herbel_params_error():
    with pytest.raises(ValueError) as e:
        model._herbel_params('green')
    assert str(e.value) == 'Galaxy type has to be blue or red.'
