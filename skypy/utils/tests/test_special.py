import skypy.utils.special as special


def test_upper_incomplete_gamma():
    assert round(special.upper_incomplete_gamma(0.5, 1.5), 6) == 0.147583
    assert round(special.upper_incomplete_gamma(-0.3, 1.5), 6) == 0.080007
