import skypy.utils.astronomy as astro


def test_convert_abs_mag_to_lum():
    assert round(astro.convert_abs_mag_to_lum(-22), 1) == 630957344.5


def test_convert_lum_to_abs_mag():
    assert round(astro.convert_lum_to_abs_mag(630957344.5), 0) == -22
