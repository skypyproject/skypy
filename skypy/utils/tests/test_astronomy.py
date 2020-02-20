import skypy.utils.astronomy as astro


def test_convert_abs_mag_to_lum():
    assert round(astro.luminosity_from_absolute_magnitude(-22), 1) \
           == 630957344.5


def test_convert_lum_to_abs_mag():
    assert round(astro.absolute_magnitude_from_luminosity(630957344.5), 0) \
           == -22
