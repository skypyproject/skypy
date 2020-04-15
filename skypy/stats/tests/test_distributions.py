from skypy.stats import check_rv, schechter


def test_schechter():
    check_rv(schechter, (-1.2, 1e-5), {
        # for alpha > 0, a = 0, the distribution is gamma
        (1.2, 1e-100): ('gamma', (2.2,))
    })
