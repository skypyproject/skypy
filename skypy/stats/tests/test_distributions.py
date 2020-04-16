from skypy.stats import check_rv, genschechter, schechter


def test_genschechter():
    check_rv(genschechter, (-1.2, 1.5, 1e-5), {
        # for alpha > 0, gamma > 0, a = 0, the distribution is gengamma
        (1.2, 1.5, 1e-100): ('gengamma', (2.2/1.5, 1.5))
    }, sample=False)


def test_schechter():
    check_rv(schechter, (-1.2, 1e-5), {
        # for alpha > 0, a = 0, the distribution is gamma
        (1.2, 1e-100): ('gamma', (2.2,))
    })
