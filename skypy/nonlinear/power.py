import numpy as np
from scipy.interpolate import interp1d


def halofit(wavenumber, redshift, linear_power_spectrum, cosmology):

    def s1(lnr, dlnk, dl2, k2):
        r2 = np.exp(2*lnr)
        return np.sum(dlnk*dl2*np.exp(-k2*r2), axis=1)

    def s2(lnr, dlnk, dl2, k2):
        r2 = np.exp(2*lnr)
        return np.sum(2*k2*r2*dlnk*dl2*np.exp(-k2*r2), axis=1)

    def s3(lnr, dlnk, dl2, k2):
        r2 = np.exp(2*lnr)
        return np.sum(4*np.square(k2)*r2*r2*dlnk*dl2*np.exp(-k2*r2), axis=1)

    z = redshift
    k = wavenumber
    P = linear_power_spectrum.T
    omega_m_0 = cosmology.Om0
    omega_l_0 = cosmology.Ode0

    # Log(k) grid for integral
    lnk = np.log(k)
    dlnk = np.zeros(k.shape)
    dlnk[0] = 0.5 * (lnk[1] - lnk[0])
    dlnk[1:-1] = 0.5 * (lnk[2:] - lnk[:-2])
    dlnk[-1] = 0.5 * (lnk[-1] - lnk[-2])

    # Equations A4 & A5
    logr = np.flip(-lnk)  # Trial solutions for log(R)
    k2 = np.square(k)
    dl2_kz = P * k * k2 / (2 * np.pi * np.pi)
    # Evaluate integral A4 at redshift zero
    sigma2 = s1(logr.reshape((logr.size, 1)), dlnk, dl2_kz[0, :], k2)
    logsigma2 = np.log(sigma2)
    interp = interp1d(logsigma2, logr)
    # Solution for log(R) at each redshift depends on the growth factor
    root = interp(-np.log(P[:, 0]/P[0, 0]))
    s1r = s1(root.reshape((root.size, 1)), dlnk, dl2_kz[0, :], k2)
    s2r = s2(root.reshape((root.size, 1)), dlnk, dl2_kz[0, :], k2)
    s3r = s3(root.reshape((root.size, 1)), dlnk, dl2_kz[0, :], k2)
    ksigmainv = np.exp(root)
    neff = (- 3 + s2r / s1r).reshape((z.size, 1))
    neff2 = np.square(neff)
    neff3 = neff2 * neff
    neff4 = neff3 * neff
    c = ((s2r*s2r)/(s1r*s1r) + (2*s2r - s3r) / s1r).reshape((z.size, 1))

    # Equations A6-13
    an = np.power(10, 1.5222 + 2.8553*neff + 2.3706*neff2 + 0.9903*neff3
                  + 0.2250*neff4 - 0.6038*c)
    bn = np.power(10, -0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*c)
    cn = np.power(10, 0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*c)
    gamman = 0.1971 - 0.0843*neff + 0.8460*c
    alphan = np.abs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*c)
    betan = 2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff4 \
        - 0.1682*c
    nun = np.power(10, 5.2105 + 3.6902*neff)

    # Equation A14
    am3 = ((1.+z)*(1.+z)*(1.+z)).reshape((z.size, 1))
    omega_m_z = am3 * omega_m_0 / (omega_l_0 + am3*omega_m_0)
    f1 = np.power(omega_m_z, -0.0307)
    f2 = np.power(omega_m_z, -0.0585)
    f3 = np.power(omega_m_z,  0.0743)

    # Equations A1, A2 & A3
    y = k * np.array(ksigmainv).reshape((z.size, 1))
    fy = 0.25*y + 0.125*np.square(y)
    dq2 = dl2_kz * (np.power(1+dl2_kz, betan)/(1+alphan*dl2_kz)) * np.exp(-fy)
    dh2p = an * np.power(y, 3*f1) / (1 + bn*np.power(y, f2)
                                     + np.power(cn*f3*y, 3-gamman))
    dh2 = dh2p / (1 + nun/np.square(y))
    pknl = 2 * np.pi * np.pi * (dq2+dh2) / (k2*k)

    return pknl.T
