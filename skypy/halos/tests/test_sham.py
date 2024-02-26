# Imports
import numpy as np
import pytest
from pytest import approx
from skypy.halos._colossus import HAS_COLOSSUS


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_quenching_funct():
    # Create catalogue
    from astropy.cosmology import WMAP9  # Cannot be FlatLambdaCDM
    from skypy.halos.mass import colossus_mass_sampler
    from skypy.halos.sham import quenching_funct
    m_min, m_max, size = 1e+10, 1e+16, 1000000
    parent_halo = colossus_mass_sampler(redshift=0.1, model='sheth99',
                                        mdef='fof', m_min=m_min, m_max=m_max,
                                        cosmology=WMAP9, sigma8=0.8, ns=1.,
                                        size=size, resolution=1000)

    # Parameters
    m_mu, sigma, base = 10**(12.1), 0.4, 0.2

    h_quench = quenching_funct(parent_halo, m_mu, sigma, base)  # Quenched halos

    assert ((h_quench == 0) | (h_quench == 1)).all()  # Check result is binary
    assert h_quench.shape == parent_halo.shape  # Check the shapes are the same

    # Make histogram of results
    qu_stack = np.stack((parent_halo, h_quench), axis=1)
    order = qu_stack[np.argsort(qu_stack[:, 0])]
    ha_order = order[:, 0]
    qu_order = order[:, 1]

    mass_bins = np.geomspace(min(parent_halo), max(parent_halo), 50)
    mass_mid = (mass_bins[:-1] + mass_bins[1:])/2

    fract = []
    for ii in range(0, len(mass_bins) - 1):
        l, r = mass_bins[ii], mass_bins[ii+1]
        # Halos in range
        h_bin = np.where(ha_order[np.where(ha_order < r)] >= l)[0]
        q_bin = qu_order[h_bin]

        if len(q_bin) != 0:
            fract.append(len(np.where(q_bin == 1)[0])/len(q_bin))
        else:
            fract.append(0)  # Some bins near end won't have any data in

    fract = np.array(fract)

    # Original PDF
    from scipy.special import erf
    mass = np.geomspace(min(parent_halo), max(parent_halo), 100)
    calc = np.log10(mass/m_mu)/sigma
    old_prob = 0.5 * (1.0 + erf(calc/np.sqrt(2)))  # Base error funct, 0 -> 1
    add_val = base/(1-base)
    prob_sub = old_prob + add_val  # Add value to plot to get baseline
    prob_array = prob_sub/prob_sub[-1]  # Normalise, base -> 1

    # Interp to get same masses
    prob = np.interp(mass_mid, mass, prob_array)

    # Compare sampled to expected distribution
    dist = (prob - fract)**2
    dist = np.delete(dist, np.where(dist > 10**(-1))[0])  # Delete values where there was no data
    assert sum(dist)/len(dist) < 1e-4  # Is average difference small enough

    # Check the first value is around baseline
    assert fract[0] == approx(base, rel=10**(-1))

    # Check the mean is correct
    assert np.interp(m_mu, mass_mid, fract) == approx((1 + base)/2, rel=10**(-1))

    # Check the end is correct
    fract_one = fract[np.where(fract > 0)]  # Remove any zero values ie where there is no data
    assert fract_one[-1] == approx(1, rel=10**(-1))

    # Check the errors trigger
    with pytest.raises(Exception) as excinfo:
        m_mu, sigma, base = -10**(12), 0.4, 0.2
        quenching_funct(parent_halo, m_mu, sigma, base)
    assert str(excinfo.value) == 'M_mu cannot be negative or 0 solar masses'
    with pytest.raises(Exception) as excinfo:
        m_mu, sigma, base = 0, 0.4, 0.2
        quenching_funct(parent_halo, m_mu, sigma, base)
    assert str(excinfo.value) == 'M_mu cannot be negative or 0 solar masses'
    with pytest.raises(Exception) as excinfo:
        m_mu, sigma, base = 10**(12), -0.3, 0.2
        quenching_funct(parent_halo, m_mu, sigma, base)
    assert str(excinfo.value) == 'sigma cannot be negative or 0'
    with pytest.raises(Exception) as excinfo:
        m_mu, sigma, base = 10**(12), 0, 0.2
        quenching_funct(parent_halo, m_mu, sigma, base)
    assert str(excinfo.value) == 'sigma cannot be negative or 0'
    with pytest.raises(Exception) as excinfo:
        m_mu, sigma, base = 10**(12), 0.3, -0.7
        quenching_funct(parent_halo, m_mu, sigma, base)
    assert str(excinfo.value) == 'baseline cannot be negative, set_baseline = -baseline'
    with pytest.raises(Exception) as excinfo:
        m_mu, sigma, base = 10**(12), 0.3, 1.3
        quenching_funct(parent_halo, m_mu, sigma, base)
    assert str(excinfo.value) == 'baseline cannot be more than 1'
    with pytest.raises(Exception) as excinfo:
        m_mu, sigma, base = 10**(12), 0.3, 0.3
        quenching_funct(10**(11), m_mu, sigma, base)
    assert str(excinfo.value) == 'Mass must be array-like'


@pytest.mark.flaky
def test_find_min():
    from scipy.integrate import trapezoid as trap
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM
    from skypy.halos.sham import find_min

    # Parameters
    m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    z_range = [0.01, 0.1]
    skyarea = 600
    max_mass = 10**(14)
    no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000

    # Run function when interpolation should work
    min_mass = find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)

    # Check type and expected number
    assert type(min_mass) is np.float64
    assert np.log10(min_mass) == approx(7.9976, rel=10**(-4))

    # Check that the expected integral gives the number of halos
    skyarea = skyarea*u.deg**2
    z = np.linspace(z_range[0], z_range[1], 1000)
    dVdz = (cosmology.differential_comoving_volume(z)*skyarea).to_value('Mpc3')

    mass_range = np.geomspace(min_mass, max_mass, 1000)
    phi_m = phi_star/m_star
    m_m = mass_range/m_star

    dndmdV = phi_m*np.e**(-m_m)*(m_m)**alpha  # Mass function
    dndV = trap(dndmdV, mass_range)
    dn = dndV*dVdz

    assert trap(dn, z) == approx(no_halo_y, rel=10**(-1))

    # Run function with a non-optimal number of halos, check for exceptions
    skyarea = 600
    with pytest.raises(Exception) as excinfo:
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_min)
    assert str(excinfo.value) == 'Number of halos not within reachable bounds'

    with pytest.raises(Exception) as excinfo:
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_max)
    assert str(excinfo.value) == 'Number of halos not within reachable bounds'

    # Check function gives expected output using 'run_anyway'
    test1 = find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                     max_mass, no_halo_min, run_anyway=True)
    test2 = find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                     max_mass, no_halo_max, run_anyway=True)
    assert test1 == approx(10**(7))
    assert test2 == approx(10**(7))

    # Check errors trigger
    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1, 10]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'The wrong number of redshifts were given'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [-0.01, 0.1]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'Redshift cannot be negative'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, -0.1]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'Redshift cannot be negative'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [-0.01, -0.1]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'Redshift cannot be negative'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.1, 0.01]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'The second redshift should be more than the first'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), 0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'Schechter function defined so alpha < 0, set_alpha = -alpha'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = -10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'M* and phi* must be positive and non-zero numbers'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = -10**(10.58), 0, -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = 600
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'M* and phi* must be positive and non-zero numbers'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = -60
        max_mass = 10**(14)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'The skyarea must be a positive non-zero number'

    with pytest.raises(Exception) as excinfo:
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.1, 0.01]
        skyarea = 60
        max_mass = 10**(4)
        no_halo_y, no_halo_min, no_halo_max = 10000, 600, 50000
        find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo_y)
        assert str(excinfo.value) == 'The maximum mass must be more than 10^6 solar masses'


@pytest.mark.flaky
def test_run_file():
    import astropy
    from scipy.integrate import trapezoid as trap
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM
    from skypy.halos.sham import run_file

    # Test catalogue is the expected length
    m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    z_range = [0.01, 0.1]
    skyarea = 600
    min_mass = 10**(7.5)
    max_mass = 10**(14)

    file_test = '/mnt/c/Users/user/Documents/skypy_dev/skypy/skypy/halos/test_gal.yaml'
    test_cat, test_z = run_file(file_test, 'galaxy', 'sm', 'z')

    skyarea = skyarea*u.deg**2
    z = np.linspace(z_range[0], z_range[1], 1000)
    dVdz = (cosmology.differential_comoving_volume(z)*skyarea).to_value('Mpc3')

    mass_range = np.geomspace(min_mass, max_mass, 1000)
    phi_m = phi_star/m_star
    m_m = mass_range/m_star

    dndmdV = phi_m*np.e**(-m_m)*(m_m)**alpha  # Mass function
    dndV = trap(dndmdV, mass_range)
    dn = dndV*dVdz

    assert type(test_cat) is astropy.table.column.Column
    assert type(test_z) is astropy.table.column.Column
    assert test_cat.ndim == 1
    assert test_z.ndim == 1
    # Length same within Poisson sampling error
    assert trap(dn, z) == approx(len(test_cat), rel=np.sqrt(trap(dn, z)))
    assert len(test_cat) == len(test_z)

    # Check errors trigger
    with pytest.raises(Exception) as excinfo:
        run_file(82, 'galaxy', 'sm', 'z')
        assert str(excinfo.value) == 'File name must be a string'

    with pytest.raises(Exception) as excinfo:
        run_file('GNDN.yaml', 'galaxy', 'sm', 'z')
        assert str(excinfo.value) == 'File does not exist'


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_gen_sub_cat():
    from astropy.cosmology import WMAP9  # Cannot be FlatLambdaCDM
    from skypy.halos.mass import colossus_mass_sampler
    from skypy.halos.sham import gen_sub_cat

    # Parameters
    alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3

    # Catalogue
    m_min, m_max, size = 1e+10, 1e+16, 1000
    parent_halo = colossus_mass_sampler(redshift=0.1, model='sheth99',
                                        mdef='fof', m_min=m_min, m_max=m_max,
                                        cosmology=WMAP9, sigma8=0.8, ns=1.,
                                        size=size, resolution=1000)
    z_halo = np.linspace(0.01, 0.1, len(parent_halo))

    # Function
    ID_halo, sub_masses, ID_sub, z_sub = gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)

    # Check array sets
    assert ID_halo.ndim == 1
    assert sub_masses.ndim == 1
    assert ID_sub.ndim == 1
    assert z_sub.ndim == 1
    assert len(ID_halo) == len(parent_halo)
    assert len(sub_masses) == len(ID_sub)
    assert len(sub_masses) == len(z_sub)

    # Check the data inside is correct
    assert min(sub_masses) >= min(parent_halo)
    assert max(sub_masses) <= max(parent_halo)/2

    assert np.all(np.isin(z_sub, z_halo))
    assert ((ID_halo < 0)).all()
    assert ((ID_sub > 0)).all()

    i_type = []
    id_list = np.append(ID_halo, ID_sub)
    for ii in id_list:
        if type(ii) is np.int64:
            i_type.append(True)
        else:
            i_type.append(False)
    assert np.all(i_type)  # Check ids are all integers

    # Check a single value works
    # Parameters
    alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3

    # Catalogue
    parent_halo = 1e+16
    z_halo = 0.01

    # Function
    ID_halo, sub_masses, ID_sub, z_sub = gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)

    assert sub_masses.size  # Check array is not empty, ie it worked

    # Check errors trigger
    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3

        # Catalogue
        parent_halo = 1e+10
        z_halo = np.linspace(0.01, 0.1, 1000)
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Parent halos must be array-like'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3

        # Catalogue
        m_min, m_max, size = 1e+10, 1e+16, 1000
        parent_halo = colossus_mass_sampler(redshift=0.1, model='sheth99',
                                            mdef='fof', m_min=m_min, m_max=m_max,
                                            cosmology=WMAP9, sigma8=0.8, ns=1.,
                                            size=size, resolution=1000)
        z_halo = 0.1
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Redshift must be array-like'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3

        # Catalogue
        m_min, m_max, size = 1e+10, 1e+16, 1000
        parent_halo = colossus_mass_sampler(redshift=0.1, model='sheth99',
                                            mdef='fof', m_min=m_min, m_max=m_max,
                                            cosmology=WMAP9, sigma8=0.8, ns=1.,
                                            size=size, resolution=1000)
        z_halo = np.linspace(0.01, 0.1, 500)
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Catalogue of halos and redshifts must be the same length'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3

        # Catalogue
        parent_halo = [-10, 1, 0]
        z_halo = np.linspace(0.01, 0.1, 3)
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Catalogue of halos and redshifts must be the same length'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [-0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Redshifts in catalogue should be positive'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = -1.91, 0.39, 0.1, 3

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Subhalo mass function defined alpha > 0, set_alpha = -alpha'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 2, 0.39, 0.1, 3

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Subhalo alpha must be less than 2'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 0.39, 0.1, 0.5

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Subhalo x cannot be less than 1'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, -0.39, 0.1, 3

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Subhalo beta must be between 0 and 1'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 1.39, -0.1, 3

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Subhalo beta must be between 0 and 1'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 0.39, -0.1, 3

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Subhalo gamma must be between 0 and 1'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        alpha, beta, gamma, x = 1.91, 1.39, 1.1, 3

        # Catalogue
        parent_halo = 10**(np.array([10, 12, 11]))
        z_halo = [0.05, 0.1, 0.01]
        gen_sub_cat(parent_halo, z_halo, alpha, beta, gamma, x)
        assert str(excinfo.value) == 'Subhalo gamma must be between 0 and 1'


@pytest.mark.flaky
def test_galaxy_cat():
    from astropy.cosmology import FlatLambdaCDM
    import os
    from skypy.halos.sham import galaxy_cat

    # Parameters
    m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    z_range = [0.01, 0.1]
    skyarea = 600
    min_mass = 10**(7.5)
    max_mass = 10**(14)
    file_name = 'unit_test.yaml'

    # Function
    cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range,
                     skyarea, min_mass, max_mass, file_name)

    # Check the catalogue
    assert cat.size

    # Check file exists
    assert os.path.exists('unit_test.yaml')

    # Open file and check structure
    # Create expected lines
    line1 = 'm_star: !numpy.power [10, 10.58]\n'
    line2 = 'phi_star: !numpy.power [10, -2.77]\n'
    line3 = 'alpha_val: -0.33\n'

    # Mass range
    line4 = 'm_min: !numpy.power [10, 7.5]\n'
    line5 = 'm_max: !numpy.power [10, 14.0]\n'

    # Observational parameters
    line6 = 'sky_area: 600.0 deg2\n'
    line7 = 'z_range: !numpy.linspace [0.01, 0.1, 100]\n'

    # Cosmology
    line8 = 'cosmology: !astropy.cosmology.FlatLambdaCDM\n'
    line9 = '  H0: 70.0\n'
    line10 = '  Om0: 0.3\n'

    # Call function
    function = 'tables:\n  galaxy:\n'
    function += '    z, sm: !skypy.galaxies.schechter_smf\n      redshift: $z_range\n'
    function += '      m_star: $m_star\n      phi_star: $phi_star\n'
    function += '      alpha: $alpha_val\n      m_min: $m_min\n'
    function += '      m_max: $m_max\n      sky_area: $sky_area\n'

    exp_struct = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9
    exp_struct += line10 + function

    file = open('unit_test.yaml', 'r')
    lines = file.readlines()

    fil_struct = ''
    for ii in lines:
        fil_struct += ii

    assert fil_struct == exp_struct  # Do lines in file match expected lines

    # Check errors trigger
    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'The wrong number of redshifts were given'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [-0.01, 0.1]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'Redshift cannot be negative'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [-0.01, -0.1]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'Redshift cannot be negative'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.1, 0.01]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'The second redshift should be more than the first'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), 0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'Schechter function defined so alpha < 0, set_alpha = -alpha'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = -10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'M* and phi* must be positive and non-zero numbers'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = -10**(10.58), 0, -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'M* and phi* must be positive and non-zero numbers'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = -6
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'The skyarea must be a positive non-zero number'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
        z_range = [0.01, 0.1]
        skyarea = 600
        min_mass = 10**(14)
        max_mass = 10**(8)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'The minimum mass should be less than the maximum mass'

    with pytest.raises(Exception) as excinfo:
        # Parameters
        m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        z_range = [0.01, 0.1]
        skyarea = 600
        min_mass = 10**(7.5)
        max_mass = 10**(14)
        file_name = 'unit_test.yaml'

        # Function
        cat = galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea,
                         min_mass, max_mass, file_name)
        assert str(excinfo.value) == 'Cosmology object must have an astropy cosmology name'

    file.close()
    os.remove('unit_test.yaml')


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_assignment():
    # Generate the catalogues
    from astropy.cosmology import WMAP9
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    from skypy.halos.mass import colossus_mass_sampler
    from skypy.halos.sham import assignment
    from skypy.halos.sham import gen_sub_cat
    from skypy.halos.sham import quenching_funct
    from skypy.galaxies._schechter import schechter_smf

    # Catalogues
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    m_min, m_max, size = 1e+10, 1e+16, 1000
    halo_cat = colossus_mass_sampler(redshift=0.1, model='sheth99',
                                     mdef='fof', m_min=m_min, m_max=m_max,
                                     cosmology=WMAP9, sigma8=0.8, ns=1.,
                                     size=size, resolution=1000)

    h_z = np.linspace(0.01, 0.1, len(halo_cat))
    z_range = [min(h_z), max(h_z)]
    skyarea = 600*u.deg**2

    alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3
    ID_halo, sub_masses, ID_sub, z_sub = gen_sub_cat(halo_cat, h_z, alpha, beta, gamma, x)

    m_star1, phi_star1, alpha1 = 10**(10.58), 10**(-2.77), -0.33
    m_star2, phi_star2, alpha2 = 10**(10.64), 10**(-4.24), -1.54
    m_star3, phi_star3, alpha3 = 10**(10.65), 10**(-2.98), -1.48
    m_star4, phi_star4, alpha4 = 10**(10.55), 10**(-3.96), -1.53

    m_min1 = 10**(6.7)
    m_min2 = 10**(6.6)
    m_min3 = 10**(7.0)
    m_min4 = 10**(7.0)

    m_max1 = 10**(14)
    m_max2 = 10**(13)

    rc_cat = schechter_smf(z_range, m_star1, phi_star1, alpha1, m_min1, m_max1,
                           skyarea, cosmo, noise=False)[1]
    rs_cat = schechter_smf(z_range, m_star2, phi_star2, alpha2, m_min2, m_max2,
                           skyarea, cosmo, noise=False)[1]
    bc_cat = schechter_smf(z_range, m_star3, phi_star3, alpha3, m_min3, m_max1,
                           skyarea, cosmo, noise=False)[1]
    bs_cat = schechter_smf(z_range, m_star4, phi_star4, alpha4, m_min4, m_max2,
                           skyarea, cosmo, noise=False)[1]

    # Quench the halos
    h_quench = quenching_funct(halo_cat, 10**(12), 0.4)
    s_quench = quenching_funct(sub_masses, 10**(12), 0.4, 0.4)

    # Order the arrays
    halo_subhalo = np.concatenate((halo_cat, sub_masses), axis=0)  # Halos
    ID_list = np.concatenate((ID_halo, ID_sub), axis=0)  # IDs
    z_list = np.concatenate((h_z, z_sub), axis=0)  # Redshifts
    q_list = np.concatenate((h_quench, s_quench), axis=0)  # Quenching

    stack = np.stack((halo_subhalo, ID_list, z_list, q_list), axis=1)
    order1 = stack[np.argsort(stack[:, 0])]
    hs_order = np.flip(order1[:, 0])
    id_hs = np.flip(order1[:, 1])
    z_hs = np.flip(order1[:, 2])
    qu_hs = np.flip(order1[:, 3])
    rc_order = np.flip(np.sort(rc_cat))  # Galaxies
    rs_order = np.flip(np.sort(rs_cat))
    bc_order = np.flip(np.sort(bc_cat))
    bs_order = np.flip(np.sort(bs_cat))

    # Function
    hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_order,
                                                          bc_order, bs_order, qu_hs, id_hs, z_hs)

    # Check arrays shape/type
    assert hs_fin.shape == gal_fin.shape == id_fin.shape == z_fin.shape == gal_type.shape
    assert hs_fin.ndim == 1

    # Check arrays contain expected values
    assert ((gal_type == 1) | (gal_type == 2) | (gal_type == 3) | (gal_type == 4)).all()
    assert len(np.where(gal_type == 1)[0]) > 1  # Check some of each were assigned
    assert len(np.where(gal_type == 2)[0]) > 1
    assert len(np.where(gal_type == 3)[0]) > 1
    assert len(np.where(gal_type == 4)[0]) > 1

    assert min(z_fin) >= min(z_range)  # z within range
    assert max(z_fin) <= max(z_range)

    i_type = []
    for ii in id_fin:
        if abs(ii - int(ii)) > 0:
            i_type.append(False)
        else:
            i_type.append(True)

    assert np.all(i_type)  # Check ids are all integers
    assert len(np.where(id_fin > 0)[0]) > 1  # Check there are some subhalos
    assert len(np.where(id_fin < 0)[0]) > 1  # Check there are some parents

    assert max(gal_fin) <= 10**(15)
    assert min(gal_fin) >= 10**(4)  # Check galaxies in correct range
    assert max(hs_fin) <= 10**(17)
    assert min(hs_fin) >= 10**(7)  # Check halos in correct range

    # Check galaxies are assigned to correct halos
    check_parent_t = gal_type[np.where(id_fin < 0)]  # Finished arrays
    check_parent_m = gal_fin[np.where(id_fin < 0)]
    check_subhal_t = gal_type[np.where(id_fin > 0)]
    check_subhal_m = gal_fin[np.where(id_fin > 0)]
    check_blu_t = gal_type[np.where(qu_hs == 0)]
    check_blu_m = gal_fin[np.where(qu_hs == 0)]
    check_red_t = gal_type[np.where(qu_hs == 1)]
    check_red_m = gal_fin[np.where(qu_hs == 1)]

    assert ((check_parent_t == 1) | (check_parent_t == 3)).all()  # Check correct tags are applied
    assert ((check_subhal_t == 2) | (check_subhal_t == 4)).all()
    assert ((check_blu_t == 3) | (check_blu_t == 4)).all()
    assert ((check_red_t == 1) | (check_red_t == 2)).all()

    cen_append = np.append(rc_order, bc_order)  # Catalogue
    sub_append = np.append(rs_order, bs_order)
    blu_append = np.append(bc_order, bs_order)
    red_append = np.append(rc_order, rs_order)

    assert np.all(np.isin(check_parent_m, cen_append))  # Check masses are from correct lists
    assert np.all(np.isin(check_subhal_m, sub_append))
    assert np.all(np.isin(check_blu_m, blu_append))
    assert np.all(np.isin(check_red_m, red_append))

    # Check errors trigger
    with pytest.raises(Exception) as excinfo:
        hs_test = -1*hs_order
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_test, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Halo masses must be positive and non-zero'

    with pytest.raises(Exception) as excinfo:
        rs_test = -1*rs_order
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_test,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Galaxy masses must be positive and non-zero'

    with pytest.raises(Exception) as excinfo:
        hs_test = np.flip(hs_order)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_test, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Halo masses were in the wrong order and have been corrected'

    with pytest.raises(Exception) as excinfo:
        hs_test = np.random.randint(min(hs_order), max(hs_order), len(hs_order))
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_test, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Halos are not in a sorted order'

    with pytest.raises(Exception) as excinfo:
        rc_test = np.flip(rc_order)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_test, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Red central galaxies were in wrong order now correct'

    with pytest.raises(Exception) as excinfo:
        rc_test = np.random.randint(min(rc_order), max(rc_order), len(rc_order))
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_test, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Red central galaxies are not in a sorted order'

    with pytest.raises(Exception) as excinfo:
        rs_test = np.flip(rs_order)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_test,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Red satellite galaxies were in wrong order now correct'

    with pytest.raises(Exception) as excinfo:
        rs_test = np.random.randint(min(rs_order), max(rs_order), len(rs_order))
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_test,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Red satellite galaxies are not in a sorted order'

    with pytest.raises(Exception) as excinfo:
        bc_test = np.flip(bc_order)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_order,
                                                              bc_test, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Blue central galaxies were in wrong order and now correct'

    with pytest.raises(Exception) as excinfo:
        bc_test = np.random.randint(min(bc_order), max(bc_order), len(bc_order))
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_order,
                                                              bc_test, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Blue central galaxies are not in a sorted order'

    with pytest.raises(Exception) as excinfo:
        bs_test = np.flip(bs_order)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_order,
                                                              bc_order, bs_test, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Blue satellite galaxies were in wrong order and now correct'

    with pytest.raises(Exception) as excinfo:
        bs_test = np.random.randint(min(bs_order), max(bs_order), len(bs_order))
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_order,
                                                              bc_order, bs_test, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'Blue satellite galaxies are not in a sorted order'

    with pytest.raises(Exception) as excinfo:
        hs_test = np.geomspace(max(hs_order), min(hs_order), 10)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_test, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)
        assert str(excinfo.value) == 'All arrays pertaining to halos must be the same shape'

    with pytest.raises(Exception) as excinfo:
        id_test = np.geomspace(min(id_hs), max(id_hs), 10)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_test, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_test,
                                                              z_hs)
        assert str(excinfo.value) == 'All arrays pertaining to halos must be the same shape'

    with pytest.raises(Exception) as excinfo:
        z_test = np.geomspace(min(z_hs), max(z_hs), 10)
        hs_fin, gal_fin, id_fin, z_fin, gal_type = assignment(hs_order, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_test)
        assert str(excinfo.value) == 'All arrays pertaining to halos must be the same shape'


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_sham_plots():
    # Generate the catalogues
    from astropy.cosmology import WMAP9
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    from skypy.halos.mass import colossus_mass_sampler
    from skypy.halos.sham import assignment
    from skypy.halos.sham import gen_sub_cat
    from skypy.halos.sham import quenching_funct
    from skypy.halos.sham import sham_plots
    from skypy.galaxies._schechter import schechter_smf

    # Catalogues
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    m_min, m_max, size = 1e+10, 1e+16, 1000
    halo_cat = colossus_mass_sampler(redshift=0.1, model='sheth99',
                                     mdef='fof', m_min=m_min, m_max=m_max,
                                     cosmology=WMAP9, sigma8=0.8, ns=1.,
                                     size=size, resolution=1000)
    h_z = np.linspace(0.01, 0.1, len(halo_cat))
    z_range = [min(h_z), max(h_z)]
    skyarea = 600*u.deg**2

    alpha, beta, gamma, x = 1.91, 0.39, 0.1, 3
    ID_halo, sub_masses, ID_sub, z_sub = gen_sub_cat(halo_cat, h_z, alpha, beta, gamma, x)

    m_star1, phi_star1, alpha1 = 10**(10.58), 10**(-2.77), -0.33
    m_star2, phi_star2, alpha2 = 10**(10.64), 10**(-4.24), -1.54
    m_star3, phi_star3, alpha3 = 10**(10.65), 10**(-2.98), -1.48
    m_star4, phi_star4, alpha4 = 10**(10.55), 10**(-3.96), -1.53

    m_min1 = 10**(6.7)
    m_min2 = 10**(6.6)
    m_min3 = 10**(7.0)
    m_min4 = 10**(7.0)

    m_max1 = 10**(14)
    m_max2 = 10**(13)

    rc_cat = schechter_smf(z_range, m_star1, phi_star1, alpha1, m_min1, m_max1,
                           skyarea, cosmo, noise=False)[1]
    rs_cat = schechter_smf(z_range, m_star2, phi_star2, alpha2, m_min2, m_max2,
                           skyarea, cosmo, noise=False)[1]
    bc_cat = schechter_smf(z_range, m_star3, phi_star3, alpha3, m_min3, m_max1,
                           skyarea, cosmo, noise=False)[1]
    bs_cat = schechter_smf(z_range, m_star4, phi_star4, alpha4, m_min4, m_max2,
                           skyarea, cosmo, noise=False)[1]

    # Quench the halos
    h_quench = quenching_funct(halo_cat, 10**(12), 0.4)
    s_quench = quenching_funct(sub_masses, 10**(12), 0.4, 0.4)

    # Order the arrays
    halo_subhalo = np.concatenate((halo_cat, sub_masses), axis=0)  # Halos
    ID_list = np.concatenate((ID_halo, ID_sub), axis=0)  # IDs
    z_list = np.concatenate((h_z, z_sub), axis=0)  # Redshifts
    q_list = np.concatenate((h_quench, s_quench), axis=0)  # Quenching

    stack = np.stack((halo_subhalo, ID_list, z_list, q_list), axis=1)
    order1 = stack[np.argsort(stack[:, 0])]
    hs_order = np.flip(order1[:, 0])
    id_hs = np.flip(order1[:, 1])
    z_hs = np.flip(order1[:, 2])
    qu_hs = np.flip(order1[:, 3])
    rc_order = np.flip(np.sort(rc_cat))  # Galaxies
    rs_order = np.flip(np.sort(rs_cat))
    bc_order = np.flip(np.sort(bc_cat))
    bs_order = np.flip(np.sort(bs_cat))

    # Assign the galaxies
    hs_fin, gal_fin, id_fin, z_fin, gal_type_fin = assignment(hs_order, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs)

    # Function
    sham_rc, sham_rs, sham_bc, sham_bs, sham_cen, sham_sub = sham_plots(hs_fin, gal_fin,
                                                                        gal_type_fin)

    # Arrays correct type and size
    assert type(sham_rc) is type(sham_rs) is type(sham_bc) is type(sham_bs) is np.ndarray
    assert sham_rc.ndim == sham_rs.ndim == sham_bc.ndim == sham_bs.ndim == 2
    assert sham_cen.ndim == sham_sub.ndim == 2
    assert sham_rc.size  # Check lists are not empty
    assert sham_rs.size
    assert sham_bc.size
    assert sham_bs.size
    assert sham_cen.size
    assert sham_sub.size

    assert z_fin.size

    # Check number of centrals and satellites correct
    no_cen = len(np.where(id_fin < 0)[0])
    no_sub = len(np.where(id_fin > 0)[0])
    assert len(sham_cen[:, 0]) == no_cen
    assert len(sham_sub[:, 0]) == no_sub

    # Check values correct
    assert np.all(np.isin(sham_rc[:, 0], hs_fin))
    assert np.all(np.isin(sham_rs[:, 0], hs_fin))
    assert np.all(np.isin(sham_bc[:, 0], hs_fin))
    assert np.all(np.isin(sham_bs[:, 0], hs_fin))

    assert np.all(np.isin(sham_rc[:, 1], gal_fin))
    assert np.all(np.isin(sham_rs[:, 1], gal_fin))
    assert np.all(np.isin(sham_bc[:, 1], gal_fin))
    assert np.all(np.isin(sham_bs[:, 1], gal_fin))

    assert np.all(np.isin(sham_cen[:, 0], hs_fin))
    assert np.all(np.isin(sham_sub[:, 0], hs_fin))
    assert np.all(np.isin(sham_cen[:, 1], gal_fin))
    assert np.all(np.isin(sham_sub[:, 1], gal_fin))

    # Check errors trigger
    with pytest.raises(Exception) as excinfo:
        hs_test = np.linspace(min(hs_order), max(hs_order), 10)
        sham_plots(hs_test, gal_fin, gal_type_fin)
        assert str(excinfo.value) == 'All arrays must be the same shape'

    with pytest.raises(Exception) as excinfo:
        gal_type_test = np.random.randint(1, 5, 10)
        sham_plots(hs_fin, gal_fin, gal_type_test)
        assert str(excinfo.value) == 'All arrays must be the same shape'

    with pytest.raises(Exception) as excinfo:
        hs_test = -1*(hs_fin)
        sham_plots(hs_test, gal_fin, gal_type_fin)
        assert str(excinfo.value) == 'Halo masses must be positive and non-zero'

    with pytest.raises(Exception) as excinfo:
        gal_test = -1*(gal_fin)
        sham_plots(hs_fin, gal_test, gal_type_fin)
        assert str(excinfo.value) == 'Galaxy masses must be positive and non-zero'


@pytest.mark.skipif(not HAS_COLOSSUS, reason='test requires colossus')
@pytest.mark.flaky
def test_run_sham():
    from astropy.cosmology import FlatLambdaCDM
    from skypy.halos.sham import run_sham

    # Parameters
    h_file = 'test_halo.yaml'
    gal_param = np.array([[10**(10.58), 10**(-2.77), -0.33], [10**(10.64), 10**(-4.24), -1.54],
                         [10**(10.65), 10**(-2.98), -1.48], [10**(10.55), 10**(-3.96), -1.53]])
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    z_range = [0.01, 0.1]
    skyarea = 60
    qu_h = np.array([10**(12.16), 0.45])
    qu_s = np.array([10**(12.16), 0.45, 0.5])

    # Function
    sham = run_sham(h_file, gal_param, cosmology, z_range, skyarea, qu_h, qu_s,
                    sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
                    print_out=False, run_anyway=True)

    # Check arrays
    assert type(sham['Halo mass']) is np.ndarray
    assert type(sham['Galaxy mass']) is np.ndarray
    assert type(sham['Galaxy type']) is np.ndarray
    assert type(sham['ID value']) is np.ndarray
    assert type(sham['Redshift']) is np.ndarray

    assert len(sham['Halo mass']) == len(sham['Galaxy mass'])
    assert len(sham['Galaxy mass']) == len(sham['Galaxy type'])
    assert len(sham['Galaxy type']) == len(sham['ID value'])
    assert len(sham['ID value']) == len(sham['Redshift'])

    # Check errors trigger
    with pytest.raises(Exception) as excinfo:
        h_f = 3
        run_sham(h_f, gal_param, cosmology, z_range, skyarea, qu_h, qu_s,
                 sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
                 print_out=False, run_anyway=True)
        assert str(excinfo.value) == 'Halo YAML file must be provided as a string'

    with pytest.raises(Exception) as excinfo:
        gal_p = [[1, 2, 3], [4, 5, 6]]
        run_sham(h_file, gal_p, cosmology, z_range, skyarea, qu_h, qu_s,
                 sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
                 print_out=False, run_anyway=True)
        assert str(excinfo.value) == 'The wrong number of galaxies are in galaxy parameters'

    with pytest.raises(Exception) as excinfo:
        gal_p = [[1, 2], [4, 5], [7, 8], [3, 4]]
        run_sham(h_file, gal_p, cosmology, z_range, skyarea, qu_h, qu_s,
                 sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
                 print_out=False, run_anyway=True)
        assert str(excinfo.value) == 'The wrong number of galaxy parameters have been provided'

    with pytest.raises(Exception) as excinfo:
        gal_p = [[1, 2], [4, 5], [7, 8]]
        run_sham(h_file, gal_p, cosmology, z_range, skyarea, qu_h, qu_s,
                 sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
                 print_out=False, run_anyway=True)
        assert str(excinfo.value) == 'Supplied galaxy parameters are not the correct shape'

    with pytest.raises(Exception) as excinfo:
        qu_ht = [1]
        run_sham(h_file, gal_param, cosmology, z_range, skyarea, qu_ht, qu_s,
                 sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
                 print_out=False, run_anyway=True)
        assert str(excinfo.value) == 'Provided incorrect number of halo quenching parameters'

    with pytest.raises(Exception) as excinfo:
        qu_st = [1]
        run_sham(h_file, gal_param, cosmology, z_range, skyarea, qu_h, qu_st,
                 sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
                 print_out=False, run_anyway=True)
        assert str(excinfo.value) == 'Provided incorrect number of halo quenching parameters'
