'''Subhalo Abundance Matching module

This module generates halos and galaxies using SkyPy from user defined
parameters, decides which halos will be assigned which galaxies and then
outputs the matched arrays

Models
======
.. autosummary::
   :nosignatures:
   :toctree: ../api/

   quenching_funct
   find_min
   run_file
   gen_sub_cat
   assignment
   scatter_proxy
   sham_plots
   run_sham
'''

# Imports
import numpy as np
from skypy.pipeline import Pipeline
from skypy.halos import mass  # Vale and Ostriker 2004 num of subhalos
from skypy.galaxies import schechter_smf
from time import time
from scipy.special import erf  # Error function for quenching
from scipy.integrate import trapezoid as trap  # Integration of galaxies
from astropy import units as u

try:
    import colossus  # noqa F401
except ImportError:
    HAS_COLOSSUS = False
else:
    HAS_COLOSSUS = True

__all__ = [
    'quenching_funct',
    'find_min',
    'run_file',
    'gen_sub_cat',
    'assignment',
    'scatter_proxy',
    'sham_plots',
    'run_sham',
 ]

# General functions


# Quenching function
def quenching_funct(mass, M_mu, sigma, baseline=0):
    r'''Quenching function applied to halos
    This function computes the fraction of quenched halos (halos assigned
    quenched galaxies) as a function of mass using the parameterisation found
    in [1]_ and returns a truth list.

    Parameters
    -----------
    mass : (nm,) array_like
        Array for the halo mass, in units of solar mass.
    M_mu : float
        The mean of the error function, or the mass at which half of the halos
        are quenched, in units of solar mass.
    sigma : float
        The standard deviation of the error function, controls the width of
        the transition from blue to red galaxies
    baseline : float
        The initial value of the error function, or the fraction quenched at
        their lowest mass. For central halos this should be zero but for
        subhalos this will be non zero.

    Returns
    --------
    truth_array: (nm,) array_like
        Array which sets which halos are quenched

    Examples
    ---------
    >>> import numpy as np
    >>> from skypy.halos import sham

    This example will generate a central halo catalogue from a yaml file and
    then pass it through the quenching function to find which halos are
    quenched, following the mass distribution of the quenching function:

    >>> #Parameters
    >>> M_mu, sigma, base = 10**(12), 0.4, 0.3
    >>> h_mass = sham.run_file('halo.yaml', table1 = 'halo', info1 = 'mass')
    >>> h_quench = sham.quenching_funct(h_mass, M_mu, sigma, base)


    References
    ----------
    .. [1] Peng Y.-j., et al., 2010, Astrophysical Journal, 721, 193
    '''
    mass = np.atleast_1d(mass)

    # Errors
    if M_mu <= 0:
        raise Exception('M_mu cannot be negative or 0 solar masses')
    if sigma <= 0:
        raise Exception('sigma cannot be negative or 0')
    if baseline < 0:
        baseline = -baseline
        raise Warning('baseline cannot be negative, set_baseline = -baseline')
    if baseline > 1:
        raise Exception('baseline cannot be more than 1')
    if len(mass) == 1:
        raise TypeError('Mass must be array-like')

    # Only works with an increasing array due to normalisation?
    qu_id = np.arange(0, len(mass))  # Random order elements (ie halo original order)

    # Elements of sorted halos
    stack1 = np.stack((mass, qu_id), axis=1)
    order1 = stack1[np.argsort(stack1[:, 0])]  # Order on the halos
    mass = order1[:, 0]
    order_qu_id = order1[:, 1]

    calc = np.log10(mass/M_mu)/sigma
    old_prob = 0.5 * (1.0 + erf(calc/np.sqrt(2)))  # Base error funct, 0 -> 1
    add_val = baseline/(1-baseline)
    prob_sub = old_prob + add_val  # Add value to plot to get baseline
    prob_array = prob_sub/prob_sub[-1]  # Normalise, base -> 1
    rand_array = np.random.uniform(size=len(mass))
    qu_arr = rand_array < prob_array

    # Reorder truth to match original order
    stack2 = np.stack((order_qu_id, qu_arr), axis=1)
    order2 = stack2[np.argsort(stack2[:, 0])]
    return order2[:, 1]


# Find minimum galaxy mass for given halo population through the integral
def find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo,
             print_out=False, run_anyway=False):
    r'''Function to find a minimum mass for a galaxy type for a given number
    of halos

    This function computes a look up table of the integral of the galaxy
    Schechter function and interpolates on the number of halos to find the
    minimum mass.

    Parameters
    -----------
    m_star : float
        Exponential tail off of Schechter function, units of solar mass
    phi_star : float
        Normalisation for the Schechter function, units of Mpc^{-3}
    alpha : float
        Power law of Schechter function
    cosmology : Cosmology
        Astropy cosmology object for calculating comoving volume
    z_range : (2, ), array_like
        Minimum and maximum redshift of galaxies
    skyarea : float
        Sky area galaxies are 'observed' from, units of deg^2
    max_mass : float
        Maximum mass to integrate Schechter function
    no_halo : float
        Number of halos to match to given galaxy function
    run_anyway (debug): Boolean
        True/False statement to determine if function runs outside of integral
        look up table

    Returns
    --------
    min_mass_log: float
        Log10 of the minimum mass returned from interpolation

    Examples
    ---------
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.halos import sham

    This example uses the Schechter parameters for red centrals from [1]_ and
    a number of halos these might be paired with to generate a minimum mass:

    >>> #Define input variables
    >>> m_star, phi_star, alpha = 10**(10.75), 10**(-2.37), -0.18
    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    >>> z_range = [0.01, 0.1]
    >>> skyarea = 600
    >>> max_mass = 10**(14)
    >>> no_halo = 10124
    >>> #Run function
    >>> min_mass = sham.find_min(m_star, phi_star, alpha, cosmology,
    ...                          z_range, skyarea, max_mass, no_halo)


    References
    ----------
    .. [1] Weigel A. K., Schawinski K., Bruderer C., 2016, Monthly Notices of the
        Royal Astronomical Society, 459, 2150
    '''
    z_range = np.atleast_1d(z_range)

    # Errors
    if z_range.shape != (2, ):
        raise Exception('The wrong number of redshifts were given')
    if z_range[0] < 0 or z_range[1] < 0:
        raise Exception('Redshift cannot be negative')
    if z_range[1] <= z_range[0]:
        raise Exception('The second redshift should be more than the first')
    if alpha >= 0:
        alpha = -1*alpha
        raise Warning('Schechter function defined so alpha < 0, set_alpha = -alpha')
    if m_star <= 0 or phi_star <= 0:
        raise Exception('M* and phi* must be positive and non-zero numbers')
    if skyarea <= 0:
        raise Exception('The skyarea must be a positive non-zero number')
    if max_mass <= 10**6:
        raise Exception('The maximum mass must be more than 10^6 solar masses')

    skyarea = skyarea*u.deg**2
    z = np.linspace(z_range[0], z_range[1], 1000)
    dVdz = (cosmology.differential_comoving_volume(z)*skyarea).to_value('Mpc3')

    # Other
    mass_mins = 10**np.arange(6, 10, 0.1)  # Minimums to integrate, linear in logspace
    phi_m = phi_star/m_star
    # Want all integral bins to have same resolution in log space, should give accurate integral
    res = 10**(-2.3)

    # Integrate
    int_vals = []
    for ii in mass_mins:
        mass_bin = 10**np.arange(np.log10(ii), np.log10(max_mass), res)  # Masses to integrate over
        m_m = mass_bin/m_star
        dndmdV = phi_m*np.e**(-m_m)*(m_m)**alpha  # Mass function
        dndV = trap(dndmdV, mass_bin)
        dn = dndV*dVdz
        int_vals.append(trap(dn, z))  # Integral per mass bin

    int_vals = np.array(int_vals)
    # Integral must be increasing so flip arrays
    min_mass = np.interp(no_halo, np.flip(int_vals), np.flip(mass_mins))

    # Region within Poisson error
    poisson_min = min(int_vals) - np.sqrt(min(int_vals))
    poisson_max = max(int_vals) + np.sqrt(max(int_vals))

    if (no_halo < poisson_min or no_halo > poisson_max) and not run_anyway:
        # Interpolation falls outside of created range
        raise Exception('Number of halos not within reachable bounds')

    elif (no_halo < poisson_min or no_halo > poisson_max) and run_anyway:
        if print_out:
            print('Outside of interpolation, but running anyway')
        return 10**(7)

    return min_mass


# Generate catalogues
def run_file(file_name, table1, info1, info2=None):
    r'''Function that runs a yaml file to generate a catalogue

    This function runs a yaml file using the SkyPy pipeline and
    produces a catalogue from the file.

    Parameters
    -----------
    file_name : str
        String of file name to be run
    table1 : str
        String of table to access from file
    info1 : str
        String of variable to access mass, this must be provided
    info2 : str
        String of variable to access redshift

    Returns
    --------
    catalogue: (nm, ) array_like
        List of masses
    redshifts: (nm, ) array_like
        List of redshifts generated for each mass, if requested

    Examples
    ---------
    >>> from skypy.halos import sham

    This example uses a yaml file for a galaxy type and returns a
    galaxy catalogue:

    >>> galaxy_cat = sham.run_file('red_central.yaml', 'galaxy', 'sm')

    This example uses a yaml file for halos and returns a
    halo catalogue and their generated redshifts:

    >>> halo_cat, h_z = sham.run_file('halo.yaml', 'halo', 'mass', 'z')

    References
    ----------
    ..
    '''
    import os
    # Errors
    if type(file_name) is not str:
        raise Exception('File name must be a string')
    if not os.path.exists(file_name):
        raise Exception('File does not exist')

    # Use pipeline
    pipe = Pipeline.read(file_name)
    pipe.execute()

    # Get information
    info = pipe[table1]
    cat = info[info1]
    if info2 is not None:
        z = info[info2]
        return cat, z

    return cat


# Generate subhalo catalogues
def gen_sub_cat(parent_halo, z_halo, sub_alpha, sub_beta, sub_gamma, sub_x):
    r'''Function that generates a catalogue of subhalos from a population
    of parent halos

    This function uses the conditional mass function and occupation number
    from [1]_ to generate a catalogue of subhalos for each parent halo
    in the halo catalogue

    Parameters
    -----------
    parent_halo : (nm, ), array_like
        Parent halo catalogue used to generate subhalos, units
        of solar mass
    z_halo : (nm, ), array_like
        Redshifts of parent halos
    sub_alpha : float
        Low mass power law slope of conditional mass function
    sub_beta : float
        Exponential cutoff of subhalo masses and a fraction of the
        parent halo
    sub_gamma : float
        Present day mass fraction of parent halo in sum of generated
        subhalo mass
    sub_x : float
        Factor that allows for stripping of mass for current subhalos,
        x = 1 is the current day stripped mass, x > 1 for unstripped
        subhalo mass

    Returns
    --------
    ID_halo: (nm, ), array_like
        ID values for parent halos
    sub_masses: (ns, ), array_like
        Subhalo masses for parent halos
    ID_sub: (ns, ), array_like
        ID values for subhalos to assign them to their parent halos
    z_sub: (ns, ), array_like
        Redshifts of subhalos (same as their parent halo)

    Examples
    ---------
    >>> from skypy.halos import sham

    This example generates a halo catalogue and then creates a
    subhalo catalogue:

    >>> halo_cat, h_z = sham.run_file('halo.yaml', 'halo', 'mass', 'z')
    >>> #Variables
    >>> a, b, c, x = 1.91, 0.39, 0.1, 3
    >>> ID_halo, sub_masses, ID_sub, z_sub = sham.gen_sub_cat(halo_cat, h_z,
    ...                                                       a, b, c, x)


    References
    ----------
    .. [1] Vale A., Ostriker J. P., 2004, Monthly Notices of the Royal
       Astronomical Society, 353, 189
    '''
    parent_halo = np.atleast_1d(parent_halo)
    z_halo = np.atleast_1d(z_halo)

    # Errors
    if len(parent_halo) != len(z_halo):
        raise Exception('Catalogue of halos and redshifts must be the same length')
    if len(np.where(parent_halo <= 0)[0]) > 0:
        raise Exception('Masses in catalogue should be positive and non-zero')
    if len(np.where(z_halo < 0)[0]) > 0:
        raise Exception('Redshifts in catalogue should be positive')
    if sub_alpha < 0:
        sub_alpha = -1*sub_alpha
        raise Warning('Subhalo mass function defined alpha > 0, set_alpha = -alpha')
    if sub_alpha >= 2:
        raise Exception('Subhalo alpha must be less than 2')
    if sub_x < 1:
        raise Exception('Subhalo x cannot be less than 1')
    if sub_beta <= 0 or sub_beta > 1:
        raise Exception('Subhalo beta must be between 0 and 1')
    if sub_gamma < 0 or sub_gamma > 1:
        raise Exception('Subhalo gamma must be between 0 and 1')

    if parent_halo.size == 1:  # ie only one halo is provided
        m_min = 10**(10)
    else:
        m_min = min(parent_halo)  # Min mass of subhalo to generate (RESOLUTION OF PARENT HALOS)
    ID_halo = -1*np.arange(1, len(parent_halo)+1, dtype=int)  # Halo IDs

    # Get list of halos that will have subhalos
    halo_to_sub = parent_halo[np.where(parent_halo >= 10**(10))]  # TODO change to a larger number?
    ID_to_sub = -1*ID_halo[np.where(parent_halo >= 10**(10))]
    z_to_sub = z_halo[np.where(parent_halo >= 10**(10))]

    # Get subhalos
    no_sub = mass.number_subhalos(halo_to_sub, sub_alpha, sub_beta,
                                  sub_gamma, sub_x, m_min, noise=True)
    sub_masses = mass.subhalo_mass_sampler(halo_to_sub, no_sub, sub_alpha, sub_beta, sub_x, m_min)

    # Assign subhalos to parents by ID value
    ID_sub = []
    z_sub = []
    ID_count = 0
    for ii in no_sub:
        ID_set = ID_to_sub[ID_count]
        ID_sub.extend(ID_set*np.ones(ii, dtype=int))  # Positive ID for subhalo
        z_sub.extend(z_to_sub[ID_count]*np.ones(ii))  # Same redshift as the parent
        ID_count += 1

    # Delete any generated subhalos smaller than the resolution
    del_sub = np.where(sub_masses < m_min)[0]
    sub_masses = np.delete(sub_masses, del_sub)
    ID_sub = np.delete(ID_sub, del_sub)
    z_sub = np.delete(z_sub, del_sub)

    return ID_halo, sub_masses, ID_sub, z_sub


# Assignment
def assignment(hs_order, rc_order, rs_order, bc_order, bs_order, qu_order, id_order, z_order,
               scatter=False):
    r'''Function that assigns galaxies to halos based on type and the
    quenching function

    This function runs through the halo catalogue, assesses whether it has a
    quenched galaxy and if it is a central or satellite, and assigns the next
    galaxy in the appropriate list. This therefore creates an ordered array
    of halos and galaxies which are assigned to each other. Unassigned halos
    and galaxies are deleted or ignored.

    Parameters
    -----------
    hs_order : (nm, ) array_like
        Ordered (most massive first) array of central and subhalo masses,
        in units of solar mass
    rc_order : (ng1, ) array_like
        Ordered (most massive first) array of red central galaxy masses,
        in units of solar mass
    rs_order : (ng2, ) array_like
        Ordered (most massive first) array of red satellite galaxy masses,
        in units of solar mass
    bc_order : (ng3, ) array_like
        Ordered (most massive first) array of blue central galaxy masses,
        in units of solar mass
    bs_order : (ng4, ) array_like
        Ordered (most massive first) array of blue satellite galaxy masses,
        in units of solar mass
    qu_order : (nm, ) array_like
        Truth array of which halos should be assigned quenched (red) galaxies
    id_order : (nm, ) array_like
        Array of IDs to determine if the halo is central or satellite
    z_order : (nm, ) array_like
        Redshifts of generated halos
    scatter : Boolean
        True/false if the galaxies lists have undergone scatter

    Returns
    --------
    hs_fin: (nh, ) array_like
        List of assigned halo masses, in units of solar masses
    gal_fin: (nh, ) array_like
        List of assigned galaxies, in units of solar mass
    id_fin: (nh, ) array_like
        List of ID values for halos
    z_fin: (nh, ) array_like
        List of redshifts for halos
    gal_type_fin: (nh, ) array_like
        List of assigned galaxy types: 'red central', 'red satellite',
        'blue central' and 'blue satellite'

    Examples
    ---------
    >>> from skypy.halos import sham
    >>> from skypy.galaxies import schechter_smf
    >>> from astropy.cosmology import FlatLambdaCDM

    This example generates the required catalogues, assigns which halos will
    be given a quenched galaxy and then assigns them. Assume the parameters
    are the same as previous examples or come from the same work.

    >>> #Generate the catalogues
    >>> halo_cat, h_z = sham.run_file('halo.yaml', 'halo', 'mass', 'z')
    >>> a, b, c, x = 1.91, 0.39, 0.1, 3
    >>> ID_halo, sub_masses, ID_sub, z_sub = sham.gen_sub_cat(halo_cat, h_z,
    ...                                                       a, b, c, x)
    >>> #Sky parameters
    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    >>> z_range = [min(h_z), max(h_z)]
    >>> skyarea = 100
    >>> #Galaxy parameters
    >>> m_star1, phi_star1, alpha1 = 10**(10.75), 10**(-2.37), -0.18
    >>> m_star2, phi_star2, alpha2 = 10**(10.72), 10**(-2.66), -0.71
    >>> m_star3, phi_star3, alpha3 = 10**(10.59), 10**(-2.52), -1.15
    >>> m_star4, phi_star4, alpha4 = 10**(10.59), 10**(-3.09), -1.31
    >>> min_mass = 10**(7)
    >>> max_mass = 10**(14)
    >>> #Generate the galaxies
    >>> rc_cat = schechter_smf(z_range, m_star1, phi_star1, alpha1, min_mass,
                           max_mass, skyarea*u.deg**2, cosmology)[1]
    >>> rs_cat = schechter_smf(z_range, m_star2, phi_star2, alpha2, min_mass,
                           max_mass, skyarea*u.deg**2, cosmology)[1]
    >>> bc_cat = schechter_smf(z_range, m_star3, phi_star3, alpha3, min_mass,
                           max_mass, skyarea*u.deg**2, cosmology)[1]
    >>> bs_cat = schechter_smf(z_range, m_star4, phi_star4, alpha4, min_mass,
                           max_mass, skyarea*u.deg**2, cosmology)[1]
    >>> #Quench the halos
    >>> m_mu, sigma, base = 10**(12), 0.4, 0.4
    >>> h_quench = sham.quenching_funct(halo_cat, m_mu, sigma)
    >>> s_quench = sham.quenching_funct(sub_masses, m_mu, sigma, base)
    >>> #Order the arrays
    >>> halo_subhalo = np.concatenate((halo_cat, sub_masses), axis=0)
    >>> ID_list = np.concatenate((ID_halo, ID_sub), axis=0)
    >>> z_list = np.concatenate((h_z, z_sub), axis=0)
    >>> q_list = np.concatenate((h_quench, s_quench), axis=0)
    >>> stack = np.stack((halo_subhalo, ID_list, z_list, q_list), axis=1)
    >>> order1 = stack[np.argsort(stack[:,0])]
    >>> hs_order = np.flip(order1[:,0])
    >>> id_hs = np.flip(order1[:,1])
    >>> z_hs = np.flip(order1[:,2])
    >>> qu_hs = np.flip(order1[:,3])
    >>> rc_order = np.flip(np.sort(rc_cat)) #Galaxies
    >>> rs_order = np.flip(np.sort(rs_cat))
    >>> bc_order = np.flip(np.sort(bc_cat))
    >>> bs_order = np.flip(np.sort(bs_cat))
    >>> #Call function
    >>> hs, gal, ID, z, gal_type = sham.assignment(hs_order, rc_order,
    ...                                            rs_order, bc_order,
    ...                                            bs_order, qu_hs,
    ...                                            id_hs, z_hs)

    References
    ----------
    ..
    '''
    hs_order = np.atleast_1d(hs_order)
    rc_order = np.atleast_1d(rc_order)
    rs_order = np.atleast_1d(rs_order)
    bc_order = np.atleast_1d(bc_order)
    bs_order = np.atleast_1d(bs_order)
    qu_order = np.atleast_1d(qu_order)
    id_order = np.atleast_1d(id_order)
    z_order = np.atleast_1d(z_order)

    # Order check
    hs_order_check = np.diff(hs_order)
    rc_order_check = np.diff(rc_order)
    rs_order_check = np.diff(rs_order)
    bc_order_check = np.diff(bc_order)
    bs_order_check = np.diff(bs_order)

    # Shape check
    hs_shape = hs_order.shape
    qu_shape = qu_order.shape
    id_shape = id_order.shape
    z_shape = z_order.shape

    # Errors
    if ((hs_order <= 0)).any():
        raise Exception('Halo masses must be positive and non-zero')
    if ((rc_order <= 0)).any() or ((rs_order <= 0)).any():
        raise Exception('Galaxy masses must be positive and non-zero')
    if ((bc_order <= 0)).any() or ((bs_order <= 0)).any():
        raise Exception('Galaxy masses must be positive and non-zero')
    if ((hs_order_check > 0)).all():
        hs_order = np.flip(hs_order)
        raise Warning('Halo masses were in the wrong order and have been correct')
    elif ((hs_order_check > 0)).any():
        raise Exception('Halos are not in a sorted order')
    if hs_shape != qu_shape or qu_shape != id_shape or id_shape != z_shape:
        raise Exception('All arrays pertaining to halos must be the same shape')

    # Only check ordering if not scattered
    if not scatter:
        if ((rc_order_check > 0)).all():
            rc_order = np.flip(rc_order)
            raise Warning('Red central galaxies were in wrong order now correct')
        elif ((rc_order_check > 0)).any():
            raise Exception('Red central galaxies are not in a sorted order')
        if ((rs_order_check > 0)).all():
            rs_order = np.flip(rs_order)
            raise Warning('Red satellite galaxies were in wrong order now correct')
        elif ((rs_order_check > 0)).any():
            raise Exception('Red satellite galaxies are not in a sorted order')
        if ((bc_order_check > 0)).all():
            bc_order = np.flip(bc_order)
            raise Warning('Blue central galaxies were in wrong order and now correct')
        elif ((bc_order_check > 0)).any():
            raise Exception('Blue central galaxies are not in a sorted order')
        if ((bs_order_check > 0)).all():
            bs_order = np.flip(bs_order)
            raise Warning('Blue satellite galaxies were in wrong order and now correct')
        elif ((bs_order_check > 0)).any():
            raise Exception('Blue satellite galaxies are not in a sorted order')

    # Assign galaxies to halos
    del_ele = []  # Elements to delete later
    gal_assigned = []  # Assigned galaxies
    gal_type_A = []  # Population assigned

    # Total number of galaxies
    gal_num = len(rc_order) + len(rs_order) + len(bc_order) + len(bs_order)

    rc_counter = 0  # Counters for each population
    rs_counter = 0
    bc_counter = 0
    bs_counter = 0

    for ii in range(len(hs_order)):
        qu = qu_order[ii]
        total_counter = rc_counter + rs_counter + bc_counter + bs_counter
        ID_A = id_order[ii]

        if total_counter == gal_num:  # All galaxies assigned
            del_array = np.arange(ii, len(hs_order))
            del_ele.extend(del_array)
            break

        if qu == 1:  # Halo is quenched
            if ID_A < 0 and rc_counter != len(rc_order):  # Halo assigned mass quenched
                gal_assigned.append(rc_order[rc_counter])
                gal_type_A.append('red central')
                rc_counter += 1

            elif ID_A > 0 and rs_counter != len(rs_order):  # Subhalo assigned environment quenched
                gal_assigned.append(rs_order[rs_counter])
                gal_type_A.append('red satellite')
                rs_counter += 1

            else:  # No red to assign
                del_ele.append(ii)  # Delete unassigned halos

        else:  # Halo not quenched
            if ID_A < 0 and bc_counter != len(bc_order):  # Halo assigned blue central
                gal_assigned.append(bc_order[bc_counter])
                gal_type_A.append('blue central')
                bc_counter += 1

            elif ID_A > 0 and bs_counter != len(bs_order):  # Subhalo assigned blue satellite
                gal_assigned.append(bs_order[bs_counter])
                gal_type_A.append('blue satellite')
                bs_counter += 1

            else:  # No blue to assign
                del_ele.append(ii)

    # Delete and array final lists
    hs_fin = np.delete(hs_order, del_ele)
    id_fin = np.delete(id_order, del_ele)
    z_fin = np.delete(z_order, del_ele)
    gal_fin = np.array(gal_assigned)
    gal_type_fin = np.array(gal_type_A)

    return hs_fin, gal_fin, id_fin, z_fin, gal_type_fin


def scatter_proxy(gal):
    r'''Function to add scatter to galaxies using a proxy mass

    This function takes the generated galaxy masses from the catalogues
    and generates a gaussian proxy mass for each, then reorders based on
    this proxy mass. This adds scatter to the galaxies.

    Parameters
    -----------
    gal : (nm, ) array_like
        List of galaxy masses to be scattered, in units of solar mass

    Returns
    --------
    new_gal : (nm, ) array_like
        Reordered galaxy masses according to generated proxy mass, highest
        to lowest mass, in units of solar mass

    Examples
    ---------
    >>> from skypy.halos import sham

    This example generates a galaxy catalogue and then reorders it

    >>> gal_cat = sham.run_file('galaxies.yaml', galaxy, mass)
    >>> new_gal = sham.scatter_proxy(gal_cat)

    References
    ----------
    ..
    '''
    # Errors
    gal = np.atleast_1d(gal)
    if ((gal <= 0)).any():
        raise Exception('Galaxy masses must be positive and non-zero')

    # Draw one sample for each galaxy
    # Use galaxy mass as mean and sigma of one order less
    proxy_gal = np.random.normal(loc=gal, scale=gal/10)

    # Order by proxy mass
    proxy_stack = np.stack((proxy_gal, gal), axis=1)
    gal_stack = proxy_stack[np.argsort(proxy_stack[:, 0])]
    return np.flip(gal_stack[:, 1])  # Return new galaxies, highest mass first


# Separate populations
def sham_plots(hs_fin, gal_fin, gal_type_fin, print_out=False):
    r'''Function that pulls assigned galaxies together into groups for
    plotting

    This function takes in the final assigned halos and galaxies and splits
    them into each type of galaxy for plotting and comparing to data. It also
    computes the separate central and satellite populations with no colour
    split.

    Parameters
    -----------
    hs_fin : (nm, ) array_like
        List of assigned halos and subhalos, in units of solar mass. The
        subhalos are assumed to be their current stripped mass
    gal_fin : (nm, ) array_like
        List of assigned galaxies, in units of solar mass
    gal_type_fin : (nm, ) array_like
        List of assigned galaxies types with the tags
        'red central', 'red satellite', 'blue central', 'blue satellite'
    print_out : Boolean
        True/false whether to output print statements for progress and
        timings

    Returns
    --------
    sham_rc: (nh1, 2) array_like
        Stacked array of halos ([:,0]) and galaxies ([:,1]) containing
        red centrals
    sham_rs: (nh2, 2) array_like
        Stacked array of halos ([:,0]) and galaxies ([:,1]) containing
        red satellites
    sham_bc: (nh3, 2) array_like
        Stacked array of halos ([:,0]) and galaxies ([:,1]) containing
        blue centrals
    sham_bs: (nh4, 2) array_like
        Stacked array of halos ([:,0]) and galaxies ([:,1]) containing
        blue satellites
    sham_cen: (nh5, 2) array_like
        Stacked array of halos ([:,0]) and galaxies ([:,1]) containing
        centrals
    sham_sub: (nh6, 2) array_like
        Stacked array of halos ([:,0]) and galaxies ([:,1]) containing
        satellites

    Examples
    ---------
    >>> from skypy.halos import sham

    This example uses assigned halos and galaxies and then finds the
    separated SHAM arrays.

    >>> hs, gal, id, z, gal_type = sham.assignment(hs_order, rc_order,
    ...                                            rs_order, bc_order,
    ...                                            bs_order, qu_order,
    ...                                            id_order, z_order)
    >>> rc, rs, bc, bs, cen, sub = sham.sham_plots(hs, gal, gal_type)

    References
    ----------
    ..
    '''
    # Errors
    if hs_fin.shape != gal_fin.shape or gal_fin.shape != gal_type_fin.shape:
        raise Exception('All arrays must be the same shape')
    if ((hs_fin <= 0)).any():
        raise Exception('Halo masses must be positive and non-zero')
    if ((gal_fin <= 0)).any():
        raise Exception('Galaxy masses must be positive and non-zero')

    rc_dots = np.where(gal_type_fin == 'red central')[0]
    rs_dots = np.where(gal_type_fin == 'red satellite')[0]
    bc_dots = np.where(gal_type_fin == 'blue central')[0]
    bs_dots = np.where(gal_type_fin == 'blue satellite')[0]

    # SHAMs by galaxy population
    sham_rc = np.stack((hs_fin[rc_dots], gal_fin[rc_dots]), axis=1)
    sham_rs = np.stack((hs_fin[rs_dots], gal_fin[rs_dots]), axis=1)
    sham_bc = np.stack((hs_fin[bc_dots], gal_fin[bc_dots]), axis=1)
    sham_bs = np.stack((hs_fin[bs_dots], gal_fin[bs_dots]), axis=1)

    # Combine both sets of centrals
    halo_all = np.concatenate((sham_rc[:, 0], sham_bc[:, 0]), axis=0)  # All halos and galaxies
    gals_all = np.concatenate((sham_rc[:, 1], sham_bc[:, 1]), axis=0)

    sham_concat = np.stack((halo_all, gals_all), axis=1)
    sham_order = sham_concat[np.argsort(sham_concat[:, 0])]  # Order by halo mass
    sham_cen = np.flip(sham_order, 0)

    # Combine both sets of satellites
    halo_sub = np.concatenate((sham_rs[:, 0], sham_bs[:, 0]), axis=0)  # All halos and galaxies
    gals_sub = np.concatenate((sham_rs[:, 1], sham_bs[:, 1]), axis=0)

    sham_concats = np.stack((halo_sub, gals_sub), axis=1)
    sham_orders = sham_concats[np.argsort(sham_concats[:, 0])]  # Order by halo mass
    sham_sub = np.flip(sham_orders, 0)

    if print_out:
        print('Created SHAMs')

    return sham_rc, sham_rs, sham_bc, sham_bs, sham_cen, sham_sub


# Called SHAM function
def run_sham(h_file, gal_param, cosmology, z_range, skyarea, qu_param,
             sub_param=[1.91, 0.39, 0.1, 3], gal_max_h=10**(14), gal_max_s=10**(13),
             print_out=False, run_anyway=False, scatter_prox=False):
    r'''Function that takes all inputs for the halos and galaxies and runs
    a SHAM over them

    This function runs all the functions required to make a complete SHAM.
    It generates the catalogues from the input values, quenches the halos
    and assigns the galaxies. Then it outputs the sorted SHAM arrays by
    galaxy type.

    Parameters
    -----------
    h_file : str
        String for YAML file of halos to generate catalogue
    gal_param : (4, 3) array_like
        Parameters (M* (solar mass), phi* (Mpc^{-3}), alpha) for Schechter
        functions of four galaxy types (red centrals, red satellites,
        blue centrals, blue satellites)
    cosmology : Cosmology
        Astropy cosmology object, must specify H0 (km/s/Mpc), Om0 and astropy
        name. Should be the same as specified in the halo YAML file
    z_range : (2, ) array_like
        Minimum and maximum redshifts to generate for galaxies. Should be the
        same as specified in the halo YAML file
    skyarea : float
        Sky area galaxies are 'observed' from, units of deg^2. Should be the
        same as specified in the halo YAML file
    qu_param : (3, ) array_like
        The mean, standard deviation and baseline of the error function
        to quench the halos (see 'quenching_funct' for more detail)
    sub_param : (4, ) array_like
        alpha, beta, gamma and x parameters for subhalo generation. Defaults
        from [1]_ (see 'gen_sub_cat' for more detail)
    gal_max_h : float
        Maximum mass central galaxy to generate, units of solar mass
    gal_max_s : float
        Maximum mass satellite galaxy to generate, units of solar mass
    print_out : Boolean
        True/false whether to output print statements for progress and timings
    run_anyway : Boolean
        True/false whether to run the SHAM code even if the galaxy abundance
        does not match the number of halos
    scatter_prox : Boolean
        True/false whether to scatter the galaxies by a proxy mass when the
        catalogues are generated

    Returns
    --------
    sham_dict: (nm, 5) dictionary
        A dictionary containing the necessary outputs stored as numpy arrays.
        - 'Halo mass' parent and subhalo masses, in units of solar mass,
          the subhalos are their present stripped mass
        - 'Galaxy mass' assigned galaxy masses, in units of solar mass
        - 'Galaxy type' type of galaxy assigned, written as 'red central',
          'red satellite', 'blue central' and 'blue satellite'
        - 'ID value' ID number of halos and subhalos, negative for halos,
          positive for subhalos
        - 'Redshift' Redshift of halos/galaxies

    Examples
    ---------
    >>> from skypy.halos import sham
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> import numpy as np

    This example shows the format for each of the inputs and receives the
    outputs. Subhalo parameters are from [1]_, galaxy parameters are from [2]_
    and quenching parameters are approximately from [3]_

    >>> #Parameters
    >>> h_file = 'halo.yaml'
    >>> rc_p = [10**(10.75), 10**(-2.37), -0.18]
    >>> rs_p = [10**(10.72), 10**(-2.66), -0.71]
    >>> bc_p = [10**(10.59), 10**(-2.52), -1.15]
    >>> bs_p = [10**(10.59), 10**(-3.09), -1.31]
    >>> gal_param = [rc_p, rs_p, bc_p, bs_p]
    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    >>> z_range = np.array([0.01, 0.1])
    >>> skyarea = 600.
    >>> qu = np.array([10**(11.9), 0.4, 0.5])
    >>> #Run function
    >>> sham_dict = run_sham(h_file, gal_param, cosmology, z_range, skyarea,
    ...                      qu, sub_param = [1.91, 0.39, 0.1, 3],
    ...                      gal_max_h = 10**(14), gal_max_s = 10**(13),
    ...                      print_out=False, run_anyway=True,
                             scatter_prox=True)

    References
    ----------
    .. [1] Vale A., Ostriker J. P., 2004, Monthly Notices of the Royal
       Astronomical Society, 353, 189
    .. [2] Weigel A. K., Schawinski K., Bruderer C., 2016, Monthly Notices of the
       Royal Astronomical Society, 459, 2150
    .. [3] Peng Y.-j., et al., 2010, Astrophysical Journal, 721, 193
    '''
    sham_st = time()

    gal_param = np.atleast_1d(gal_param)
    qu_param = np.atleast_1d(qu_param)

    # Check that all inputs are of the correct type and size
    if type(h_file) is not str:
        raise Exception('Halo YAML file must be provided as a string')
    if gal_param.shape != (4, 3):
        if gal_param.shape[0] != 4:
            raise Exception('The wrong number of galaxies are in galaxy parameters')
        elif gal_param.shape[1] != 3:
            raise Exception('The wrong number of galaxy parameters have been provided')
        else:
            raise Exception('Supplied galaxy parameters are not the correct shape')
    if qu_param.shape != (3,):
        raise Exception('Provided incorrect number of quenching parameters')
    if z_range.shape != (2, ):
        raise Exception('The wrong number of redshifts were given')
    if z_range[0] < 0 or z_range[1] < 0:
        raise Exception('Redshift cannot be negative')
    if z_range[1] <= z_range[0]:
        raise Exception('The second redshift should be more than the first')

    # Check galaxy parameters are correct sign
    if np.any(np.array([gal_param[0][0], gal_param[1][0], gal_param[2][0], gal_param[3][0]]) <= 0):
        raise Warning('M* values must be positive and non-zero')
    if np.any(np.array([gal_param[0][1], gal_param[1][1], gal_param[2][1], gal_param[3][1]]) <= 0):
        raise Exception('phi* values must be positive and non-zero')
    if np.any(np.array([gal_param[0][2], gal_param[1][2], gal_param[2][2], gal_param[3][2]]) > 0):
        raise Warning('Galaxy Schechter function alphas must be < 0')
    if skyarea <= 0:
        raise Exception('The skyarea must be a positive non-zero number')
    if cosmology.name is None:
        raise Exception('Cosmology object must have an astropy cosmology name')

    # Generate parent halos from YAML file
    # TODO remove hard coded table/variable names
    h_st = time()
    parent_halo, z_halo = run_file(h_file, 'halo', 'mass', 'z')
    if print_out:
        print('Halo catalogues generated in', round((time() - h_st), 2), 's')

    # Generate subhalos and IDs
    sub_alpha = sub_param[0]  # Vale and Ostriker 2004 mostly
    sub_beta = sub_param[1]
    sub_gamma = sub_param[2]
    sub_x = sub_param[3]

    sub_tim = time()
    ID_halo, subhalo_m, ID_sub, z_sub = gen_sub_cat(parent_halo, z_halo, sub_alpha,
                                                    sub_beta, sub_gamma, sub_x)

    if print_out:
        print('Generated subhalos and IDs in', round((time() - sub_tim), 2), 's')
        print('')

    # Quench halos and subhalos
    M_mu = qu_param[0]  # Parameters
    sigma = qu_param[1]
    baseline = qu_param[2]

    h_quench = quenching_funct(parent_halo, M_mu, sigma, 0)  # Quenched halos
    sub_quench = quenching_funct(subhalo_m, M_mu, sigma, baseline)  # Quenched subhalos

    # Galaxy Schechter function parameters
    rc_param = gal_param[0]
    rs_param = gal_param[1]
    bc_param = gal_param[2]
    bs_param = gal_param[3]

    # Number of halos for each population
    rc_halo = len(np.where(h_quench == 1)[0])
    rs_halo = len(np.where(sub_quench == 1)[0])
    bc_halo = len(np.where(h_quench == 0)[0])
    bs_halo = len(np.where(sub_quench == 0)[0])

    # Find galaxy mass range (m_star, phi_star, alpha, tag)
    # TODO Add a way to import a look up table
    range1 = time()
    rc_min = find_min(rc_param[0], rc_param[1], rc_param[2], cosmology, z_range, skyarea,
                      gal_max_h, rc_halo, print_out, run_anyway)
    if print_out:
        print('Red central log(minimum mass)', np.log10(rc_min), 'in',
              round((time() - range1), 4), 's')

    range2 = time()
    rs_min = find_min(rs_param[0], rs_param[1], rs_param[2], cosmology, z_range, skyarea,
                      gal_max_s, rs_halo, print_out, run_anyway)
    if print_out:
        print('Red satellite log(minimum mass)', np.log10(rs_min), 'in',
              round((time() - range2), 4), 's')

    range3 = time()
    bc_min = find_min(bc_param[0], bc_param[1], bc_param[2], cosmology, z_range, skyarea,
                      gal_max_h, bc_halo, print_out, run_anyway)
    if print_out:
        print('Blue central log(minimum mass)', np.log10(bc_min), 'in',
              round((time() - range3), 4), 's')

    range4 = time()
    bs_min = find_min(bs_param[0], bs_param[1], bs_param[2], cosmology, z_range, skyarea,
                      gal_max_s, bs_halo, print_out, run_anyway)
    if print_out:
        print('Blue satellite log(minimum mass)', np.log10(bs_min), 'in',
              round((time() - range4), 4), 's')
        print('')

    # Get a catalogue for each population
    cat_time = time()

    redshift = np.linspace(z_range[0], z_range[1], 100)  # Redshift volume

    rc_cat = schechter_smf(redshift, rc_param[0], rc_param[1], rc_param[2], rc_min,
                           gal_max_h, skyarea*u.deg**2, cosmology)[1]
    rs_cat = schechter_smf(redshift, rs_param[0], rs_param[1], rs_param[2], rs_min,
                           gal_max_s, skyarea*u.deg**2, cosmology)[1]
    bc_cat = schechter_smf(redshift, bc_param[0], bc_param[1], bc_param[2], bc_min,
                           gal_max_h, skyarea*u.deg**2, cosmology)[1]
    bs_cat = schechter_smf(redshift, bs_param[0], bs_param[1], bs_param[2], bs_min,
                           gal_max_s, skyarea*u.deg**2, cosmology)[1]

    if scatter_prox:
        if print_out:
            print('Using scattering')
            print('')
        rc_order = scatter_proxy(rc_cat)
        rs_order = scatter_proxy(rs_cat)
        bc_order = scatter_proxy(bc_cat)
        bs_order = scatter_proxy(bs_cat)

    if print_out:
        print('Galaxy catalogues generated in', round((time() - cat_time), 2), 's')

    # Order and process DM and galaxies
    # Concatenate halos and subhalos
    halo_subhalo = np.concatenate((parent_halo, subhalo_m), axis=0)
    ID_list = np.concatenate((ID_halo, ID_sub), axis=0)
    z_list = np.concatenate((z_halo, z_sub), axis=0)
    q_list = np.concatenate((h_quench, sub_quench), axis=0)

    # Sort lists by halo mass
    stack = np.stack((halo_subhalo, ID_list, z_list, q_list), axis=1)
    order1 = stack[np.argsort(stack[:, 0])]
    hs_order = np.flip(order1[:, 0])
    id_hs = np.flip(order1[:, 1])
    z_hs = np.flip(order1[:, 2])
    qu_hs = np.flip(order1[:, 3])

    # List galaxies if not already sorted by scattering
    if not scatter_prox:
        if print_out:
            print('No scatter applied')
            print('')
        rc_order = np.flip(np.sort(rc_cat))
        rs_order = np.flip(np.sort(rs_cat))
        bc_order = np.flip(np.sort(bc_cat))
        bs_order = np.flip(np.sort(bs_cat))

    # Assignment of galaxies
    assign_st = time()
    hs_fin, gal_fin, id_fin, z_fin, gal_type_fin = assignment(hs_order, rc_order, rs_order,
                                                              bc_order, bs_order, qu_hs, id_hs,
                                                              z_hs, scatter=scatter_prox)

    if print_out:
        print('Galaxies assigned in', round((time() - assign_st), 2), 's')
        print('')

    # Strip subhalos
    sub_loc = np.where(id_fin > 0)
    hs_fin[sub_loc] = hs_fin[sub_loc]/sub_x

    # Create output dictionary
    sham_dict = {
        'Halo mass': hs_fin,
        'Galaxy mass': gal_fin,
        'Galaxy type': gal_type_fin,
        'ID value': id_fin,
        'Redshift': z_fin
    }

    if print_out:
        print('SHAM run in', round(((time() - sham_st)/60), 2), 'min')

    return sham_dict
