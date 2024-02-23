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
   galaxy_cat
   assignment
   sham_arrays
   run_sham
'''

#Imports
import numpy as np
from skypy.pipeline import Pipeline
from skypy.halos import mass #Vale and Ostriker 2004 num of subhalos
from time import time
from scipy.special import erf #Error function for quenching
from scipy.integrate import trapezoid as trap #Integration of galaxies
from astropy import units as u
import os

__all__ = [
    'quenching_funct',
    'find_min',
    'run_file',
    'gen_sub_cat',
    'galaxy_cat',
    'assignment',
    'sham_plots',
    'run_sham',
 ]

#General functions


#Quenching function
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
    .. [1] Peng Y.-j., et al., 2010, ApJ, 721, 193
       
    '''
    mass = np.atleast_1d(mass)
    
    #Errors
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

    #Only works with an increasing array due to normalisation?
    qu_id = np.arange(0, len(mass)) #Random order elements (ie halo original order)

    #Elements of sorted halos
    stack1 = np.stack((mass, qu_id), axis=1)
    order1 = stack1[np.argsort(stack1[:,0])] #Order on the halos
    mass = order1[:, 0]
    order_qu_id = order1[:, 1]

    calc = np.log10(mass/M_mu)/sigma
    old_prob = 0.5 * (1.0 + erf(calc/np.sqrt(2))) #Base error funct, 0 -> 1
    add_val = baseline/(1-baseline)
    prob_sub = old_prob + add_val #Add value to plot to get baseline
    prob_array = prob_sub/prob_sub[-1] #Normalise, base -> 1
    rand_array = np.random.uniform(size=len(mass))
    qu_arr = rand_array < prob_array

    #Reorder truth to match original order
    stack2 = np.stack((order_qu_id, qu_arr), axis=1)
    order2 = stack2[np.argsort(stack2[:,0])] 
    return order2[:, 1]


#Find minimum galaxy mass for given halo population through the integral
#@units.quantity_input(sky_area=units.sr) #To make area work?
        
def find_min(m_star, phi_star, alpha, cosmology, z_range, skyarea, max_mass, no_halo, print_out = False, run_anyway = False):
    #Find minimum mass for halo type
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
    >>> m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
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
    .. [1] Birrer S., Lilly S., Amara A., Paranjape A., Refregier A., 2014,
        The Astrophysical Journal, 793, 12
       
    '''
    z_range = np.atleast_1d(z_range)

    #Errors
    if z_range.shape != (2, ):
        raise Exception('The wrong number of redshifts were given')
    if z_range[0] < 0 or z_range[1] < 0:
        raise Exception('Redshift cannot be negative')
    if z_range[1] <= z_range[0]:
        raise Exception('The second redshift should be more than the first')
    if alpha >= 0:
        alpha = -1*alpha
        raise Warning('The Schechter function for galaxies is defined such that alpha should be negative, set_alpha = -alpha')
    if m_star <= 0 or phi_star <= 0:
        raise Exception('M* and phi* must be positive and non-zero numbers')
    if skyarea <= 0:
        raise Exception('The skyarea must be a positive non-zero number')
    if max_mass <= 10**6:
        raise Exception('The maximum mass must be more than 10^6 solar masses')

    skyarea = skyarea*u.deg**2
    z = np.linspace(z_range[0], z_range[1], 1000)
    dVdz = (cosmology.differential_comoving_volume(z)*skyarea).to_value('Mpc3')  
    
    #Other
    mass_mins = 10**np.arange(6, 10, 0.1) #Minimums to integrate, linear in logspace
    phi_m = phi_star/m_star
    res = 10**(-2.3) #Want all integral bins to have same resolution in log space, this should give accurate integral value
    
    #Integrate
    int_vals = []
    for ii in mass_mins:
        mass_bin = 10**np.arange(np.log10(ii), np.log10(max_mass), res) #Masses to integrate over
        m_m  = mass_bin/m_star
        dndmdV = phi_m*np.e**(-m_m)*(m_m)**alpha #Mass function
        dndV = trap(dndmdV, mass_bin)
        dn = dndV*dVdz #n(z)*V(z)
        int_vals.append(trap(dn, z)) #Integral per mass bin
    
    int_vals = np.array(int_vals)
        
    min_mass = np.interp(no_halo, np.flip(int_vals), np.flip(mass_mins)) #x must be increasing so flip arrays
    
    #Region within Poisson error
    poisson_min = min(int_vals) - np.sqrt(min(int_vals))
    poisson_max = max(int_vals) + np.sqrt(max(int_vals))
    
    if (no_halo < poisson_min or no_halo > poisson_max) and run_anyway == False:
        #Interpolation falls outside of created range
        raise Exception('Number of halos not within reachable bounds')
        
    elif (no_halo < poisson_min or no_halo > poisson_max) and run_anyway == True:
        if print_out == True:
            print('Outside of interpolation, but running anyway')
        return 10**(7)
        
    return min_mass

#Generate catalogues
def run_file(file_name, table1, info1, info2 = None):
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
    #Errors
    if type(file_name) != str:
        raise Exception('File name must be a string')
    try:
        os.path.exists(file_name)
    except:
        raise Exception('File does not exist')

    #Use pipeline
    pipe = Pipeline.read(file_name)
    pipe.execute()
    
    #Get information
    info = pipe[table1]
    cat = info[info1]
    if info2 != None:
        z = info[info2]
        return cat, z
    
    return cat
    

#Generate subhalo catalogues
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

    #Errors
    if len(parent_halo) != len(z_halo):
        raise Exception('Catalogue of halos and redshifts must be the same length')
    if len(np.where(parent_halo <= 0)[0]) > 0:
        raise Exception('Masses in catalogue should be positive and non-zero')
    if len(np.where(z_halo < 0)[0]) > 0:
        raise Exception('Redshifts in catalogue should be positive')
    if sub_alpha < 0:
        sub_alpha = -1*sub_alpha
        raise Warning('The subhalo mass function is defined such that alpha should be positive, set_alpha = -alpha')
    if sub_alpha >= 2:
        raise Exception('Subhalo alpha must be less than 2')
    if sub_x < 1:
        raise Exception('Subhalo x cannot be less than 1')
    if sub_beta <= 0 or sub_beta > 1:
        raise Exception('Subhalo beta must be between 0 and 1')
    if sub_gamma < 0 or sub_gamma > 1:
        raise Exception('Subhalo gamma must be between 0 and 1')
    
    if parent_halo.size == 1: #ie only one halo is provided
        m_min = 10**(10)
    else:
        m_min = min(parent_halo) #Min mass of subhalo to generate (SET AT RESOLUTION OF PARENT HALOS)
    ID_halo = -1*np.arange(1, len(parent_halo)+1, dtype=int) #Halo IDs
    
    #Get list of halos that will have subhalos
    halo_to_sub = parent_halo[np.where(parent_halo >= 10**(10))] #ADD - change this hardcoded value to a larger number?
    ID_to_sub = -1*ID_halo[np.where(parent_halo >= 10**(10))]
    z_to_sub = z_halo[np.where(parent_halo >= 10**(10))]

    #Get subhalos
    no_sub = mass.number_subhalos(halo_to_sub, sub_alpha, sub_beta, sub_gamma, sub_x, m_min, noise=True)
    sub_masses = mass.subhalo_mass_sampler(halo_to_sub, no_sub, sub_alpha, sub_beta, sub_x, m_min)
    
    #Assign subhalos to parents by ID value
    ID_sub = []
    z_sub = []
    ID_count = 0
    for ii in no_sub:
        ID_set = ID_to_sub[ID_count]
        ID_sub.extend(ID_set*np.ones(ii, dtype=int)) #Positive ID for subhalo
        z_sub.extend(z_to_sub[ID_count]*np.ones(ii)) #Same redshift as the parent
        ID_count +=1
        
    #Delete any generated subhalos smaller than the resolution
    del_sub = np.where(sub_masses < m_min)[0]
    sub_masses = np.delete(sub_masses, del_sub)
    ID_sub = np.delete(ID_sub, del_sub)
    z_sub = np.delete(z_sub, del_sub)
        
    return ID_halo, sub_masses, ID_sub, z_sub

#Generate galaxies

#Generate YAML file and get catalogue
def galaxy_cat(m_star, phi_star, alpha, cosmology, z_range, skyarea, min_mass, max_mass, file_name):
    r'''Function that generates a galaxy catalogue by generating a YAML file
    and running it

    This function generates a YAML file using galaxy Schechter mass function
    parameters plus input cosmology and mass ranges, and then runs the file
    to generate a catalogue of masses.

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
    min_mass : float
        Minimum mass of galaxies to generate
    max_mass : float
        Maximum mass of galaxies to generate
    file_name : str
        String of file name to be run

    Returns
    --------
    catalogue: (nm, ) array_like
        List of masses
    redshifts: (nm, ) array_like
        List of redshifts generated for each mass, if requested

    Examples
    ---------
    >>> from skypy.halos import sham
    >>> from astropy.cosmology import FlatLambdaCDM

    This example generates a galaxy catalogue using the Schechter
    parameters for red centrals from [1]_

    >>> #Parameters
    >>> m_star, phi_star, alpha = 10**(10.58), 10**(-2.77), -0.33
    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    >>> z_range = [0.01, 0.1]
    >>> skyarea = 600
    >>> min_mass = 10**(7)
    >>> max_mass = 10**(14)
    >>> #Run function
    >>> galaxy_cat = sham.galaxy_cat(m_star, phi_star, alpha, cosmology,
    ...                              z_range, skyarea, min_mass, max_mass,
    ...                              'red_central.yaml')


    References
    ----------
    .. [1] Birrer S., Lilly S., Amara A., Paranjape A., Refregier A., 2014,
        The Astrophysical Journal, 793, 12
       
    '''
    z_range = np.atleast_1d(z_range)
    
    #Errors
    if z_range.shape != (2, ):
        raise Exception('The wrong number of redshifts were given')
    if z_range[0] < 0 or z_range[1] < 0:
        raise Exception('Redshift cannot be negative')
    if z_range[1] <= z_range[0]:
        raise Exception('The second redshift should be more than the first')
    if alpha > 0:
        alpha = -1*alpha
        raise Warning('The Schechter function for galaxies is defined such that alpha should be negative, set_alpha = -alpha')
    if m_star <= 0 or phi_star <= 0:
        raise Exception('M* and phi* must be positive and non-zero numbers')
    if skyarea <= 0:
        raise Exception('The skyarea must be a positive non-zero number')
    if min_mass > max_mass:
        raise Exception('The minimum mass should be less than the maximum mass')
    if cosmology.name == None:
        raise Exception('Cosmology object must have an astropy cosmology name')

    #Galaxy parameters
    line1 = 'm_star: !numpy.power [10, ' + str(np.log10(m_star)) + ']\n'
    line2 = 'phi_star: !numpy.power [10, ' + str(np.log10(phi_star)) + ']\n'
    line3 = 'alpha_val: ' + str(alpha) + '\n'
    
    #Mass range
    line4 = 'm_min: !numpy.power [10, ' + str(np.log10(min_mass)) + ']\n'
    line5 = 'm_max: !numpy.power [10, ' + str(np.log10(max_mass)) + ']\n'
    
    #Observational parameters
    if type(skyarea) != float:
        skyarea = float(skyarea)
    line6 = 'sky_area: ' + str(skyarea) + ' deg2\n'
    line7 = 'z_range: !numpy.linspace [' + str(z_range[0]) + ', ' + str(z_range[1]) + ', 100]\n'
    
    #Cosmology
    line8 = 'cosmology: !astropy.cosmology.' + cosmology.name + '\n'
    line9 = '  H0: ' +  str(cosmology.h*100) + '\n'
    line10 = '  Om0: ' +  str(cosmology.Om0) + '\n'
    
    #Call function
    function = 'tables:\n  galaxy:\n    z, sm: !skypy.galaxies.schechter_smf\n      redshift: $z_range\n'
    function += '      m_star: $m_star\n      phi_star: $phi_star\n      alpha: $alpha_val\n      m_min: $m_min\n'
    function += '      m_max: $m_max\n      sky_area: $sky_area\n'

    #Make one large string
    yaml_lines = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + function
    
    file_gal = open(file_name, 'w')
    file_gal.write(yaml_lines)
    file_gal.close()
    
    #Execute file
    return run_file(file_name, 'galaxy', 'sm')

#Assignment
def assignment(hs_order, rc_order, rs_order, bc_order, bs_order, qu_order, id_order, z_order):
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
        List of assigned galaxy types, 1 = red central, 2 = red satellite,
        3 = blue central, 4 = blue satellite

    Examples
    ---------
    >>> from skypy.halos import sham
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
    >>> m_star1, phi_star1, alpha1 = 10**(10.58), 10**(-2.77), -0.33
    >>> m_star2, phi_star2, alpha2 = 10**(10.64), 10**(-4.24), -1.54
    >>> m_star3, phi_star3, alpha3 = 10**(10.65), 10**(-2.98), -1.48
    >>> m_star4, phi_star4, alpha4 = 10**(10.55), 10**(-3.96), -1.53
    >>> min_mass = 10**(7)
    >>> max_mass = 10**(14)
    >>> #Generate the galaxies
    >>> rc_cat = sham.galaxy_cat(m_star1, phi_star1, alpha1, cosmology,
    ...                          z_range, skyarea, min_mass, max_mass,
    ...                          'rc_file.yaml')
    >>> rs_cat = sham.galaxy_cat(m_star2, phi_star2, alpha2, cosmology,
    ...                          z_range, skyarea, min_mass, max_mass,
    ...                          'rs_file.yaml')
    >>> bc_cat = sham.galaxy_cat(m_star3, phi_star3, alpha3, cosmology,
    ...                          z_range, skyarea, min_mass, max_mass,
    ...                          'bc_file.yaml')
    >>> bs_cat = sham.galaxy_cat(m_star4, phi_star4, alpha4, cosmology,
    ...                          z_range, skyarea, min_mass, max_mass,
    ...                          'bs_file.yaml')
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

    #Order check
    hs_order_check = np.diff(hs_order)
    rc_order_check = np.diff(rc_order)
    rs_order_check = np.diff(rs_order)
    bc_order_check = np.diff(bc_order)
    bs_order_check = np.diff(bs_order)

    #Errors
    if ((hs_order <= 0)).any():
        raise Exception('Halo masses must be positive and non-zero')
    if ((rc_order <= 0)).any() or ((rs_order <= 0)).any() or ((bc_order <= 0)).any() or ((bs_order <= 0)).any():
        raise Exception('Galaxy masses must be positive and non-zero')
    if ((hs_order_check > 0)).all():
        hs_order = np.flip(hs_order)
        raise Warning('Halo masses were in the wrong order and have been corrected')
    elif ((hs_order_check > 0)).any():
        raise Exception('Halos are not in a sorted order')
    if ((rc_order_check > 0)).any():
        rc_order = np.flip(rc_order)
        raise Warning('Red central galaxy masses were in the wrong order and have been corrected')
    elif ((rc_order_check > 0)).any():
        raise Exception('Red central galaxies are not in a sorted order')
    if ((rs_order_check > 0)).all():
        rs_order = np.flip(rs_order)
        raise Warning('Red satellite galaxy masses were in the wrong order and have been corrected')
    elif ((rs_order_check > 0)).any():
        raise Exception('Red satellite galaxies are not in a sorted order')
    if ((bc_order_check > 0)).all():
        bc_order = np.flip(bc_order)
        raise Warning('Blue central galaxy masses were in the wrong order and have been corrected')
    elif ((bc_order_check > 0)).any():
        raise Exception('Blue central galaxies are not in a sorted order')
    if ((bs_order_check > 0)).all():
        bs_order = np.flip(bs_order)
        raise Warning('Blue satellite galaxy masses were in the wrong order and have been corrected')
    elif ((bs_order_check > 0)).any():
        raise Exception('Blue satellite galaxies are not in a sorted order')
    if hs_order.shape != qu_order.shape or qu_order.shape != id_order.shape or id_order.shape != z_order.shape:
        raise Exception('All arrays pertaining to halos must be the same shape')
    

    #Assign galaxies to halos
    del_ele = [] #Elements to delete later
    gal_assigned = [] #Assigned galaxies
    gal_type_A = [] #Population assigned

    gal_num = len(rc_order) + len(rs_order) + len(bc_order) + len(bs_order) #Total number of galaxies

    rc_counter = 0 #Counters for each population
    rs_counter = 0
    bc_counter = 0
    bs_counter = 0

    for ii in range(len(hs_order)):
        qu = qu_order[ii]
        total_counter = rc_counter + rs_counter + bc_counter + bs_counter
        ID_A = id_order[ii]

        if total_counter == gal_num: #All galaxies assigned
            del_array = np.arange(ii, len(hs_order))
            del_ele.extend(del_array)
            break

        if qu == 1: #Halo is quenched
            if ID_A < 0 and rc_counter != len(rc_order): #Halo assigned a mass quenched
                gal_assigned.append(rc_order[rc_counter])
                gal_type_A.append(1)
                rc_counter += 1

            elif ID_A > 0 and rs_counter != len(rs_order): #Subhalo assigned an environment quenched
                gal_assigned.append(rs_order[rs_counter])
                gal_type_A.append(2)
                rs_counter += 1

            else: #No red to assign
                del_ele.append(ii) #Delete unassigned halos

        else: #Halo not quenched
            if ID_A < 0 and bc_counter != len(bc_order): #Halo assigned a blue central
                gal_assigned.append(bc_order[bc_counter])
                gal_type_A.append(3)
                bc_counter += 1

            elif ID_A > 0 and bs_counter != len(bs_order): #Subhalo assigned a blue satellite
                gal_assigned.append(bs_order[bs_counter])
                gal_type_A.append(4)
                bs_counter += 1

            else: #No blue to assign
                del_ele.append(ii)
    
    #Delete and array final lists
    hs_fin = np.delete(hs_order, del_ele)
    id_fin = np.delete(id_order, del_ele)
    z_fin = np.delete(z_order, del_ele)
    gal_fin = np.array(gal_assigned)
    gal_type_fin = np.array(gal_type_A)
    
    return hs_fin, gal_fin, id_fin, z_fin, gal_type_fin

#Separate populations
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
        List of assigned galaxies types with the tags 1,2,3,4 for
        red centrals, red satellites, blue centrals, blue satellites
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
    if hs_fin.shape != gal_fin.shape or gal_fin.shape != gal_type_fin.shape:
        raise Exception('All arrays must be the same shape')
    if ((hs_fin <= 0)).any():
        raise Exception('Halo masses must be positive and non-zero')
    if ((gal_fin <= 0)).any():
        raise Exception('Galaxy masses must be positive and non-zero')

    rc_dots = np.where(gal_type_fin == 1)[0]
    rs_dots = np.where(gal_type_fin == 2)[0]
    bc_dots = np.where(gal_type_fin == 3)[0]
    bs_dots = np.where(gal_type_fin == 4)[0]
    
    #SHAMs by galaxy population
    sham_rc = np.stack((hs_fin[rc_dots], gal_fin[rc_dots]),axis=1) 
    sham_rs = np.stack((hs_fin[rs_dots], gal_fin[rs_dots]),axis=1)
    sham_bc = np.stack((hs_fin[bc_dots], gal_fin[bc_dots]),axis=1)
    sham_bs = np.stack((hs_fin[bs_dots], gal_fin[bs_dots]),axis=1)
    
    #Combine both sets of centrals
    halo_all = np.concatenate((sham_rc[:,0], sham_bc[:,0]), axis=0) #All halos and galaxies
    gals_all = np.concatenate((sham_rc[:,1], sham_bc[:,1]), axis=0)

    sham_concat = np.stack((halo_all, gals_all),axis=1)
    sham_order = sham_concat[np.argsort(sham_concat[:,0])] #Order by halo mass
    sham_cen = np.flip(sham_order, 0)
    
    #Combine both sets of satellites
    halo_sub = np.concatenate((sham_rs[:,0], sham_bs[:,0]), axis=0) #All halos and galaxies
    gals_sub = np.concatenate((sham_rs[:,1], sham_bs[:,1]), axis=0)

    sham_concats = np.stack((halo_sub, gals_sub),axis=1)
    sham_orders = sham_concats[np.argsort(sham_concats[:,0])] #Order by halo mass
    sham_sub = np.flip(sham_orders, 0)

    if print_out == True:
        print('Created SHAMs')
    
    return sham_rc, sham_rs, sham_bc, sham_bs, sham_cen, sham_sub

#Called SHAM function
def run_sham(h_file, gal_param, cosmology, z_range, skyarea, qu_h_param, qu_s_param, sub_param = [1.91, 0.39, 0.1, 3],
             gal_max_h = 10**(14), gal_max_s = 10**(13), print_out=False, run_anyway=False):
    r'''Function that takes all inputs for the halos and galaxies and creates
    runs a SHAM over them

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
    qu_h_param : (2, ) array_like
        The mean and standard deviation of the error function to quench
        central halos (see 'quenching_funct' for more detail)
    qu_s_param : (3, ) array_like
        The mean, standard deviation and baseline of the error function
        to quench satellite halos (see 'quenching_funct' for more detail)
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

    Returns
    --------
    sham_dict: (nm, 5) dictionary
        A dictionary containing the necessary outputs stored as numpy arrays.
        - 'Halo mass' parent and subhalo masses, in units of solar mass,
          the subhalos are their present stripped mass
        - 'Galaxy mass' assigned galaxy masses, in units of solar mass
        - 'Galaxy type' type of galaxy assigned, written as 1, 2, 3 or 4
          for red centrals, red satellites, blue centrals and blue satellites
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
    >>> gal_param = np.array([[10**(10.58), 10**(-2.77), -0.33],
    ...                      [10**(10.64), 10**(-4.24), -1.54],
    ...                      [10**(10.65), 10**(-2.98), -1.48],
    ...                      [10**(10.55), 10**(-3.96), -1.53]])
    >>> cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
    >>> z_range = np.array([0.01, 0.1])
    >>> skyarea = 600.
    >>> qu_h = np.array([10**(12.1), 0.45])
    >>> qu_s = np.array([10**(11.9), 0.4, 0.5])
    >>> #Run function
    >>> sham_dict = run_sham(h_file, gal_param, cosmology, z_range, skyarea,
    ...                      qu_h, qu_s, sub_param = [1.91, 0.39, 0.1, 3],
    ...                      gal_max_h = 10**(14), gal_max_s = 10**(13),
    ...                      print_out=False, run_anyway=True)
    
    References
    ----------
    .. [1] Vale A., Ostriker J. P., 2004, Monthly Notices of the Royal
       Astronomical Society, 353, 189
    .. [2] Birrer S., Lilly S., Amara A., Paranjape A., Refregier A., 2014,
       The Astrophysical Journal, 793, 12
    .. [3] Peng Y.-j., et al., 2010, ApJ, 721, 193
    '''
    sham_st = time()

    gal_param = np.atleast_1d(gal_param)
    qu_h_param = np.atleast_1d(qu_h_param)
    qu_s_param = np.atleast_1d(qu_s_param)
    
    #Check that all inputs are of the correct type and size
    if type(h_file) != str:
        raise Exception('Halo YAML file must be provided as a string')
    if gal_param.shape != (4,3):
        if gal_param.shape[0] != 4:
            raise Exception('The wrong number of galaxies have been provided in their parameters')
        elif gal_param.shape[1] != 3:
            raise Exception('The wrong number of galaxy parameters have been provided')
        else:
            raise Exception('Supplied galaxy parameters are not the correct shape')
    if qu_h_param.shape != (2,):
        raise Exception('Provided incorrect number of halo quenching parameters')
    if qu_s_param.shape != (3,):
        raise Exception('Provided incorrect number of subhalo quenching parameters')
    
    
    #Generate parent halos from YAML file (TODO: remove hard coded table/variable names)
    h_st = time()
    parent_halo, z_halo = run_file(h_file, 'halo', 'mass', 'z')
    if print_out == True:
        print('Halo catalogues generated in', round((time() - h_st), 2), 's')
    
    #Generate subhalos and IDs
    sub_alpha = sub_param[0] #Vale and Ostriker 2004 mostly
    sub_beta = sub_param[1]
    sub_gamma = sub_param[2]
    sub_x = sub_param[3]
    
    sub_tim = time()
    ID_halo, subhalo_m, ID_sub, z_sub = gen_sub_cat(parent_halo, z_halo, sub_alpha, sub_beta, sub_gamma, sub_x)
    
    if print_out == True:
        print('Generated subhalos and IDs in', round((time() - sub_tim), 2), 's')
        print('')
    
    #Quench halos and subhalos
    M_mu = qu_h_param[0] #Parameters
    sigma = qu_h_param[1]
    M_mus = qu_s_param[0]
    sigmas = qu_s_param[1]
    baseline_s = qu_s_param[2]

    h_quench = quenching_funct(parent_halo, M_mu, sigma, 0) #Quenched halos
    sub_quench = quenching_funct(subhalo_m, M_mus, sigmas, baseline_s) #Quenched subhalos
    
    #Galaxy Schechter function parameters
    rc_param = gal_param[0]
    rs_param = gal_param[1]
    bc_param = gal_param[2]
    bs_param = gal_param[3]
    
    #Number of halos for each population
    rc_halo = len(np.where(h_quench == 1)[0])
    rs_halo = len(np.where(sub_quench == 1)[0])
    bc_halo = len(np.where(h_quench == 0)[0])
    bs_halo = len(np.where(sub_quench == 0)[0])
    
    #Find galaxy mass range (m_star, phi_star, alpha, tag)
    #TODO Add a way to import a look up table
    range1 = time()
    rc_min = find_min(rc_param[0], rc_param[1], rc_param[2], cosmology, z_range, skyarea,
                      gal_max_h, rc_halo, print_out, run_anyway)
    if print_out == True:
        print('Red central log(minimum mass)', np.log10(rc_min), 'in', round((time() - range1), 4), 's')
    
    range2 = time()
    rs_min = find_min(rs_param[0], rs_param[1], rs_param[2], cosmology, z_range, skyarea,
                      gal_max_s, rs_halo, print_out, run_anyway)
    if print_out == True:
        print('Red satellite log(minimum mass)', np.log10(rs_min), 'in', round((time() - range2), 4), 's')
            
    range3 = time()
    bc_min = find_min(bc_param[0], bc_param[1], bc_param[2], cosmology, z_range, skyarea,
                      gal_max_h, bc_halo, print_out, run_anyway)
    if print_out == True:
        print('Blue central log(minimum mass)', np.log10(bc_min), 'in', round((time() - range3), 4), 's')
    
    range4 = time()
    bs_min = find_min(bs_param[0], bs_param[1], bs_param[2], cosmology, z_range, skyarea,
                      gal_max_s, bs_halo, print_out, run_anyway)
    if print_out == True:
        print('Blue satellite log(minimum mass)', np.log10(bs_min), 'in', round((time() - range4), 4), 's')
        print('')
    
    #Get a catalogue for each population
    cat_time = time()
    rc_cat = galaxy_cat(rc_param[0], rc_param[1], rc_param[2], cosmology, z_range, skyarea, rc_min, gal_max_h, 'rc_test.yaml')
    rs_cat = galaxy_cat(rs_param[0], rs_param[1], rs_param[2], cosmology, z_range, skyarea, rs_min, gal_max_s, 'rs_test.yaml')
    bc_cat = galaxy_cat(bc_param[0], bc_param[1], bc_param[2], cosmology, z_range, skyarea, bc_min, gal_max_h, 'bc_test.yaml')
    bs_cat = galaxy_cat(bs_param[0], bs_param[1], bs_param[2], cosmology, z_range, skyarea, bs_min, gal_max_s, 'bs_test.yaml')
    
    if print_out == True:
        print('Galaxy catalogues generated in', round((time() - cat_time), 2), 's')
    
    #Clean up the files
    os.remove('rc_test.yaml')
    os.remove('rs_test.yaml')
    os.remove('bc_test.yaml')
    os.remove('bs_test.yaml')
    
    #Order and process DM and galaxies----------------------------------------------------------------
    #Concatenate halos and subhalos
    halo_subhalo = np.concatenate((parent_halo, subhalo_m), axis=0)
    ID_list = np.concatenate((ID_halo, ID_sub), axis=0)
    z_list = np.concatenate((z_halo, z_sub), axis=0)
    q_list = np.concatenate((h_quench, sub_quench), axis=0)
    
    #Sort lists by halo mass
    stack = np.stack((halo_subhalo, ID_list, z_list, q_list), axis=1)
    order1 = stack[np.argsort(stack[:,0])]
    hs_order = np.flip(order1[:,0])
    id_hs = np.flip(order1[:,1])
    z_hs = np.flip(order1[:,2])
    qu_hs = np.flip(order1[:,3])

    #List galaxies
    rc_order = np.flip(np.sort(rc_cat))
    rs_order = np.flip(np.sort(rs_cat))
    bc_order = np.flip(np.sort(bc_cat))
    bs_order = np.flip(np.sort(bs_cat))
    
    #Assignment of galaxies
    assign_st = time()
    hs_fin, gal_fin, id_fin, z_fin, gal_type_fin = assignment(hs_order, rc_order, rs_order, bc_order, bs_order,
                                                              qu_hs, id_hs, z_hs)
    
    if print_out == True:
        print('Galaxies assigned in', round((time() - assign_st), 2), 's')
        print('')
    
    #Strip subhalos
    sub_loc = np.where(id_fin > 0)
    hs_fin[sub_loc] = hs_fin[sub_loc]/sub_x

    #Create output dictionary
    sham_dict = {
        'Halo mass': hs_fin,
        'Galaxy mass': gal_fin,
        'Galaxy type': gal_type_fin,
        'ID value': id_fin,
        'Redshift': z_fin
    }

    if print_out == True:
        print('SHAM run in', round(((time() - sham_st)/60), 2), 'min')
    
    return sham_dict