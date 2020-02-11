'''Galaxy ellipticity module.

This module provides models to sample galaxy ellipticities. SkyPy uses the
epsilon definition of ellipticity, e = (1-q)/(1+q).
'''

from scipy import stats

'''Ellipticities from the COSMOS 23.5 sample.

The ellipticity distribution is obtained by fitting a beta distribution to the
observed ellipticities (via `scipy.stats.beta.fit()`).

The ellipticities are taken from the Sérsic fits to the COSMOS 23.5 sample
provided by GREAT3 (Mandelbaum et al. 2014).
'''
cosmos235beta = stats.beta(1.6249904857918940, 3.804195234419490)

'''Ellipticities from the COSMOS 25.2 sample.

The ellipticity distribution is obtained by fitting a beta distribution to the
observed ellipticities (via `scipy.stats.beta.fit()`).

The ellipticities are taken from the Sérsic fits to the COSMOS 25.2 sample
provided by GREAT3 (Mandelbaum et al. 2014).
'''
cosmos252beta = stats.beta(1.8929446349798722, 3.466296193117389)
