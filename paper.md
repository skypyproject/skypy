---
title: '`SkyPy`: A package for modelling the Universe'
tags:
  - astronomy
  - astrophysics
  - cosmology
  - simulations
authors:
  - name: Adam Amara^[adam.amara\@port.ac.uk]
    orcid: 0000-0003-3481-3491
    affiliation: 1
  - name: Lucia F. de la Bella
    orcid: 0000-0002-1064-3400
    affiliation: "1, 2"
  - name: Simon Birrer
    orcid: 0000-0003-3195-5507
    affiliation: 3
  - name: Sarah Bridle
    orcid: 0000-0002-0128-1006
    affiliation: 2
  - name: Juan Pablo Cordero
    orcid: 0000-0002-6625-7656
    affiliation: 2
  - name: Ginevra Favole
    orcid: 0000-0002-8218-563X
    affiliation: 4
  - name: Ian Harrison
    orcid: 0000-0002-4437-0770
    affiliation: "5, 2"
  - name: Ian W. Harry
    orcid: 0000-0002-5304-9372
    affiliation: 1
  - name: Coleman Krawczyk
    orcid: 0000-0001-9233-2341
    affiliation: 1
  - name: Andrew Lundgren
    orcid: 0000-0002-0363-4469
    affiliation: 1
  - name: Brian Nord
    orcid: 0000-0001-6706-8972
    affiliation: "6, 7, 8"
  - name: Laura K. Nuttall
    orcid: 0000-0002-8599-8791
    affiliation: 1
  - name: Richard P. Rollins^[richard.rollins\@ed.ac.uk]
    orcid: 0000-0003-1291-1023
    affiliation: "9, 2"
  - name: Philipp Sudek
    orcid: 0000-0001-8685-2308
    affiliation: 1
  - name: Sut-Ieng Tam
    orcid: 0000-0002-6724-833X
    affiliation: 10
  - name: Nicolas Tessore
    orcid: 0000-0002-9696-7931
    affiliation: 11
  - name: Arthur E. Tolley
    orcid: 0000-0001-9841-943X
    affiliation: 1
  - name: Keiichi Umetsu
    orcid: 0000-0002-7196-4822
    affiliation: 10
  - name: Andrew R. Williamson
    orcid: 0000-0002-7627-8688
    affiliation: 1
  - name: Laura Wolz
    orcid: 0000-0003-3334-3037
    affiliation: 2
affiliations:
  - name: Institute of Cosmology and Gravitation, University of Portsmouth
    index: 1
  - name: Jodrell Bank Centre for Astrophysics, University of Manchester
    index: 2
  - name: Kavli Institute for Particle Astrophysics and Cosmology and Department of Physics, Stanford University
    index: 3
  - name: Institute of Physics, Laboratory of Astrophysics, Ecole Polytechnique Fédérale de Lausanne
    index: 4
  - name: Department of Physics, University of Oxford
    index: 5
  - name: Fermi National Accelerator Laboratory
    index: 6
  - name: Kavli Institute for Cosmological Physics, University of Chicago
    index: 7
  - name: Department of Astronomy and Astrophysics, University of Chicago
    index: 8
  - name: Institute for Astronomy, University of Edinburgh
    index: 9
  - name: Institute of Astronomy and Astrophysics, Academia Sinica
    index: 10
  - name: Department of Physics and Astronomy, University College London
    index: 11

date: 30 June 2021
bibliography: paper.bib

---

# Summary

`SkyPy` is an open-source Python package for simulating the astrophysical sky. It comprises a library of physical and empirical models across a range of observables and a command line script to run end-to-end simulations. The library provides functions that sample realisations of sources and their associated properties from probability distributions. Simulation pipelines are constructed from these models using a YAML-based configuration syntax, while task scheduling and data dependencies are handled internally and the modular design allows users to interface with external software. `SkyPy` is developed and maintained by a diverse community of domain experts with a focus on software sustainability and interoperability. By fostering co-development, it provides a framework for correlated simulations of a range of cosmological probes including galaxy populations, large scale structure, the cosmic microwave background, supernovae and gravitational waves.

Version `0.4` implements functions that model various properties of galaxies including luminosity functions, redshift distributions and optical photometry from spectral energy distribution templates. Future releases will provide additional modules, for example to simulate populations of dark matter halos and model the galaxy-halo connection, making use of existing software packages from the astrophysics community where appropriate.

# Statement of need

An open-data revolution in astronomy led by past, ongoing, and future legacy surveys such as *Euclid* [@Euclid2011], the Rubin Observatory Legacy Survey of Space and Time [@LSST2019], *Planck* [@Planck2020] and the Laser Interferometer Gravitational-Wave Observatory [@LIGO2015] means access to data is no longer the primary barrier to research. Instead, access to increasingly sophisticated analysis methods is becoming a significant challenge. Researchers frequently need to model multiple astronomical probes and systematics to perform a statistically rigorous analysis that fully exploits the available data. In particular, forward modelling and machine learning have emerged as important techniques for the next generation of surveys and both depend on realistic simulations. However, existing software is frequently closed-source, outdated, unmaintained or developed for specific projects and surveys making it unsuitable for the wider research community. As a consequence astronomers routinely expend significant effort replicating or re-developing existing code. The growing need for skill development and knowledge sharing in astronomy is evidenced by a number of open initiatives focused on software, statistics and machine learning e.g., Astropy [@Astropy2013; @Astropy2018], OpenAstronomy (https://openastronomy.org), Dark Machines (http://darkmachines.org), The Deep Skies Lab (https://deepskieslab.com), and the Cosmo-Statistics Initiative (https://cosmostatistics-initiative.org). `SkyPy` was established as a part of this open ecosystem to meet the research community’s need for realistic simulations and enable forward modelling and machine learning applications.

# Acknowledgements

AA and PS acknowledge support from a Royal Society Wolfson Fellowship grant. LFB, SB, IH, and RPR acknowledge support from the European Research Council in the form of a Consolidator Grant with number 681431. IH also acknowledges support from the Beecroft Trust. JPC acknowledges support granted by Agencia Nacional de Investigación y Desarrollo (ANID) DOCTORADO BECAS CHILE/2016 - 72170279. GF acknowledges financial support from the SNF 175751 "Cosmology with 3D maps of the Universe" research grant. AL and ARW thanks the STFC for support through the grant ST/S000550/1. LKN thanks the UKRI Future Leaders Fellowship for support through the grant MR/T01881X/1. This manuscript has been authored by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the U.S. Department of Energy, Office of Science, Office of High Energy Physics.

# References
