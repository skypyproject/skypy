from astropy.cosmology import default_cosmology, z_at_value
from astropy.table import Table, vstack
from copy import deepcopy
import numpy as np
from skypy.pipeline import Pipeline

__all__ = [
    'Lightcone',
]


class Lightcone:
    r'''Class for running a lightcone simulation.

    This class runs a simulaion pipeline for a set of independent redshift
    slices and returns their combined results.
    '''

    def __init__(self, configuration):
        '''Construct the pipeline.

        Parameters
        ----------
        configuration : dict-like
            Configuration for the lightcone simulation.

        Notes
        -----
        'configuration' should contain an entry 'lightcone' which is a
        dictionary defining 'z_min', 'z_max' and 'n_slice'. These are the
        minimum and maximum redshift of the simulation and the number of
        redshift slices respectively. The redshift range is subdivided into
        slices of equal comoving distance and each slice is simulated as a
        SkyPy Pipeline taking the other entries in 'configuraion'.
        '''

        self.config = deepcopy(configuration)
        self.lightcone_config = self.config.pop('lightcone')

    def execute(self, parameters={}):

        # Lightcone parameters
        z_min = self.lightcone_config['z_min']
        z_max = self.lightcone_config['z_max']
        n_slice = self.lightcone_config['n_slice']

        params = {'z_min': z_min,
                  'z_max': z_min,
                  'slice_z_min': None,
                  'slice_z_max': None,
                  'slice_z_mid': None, }

        # Additional user-defined parameters
        params.update(parameters)

        # Update config with ligthcone parameters and user parameters
        if 'parameters' in self.config:
            self.config['parameters'].update(params)
        else:
            self.config['parameters'] = params

        # SkyPy Pipeline object
        pipeline = Pipeline(self.config)

        # Initialise empty tables
        self.tables = {k: Table() for k in pipeline.table_config.keys()}

        # Cosmology from pipeline
        if pipeline.cosmology:
            self.cosmology = pipeline.get_value(pipeline.cosmology)
        else:
            self.cosmology = default_cosmology.get()

        # Calculate equispaced comoving distance slices in redshift space
        chi_min = self.cosmology.comoving_distance(z_min)
        chi_max = self.cosmology.comoving_distance(z_max)
        chi = np.linspace(chi_min, chi_max, n_slice + 1)
        chi_mid = (chi[:-1] + chi[1:]) / 2
        z = [z_at_value(self.cosmology.comoving_distance, c, z_min, z_max) for c in chi[1:-1]]
        z_mid = [z_at_value(self.cosmology.comoving_distance, c, z_min, z_max) for c in chi_mid]
        redshift_slices = zip([z_min] + z, z + [z_max], z_mid)

        # Simulate redshift slices and append results to tables
        for slice_z_min, slice_z_max, slice_z_mid in redshift_slices:
            slice_params = {'slice_z_min': slice_z_min,
                            'slice_z_max': slice_z_max,
                            'slice_z_mid': slice_z_mid}
            pipeline.execute(parameters=slice_params)
            for k, v in self.tables.items():
                self.tables[k] = vstack((v, pipeline[k]))

    def write(self, file_format=None, overwrite=False):
        r'''Write pipeline results to disk.

        Parameters
        ----------
        file_format : str
            File format used to write tables. Files are written using the
            Astropy unified file read/write interface; see [1]_ for supported
            file formats. If None (default) tables are not written to file.
        overwrite : bool
            Whether to overwrite any existing files without warning.

        References
        ----------
        .. [1] https://docs.astropy.org/en/stable/io/unified.html

        '''
        if file_format:
            for name, data in self.tables.items():
                filename = '.'.join((name, file_format))
                data.write(filename, overwrite=overwrite)
