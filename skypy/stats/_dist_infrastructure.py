import scipy.stats as st
from ._dist_params import distcont
from unittest.mock import patch


class rv_continuous(st.rv_continuous):
    @patch('scipy.stats._distn_infrastructure.distcont', distcont)
    def __init__(self, *args, **kwargs):
        super(rv_continuous, self).__init__(*args, **kwargs)

        self.__doc__ = self.__doc__.replace(
            'from scipy.stats import {}'.format(self.name),
            'from skypy.stats import {}'.format(self.name))
