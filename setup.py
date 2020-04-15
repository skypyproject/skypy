#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import builtins

# Ensure that astropy-helpers is available
import ah_bootstrap  # noqa

from setuptools import setup
from setuptools.config import read_configuration

from astropy_helpers.setup_helpers import register_commands, get_package_info
from astropy_helpers.version_helpers import generate_version_py
try:
    from astropy_helpers.commands.build_sphinx import AstropyBuildDocs
    sphinx_present = True
except ImportError:
    sphinx_present = False

# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = read_configuration('setup.cfg')['metadata']['name']

# Create a dictionary with setup command overrides. Note that this gets
# information about the package (name and version) from the setup.cfg file.
cmdclass = register_commands()

if sphinx_present:
    class build_docs(AstropyBuildDocs):
        def run(self):
            import shutil
            super().run()
            shutil.copy('images/skypy_image.png',
                        './docs/_build/html/_static/astropy_logo_32.png')
            shutil.copy('images/skypy_image.svg',
                        './docs/_build/html/_static/astropy_logo.svg')


    cmdclass['build_docs'] = build_docs

# Freeze build information in version.py. Note that this gets information
# about the package (name and version) from the setup.cfg file.
version = generate_version_py()

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()

setup(version=version, cmdclass=cmdclass, **package_info)
