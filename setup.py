from os import path
import sys
from setuptools import setup

sys.path.insert(0, "seismolab")
from version import __version__

# Load requirements
requirements = None
with open('requirements.txt') as file:
    requirements = file.read().splitlines()

# If Python3: Add "README.md" to setup.
# Useful for PyPI. Irrelevant for users using Python2.
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

# Command-line tools
entry_points = {'console_scripts': [
    'query_gaia = seismolab.gaia:query_from_commandline'
]}

desc='Codes to process variable star data from Kepler-TESS-Gaia surveys.'

setup(  name='seismolab',
        version=__version__,
        description=desc,
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Attila Bodi',
        author_email='bodi.attila@csfk.org',
        url='https://github.com/konkolyseismolab/',
        #package_dir={'seismolab':'src'},
        packages=['seismolab',
            'seismolab.fourier',
            'seismolab.gaia',
            'seismolab.OC',
            'seismolab.template',
        ],
        install_requires=requirements,
        entry_points=entry_points
    )
