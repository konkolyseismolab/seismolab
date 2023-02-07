import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__

#from . import fourier, gaia, template

__all__ = ['fourier', 'gaia', 'template']

#from seismolab.fourier import *
