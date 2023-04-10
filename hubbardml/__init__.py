"""
Please refer to the documentation provided in the README.md
"""
from .version import *
from .projects import *
from .training import *
from . import datasets
from . import keys
from . import models
from . import parse_pw
from . import plots
from . import projects
from . import similarities
from . import sites
from . import training
from . import utils
from . import version

__all__ = (
    models.__all__
    + training.__all__
    + version.__all__
    + projects.__all__
    + ("datasets", "keys", "models", "plots", "sites", "utils", "version", "similarities")
)
