# -*- coding: utf-8 -*-
"""Plugins for MincePy"""
from . import models
from . import training


def get_mincepy_types():
    """The central entry point to provide historian type helpers"""
    types = list()
    types.extend(models.HISTORIAN_TYPES)
    types.extend(training.HISTORIAN_TYPES)

    return types
