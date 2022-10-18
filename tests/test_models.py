#!/usr/bin/env python3

import mincepy
from mincepy import testing
from mincepy.testing import historian, archive_uri, mongodb_archive
from hubbardml import models

SPECIES = ("H", "C", "N", "F")


def test_umodel_load_save(historian: mincepy.Historian):  # noqa: F811
    reference = models.UModel(SPECIES)
    loaded = testing.do_round_trip(historian, models.VModel, SPECIES)

    assert reference.species == loaded.species


def test_vmodel_load_save(historian: mincepy.Historian):  # noqa: F811
    reference = models.VModel(SPECIES)
    loaded = testing.do_round_trip(historian, models.VModel, SPECIES)

    assert reference.species == loaded.species
