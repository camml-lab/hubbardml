#!/usr/bin/env python3

import mincepy
from mincepy import testing
from mincepy.testing import historian, archive_uri, mongodb_archive
from hubbardml import models

SPECIES = ("H", "C", "N", "F")


def test_umodel_load_save(historian: mincepy.Historian):  # noqa: F811
    graph = models.UGraph(SPECIES)
    reference = models.UModel(graph)
    loaded = testing.do_round_trip(historian, models.UModel, graph)

    assert reference.species == loaded.species


def test_vmodel_load_save(historian: mincepy.Historian):  # noqa: F811
    graph = models.VGraph(SPECIES)
    reference = models.VModel(graph)
    loaded = testing.do_round_trip(historian, models.VModel, graph)

    assert reference.species == loaded.species
