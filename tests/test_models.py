#!/usr/bin/env python3

import mincepy
from mincepy import testing
from mincepy.testing import historian, archive_uri, mongodb_archive
import hubbardml

SPECIES = ("H", "C", "N", "F")


def test_umodel_load_save(historian: mincepy.Historian):  # noqa: F811
    graph = hubbardml.UGraph(SPECIES)
    reference = hubbardml.UModel(graph)
    loaded = testing.do_round_trip(historian, hubbardml.UModel, graph)

    assert reference.species == loaded.species


def test_vmodel_load_save(historian: mincepy.Historian):  # noqa: F811
    graph = hubbardml.VGraph(SPECIES)
    reference = hubbardml.VModel(graph)
    loaded = testing.do_round_trip(historian, hubbardml.VModel, graph)

    assert reference.species == loaded.species
