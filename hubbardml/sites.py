from typing import Iterable

import e3psi


class Site(e3psi.IrrepsObj):
    """Parent class for a site"""


class PSite(Site):
    """A p-block site with a species label and two occupation matrices"""

    def __init__(self, species: Iterable[str]):
        super().__init__()
        self.specie = e3psi.SpecieOneHot(species)
        self.occs_1 = e3psi.OccuMtx("1o")
        self.occs_2 = e3psi.OccuMtx("1o")


class DSite(Site):
    """A d-block site with a species label and two occupation matrices"""

    def __init__(self, species: Iterable[str]):
        super().__init__()
        self.specie = e3psi.SpecieOneHot(species)
        self.occs_1 = e3psi.OccuMtx("2e")
        self.occs_2 = e3psi.OccuMtx("2e")


class PDSite(e3psi.IrrepsObj):
    """A site that has a p and d block occupations"""

    def __init__(self, species: Iterable[str]):
        super().__init__()
        self.specie = e3psi.SpecieOneHot(species)
        self.p_occs_1 = e3psi.OccuMtx("1o")
        self.p_occs_2 = e3psi.OccuMtx("1o")
        self.d_occs_1 = e3psi.OccuMtx("2e")
        self.d_occs_2 = e3psi.OccuMtx("2e")


class VEdge(e3psi.IrrepsObj):
    """An intersite V edge containing a Hubbard V value and a separation distance"""

    def __init__(self) -> None:
        super().__init__()
        self.v = e3psi.Attr("1x0e")  # The Hubbard V value
        self.dist = e3psi.Attr("1x0e")  # The distance between the two sites
