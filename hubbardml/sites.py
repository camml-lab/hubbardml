from typing import Iterable

import e3psi
from e3nn import o3


class PSite(e3psi.IrrepsObj):
    """A p-block site with a species label and two occupation matrices"""

    def __init__(self, species: Iterable[str]):
        super().__init__()
        self.specie = e3psi.SpecieOneHot(species)
        self.occs_1 = e3psi.OccuMtx("1o")
        self.occs_2 = e3psi.OccuMtx("1o")


class DSite(e3psi.IrrepsObj):
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
        self.one = e3psi.Attr(o3.Irreps("0e"))
        self.v = e3psi.Attr("1x0e")
        self.dist = e3psi.Attr("1x0e")
