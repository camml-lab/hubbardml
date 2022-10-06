from typing import List

import e3psi


class DSite(e3psi.IrrepsObj):
    """A d-block site with a species label and two occupation matrices"""

    def __init__(self, species: List[str]):
        super().__init__()
        # self.one = e3psi.Attr(o3.Irreps('0e'))
        self.specie = e3psi.SpecieOneHot(species)
        self.occs_1 = e3psi.OccuMtx("2e")
        self.occs_2 = e3psi.OccuMtx("2e")


class PSite(e3psi.IrrepsObj):
    """A p-block site with a species label and two occupation matrices"""

    def __init__(self, species: List[str]):
        super().__init__()
        # self.one = e3psi.Attr(o3.Irreps('0e'))
        self.specie = e3psi.SpecieOneHot(species)
        self.occs_1 = e3psi.OccuMtx("1e")
        self.occs_2 = e3psi.OccuMtx("1e")


class VEdge(e3psi.IrrepsObj):
    """An intersite V edge containing a Hubbard V value and a separation distance"""

    def __init__(self) -> None:
        super().__init__()
        # self.one = e3psi.Attr(o3.Irreps('0e'))
        self.v = e3psi.Attr("1x0e")
        self.dist = e3psi.Attr("1x0e")
