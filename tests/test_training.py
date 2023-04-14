import mincepy
import torch.optim
from mincepy import testing
from mincepy.testing import historian, archive_uri, mongodb_archive

import hubbardml
from hubbardml import keys

SPECIES = ("C", "H", "F")


def test_trainer_save_load(historian: mincepy.Historian, dataframe):  # noqa: F811
    dataframe = dataframe[dataframe.param_type == "V"]
    dataframe = dataframe[~(dataframe[keys.ATOM_1_ELEMENT] == dataframe[keys.ATOM_2_ELEMENT])]

    species = set(dataframe[hubbardml.keys.ATOM_1_ELEMENT])

    model = hubbardml.VModel(hubbardml.VGraph(species))
    opt = (torch.optim.Adam(model.parameters(), lr=0.001),)
    loss_fn = (torch.nn.MSELoss(),)

    testing.do_round_trip(
        historian,
        hubbardml.Trainer.from_frame,
        model=model,
        opt=opt,
        loss_fn=loss_fn,
        frame=dataframe,
    )
