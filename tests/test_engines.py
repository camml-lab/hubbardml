import torch

from hubbardml import engines


def test_mae():
    x, y = torch.rand(100, 1), torch.rand(100, 1)
    stats_mae = torch.abs(x - y).mean()

    engine = engines.Engine(torch.nn.Identity())
    mae = engines.Mae("loss")
    engine.add_engine_listener(mae)

    y_pred = engine.run(zip(x, y), return_outputs=True)
    assert torch.allclose(x, y_pred.reshape(x.shape))
    assert torch.isclose(engine.metrics["loss"], stats_mae)
    assert torch.isclose(mae.compute(), stats_mae)

    # Now, try a different batch size
    x, y = torch.rand(25, 4), torch.rand(25, 4)
    stats_mae = torch.abs(x - y).mean()
    y_pred = engine.run(zip(x, y), return_outputs=True)
    assert torch.allclose(x, y_pred.reshape(x.shape))
    assert torch.isclose(engine.metrics["loss"], stats_mae)
    assert torch.isclose(mae.compute(), stats_mae)
