import numpy as np
import pandas as pd
import pytest
from drone_delivery.model import prepare_data, train_and_evaluate, flag_heavy


def sample_frame():
    rng = np.random.default_rng(13)
    n = 60
    speed = rng.choice([4, 6, 8, 10, 12], size=n)
    altitude = rng.choice([25, 50, 75, 100], size=n).astype(float)
    payload = 150 + 10 * speed + .4 * altitude + rng.normal(0, 10, n)
    return pd.DataFrame({"speed": speed, "altitude": altitude, "payload": payload})


def test_training_pipeline():
    experiment = train_and_evaluate(sample_frame())
    assert len(experiment.predictions) == 20
    assert np.isfinite(experiment.mse)
    assert np.isfinite(experiment.threshold)
    assert experiment.r2 > .5


def test_imputation():
    frame = sample_frame()
    frame.loc[0, "altitude"] = np.nan
    assert np.isfinite(train_and_evaluate(frame).mse)


def test_missing_required_column():
    with pytest.raises(ValueError, match="Missing"):
        prepare_data(pd.DataFrame({"speed": [1, 2]}))


def test_flag_boundary():
    assert not flag_heavy(280, 280)
    assert flag_heavy(281, 280)
