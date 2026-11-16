"""Reference points for P2: how predictable is final quality at day k at all?

* trivial  - always predict the training-set mean quality
* oracle   - gradient boosting on the TRUE hidden C-variant fraction trajectory
             up to day k (information no real sensor gives you).  This bounds
             what any sensor-based model can be expected to reach.

    python p2_long_horizon_transformer/oracle.py
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fermsim import chronological_split  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"


def main():
    d = np.load(ROOT / "data" / "campaigns.npz")
    q, c = d["quality"], d["c_fraction"]                 # c: (n, 720) hourly hidden state
    tr, cal, te = chronological_split(len(q))
    res = {"trivial_mean_mae": float(np.abs(q[te] - q[tr].mean()).mean())}
    for day in (5, 10, 15, 20):
        x = c[:, : day * 24 : 6]                          # 6-hourly hidden trajectory
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(x[tr], q[tr])
        res[f"oracle_mae_day{day}"] = float(np.abs(m.predict(x[te]) - q[te]).mean())
    OUT.mkdir(exist_ok=True)
    (OUT / "oracle.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
