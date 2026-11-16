"""Project 2 - Early prediction of final product quality from long process
time series, with calibrated uncertainty.

Question: after observing only the first k days of a 30-day campaign, how well
can we predict the final paste quality - and can we put an honest error bar
on that prediction?

Models (same inputs, same chronological split):
    * Gradient boosting on hand-crafted summary features (strong baseline)
    * LSTM over the 2-hourly sensor sequence
    * Transformer encoder over 12-hour patches of the sequence

Uncertainty: split conformal prediction on a held-out calibration block of
campaigns gives 90 % prediction intervals with a finite-sample coverage
guarantee (under exchangeability).

Evaluation is chronological - train on early campaigns, calibrate on the next
block, test on the latest - so models are always scored on "future"
campaigns, including the effect of slow plant drift.

    python scripts/generate_data.py
    python p2_long_horizon_transformer/train.py        # ~10-20 min on CPU
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fermsim import chronological_split  # noqa: E402
from fermsim.training import fit, set_seed, split_conformal  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
STEP_H = 2                      # resample to 2-hourly
PATCH = 6                       # 6 x 2 h = 12-hour patches
torch.set_num_threads(8)


# ----------------------------------------------------------------------------- models
class LSTMRegressor(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


class PatchTransformer(nn.Module):
    """Patch the sequence into 12 h tokens, add learned positions, prepend a
    CLS token, and regress quality from the CLS output."""

    def __init__(self, n_in, max_tokens=64, d=64, layers=3, heads=4):
        super().__init__()
        self.embed = nn.Linear(n_in * PATCH, d)
        self.pos = nn.Parameter(torch.randn(1, max_tokens + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        b, t, c = x.shape
        tokens = self.embed(x.reshape(b, t // PATCH, PATCH * c))
        z = torch.cat([self.cls.expand(b, -1, -1), tokens], 1) + self.pos[:, : tokens.shape[1] + 1]
        return self.head(self.encoder(z)[:, 0]).squeeze(-1)


# ----------------------------------------------------------------------------- data
def load(days):
    d = np.load(ROOT / "data" / "campaigns.npz")
    s = d["sensors"]
    n, H, c = s.shape
    s = s.reshape(n, H // STEP_H, STEP_H, c).mean(2)          # 2-hourly
    x = s[:, : days * 24 // STEP_H]
    return x.astype(np.float32), d["quality"].astype(np.float32)


def summary_features(x):
    t = np.arange(x.shape[1])
    slope = np.polyfit(t, x.transpose(1, 0, 2).reshape(len(t), -1), 1)[0].reshape(x.shape[0], x.shape[2])
    last = x[:, -12:]
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), last.mean(1), last.std(1), slope], 1)


def run_horizon(days, seeds=(0, 1, 2)):
    x, y = load(days)
    tr, cal, te = chronological_split(len(y))
    n_val = int(0.15 * len(tr))
    fit_idx, val_idx = tr[:-n_val], tr[-n_val:]          # early-stopping block (still before cal)

    mu, sd = x[fit_idx].mean((0, 1)), x[fit_idx].std((0, 1)) + 1e-6
    xn = (x - mu) / sd
    ym, ys = y[fit_idx].mean(), y[fit_idx].std()
    yn = (y - ym) / ys

    preds = {}
    # gradient boosting baseline
    f = summary_features(x)
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
    hgb.fit(f[tr], y[tr])
    preds["Gradient boosting"] = [hgb.predict(f)]

    def loader(idx, shuffle):
        ds = TensorDataset(torch.tensor(xn[idx]), torch.tensor(yn[idx]))
        return DataLoader(ds, batch_size=32, shuffle=shuffle)

    loss_fn = lambda m, b: nn.functional.mse_loss(m(b[0]), b[1])  # noqa: E731
    for name, make in [("LSTM", lambda: LSTMRegressor(x.shape[2])),
                       ("Transformer", lambda: PatchTransformer(x.shape[2]))]:
        preds[name] = []
        for s in seeds:
            set_seed(s)
            m = fit(make(), loader(fit_idx, True), loader(val_idx, False), loss_fn, epochs=120, lr=1e-3)
            with torch.no_grad():
                preds[name].append(m(torch.tensor(xn)).numpy() * ys + ym)

    res = {}
    for name, ps in preds.items():
        maes = [float(np.mean(np.abs(p[te] - y[te]))) for p in ps]
        p = np.mean(ps, 0)                                   # seed ensemble for intervals
        q = split_conformal(p[cal] - y[cal], alpha=0.1)
        cover = float(np.mean(np.abs(p[te] - y[te]) <= q))
        res[name] = dict(test_mae_mean=float(np.mean(maes)), test_mae_std=float(np.std(maes)),
                         interval_halfwidth_90=q, test_coverage_90=cover, pred=p)
    return res, y, te


def main():
    OUT.mkdir(exist_ok=True)
    horizons = [5, 10, 15, 20]
    all_res = {}
    for d in horizons:
        res, y, te = run_horizon(d)
        all_res[d] = res
        print(f"day {d:2d}: " + " | ".join(f"{k}: MAE {v['test_mae_mean']:.2f}+/-{v['test_mae_std']:.2f}, "
                                          f"cov {v['test_coverage_90']:.2f}, +/-{v['interval_halfwidth_90']:.1f}"
                                          for k, v in res.items()), flush=True)
        if d == 10:
            showcase = (res, y, te)

    # figure 1: error vs how much of the campaign has been observed
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"Gradient boosting": "tab:green", "LSTM": "tab:orange", "Transformer": "tab:blue"}
    for name, c in colors.items():
        m = [all_res[d][name]["test_mae_mean"] for d in horizons]
        e = [all_res[d][name]["test_mae_std"] for d in horizons]
        ax[0].errorbar(horizons, m, yerr=e, marker="o", capsize=3, color=c, label=name)
    ax[0].set_xlabel("days of campaign observed"); ax[0].set_ylabel("test MAE (quality points)")
    ax[0].set_title("Final-quality prediction error vs. observation window"); ax[0].legend()

    # figure 2: conformal intervals on test campaigns at day 10 (transformer)
    res, y, te = showcase
    r = res["Transformer"]
    order = np.argsort(r["pred"][te])
    p, q, yt = r["pred"][te][order], r["interval_halfwidth_90"], y[te][order]
    xs = np.arange(len(te))
    ax[1].fill_between(xs, p - q, p + q, color="tab:blue", alpha=0.2, label="90 % conformal interval")
    ax[1].plot(xs, p, color="tab:blue", lw=1, label="prediction")
    inside = np.abs(yt - p) <= q
    ax[1].scatter(xs[inside], yt[inside], s=6, color="black", label="actual (inside)")
    ax[1].scatter(xs[~inside], yt[~inside], s=10, color="red", label="actual (outside)")
    ax[1].set_xlabel("test campaigns (sorted by prediction)"); ax[1].set_ylabel("final quality")
    ax[1].set_title(f"Day-10 transformer: coverage {r['test_coverage_90']:.0%}"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "quality_prediction.png", dpi=150)

    clean = {d: {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in res.items()}
             for d, res in all_res.items()}
    (OUT / "metrics.json").write_text(json.dumps(clean, indent=2))


if __name__ == "__main__":
    main()
