"""Run-balanced scoring of optical spectra, vapor and CPI. Original Jeju fit objective."""

import numpy as np

WINDOWS = ["full", "prefix", "late"]
BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, np.inf)]


class Scoring:
    def scores(self, result):
        e = self.e
        y = result["y"]
        x = result["x"]
        dist = self.sim.distance(y) @ self.size_weight
        band_sim = y @ self.mask.T
        band_obs = e.obs @ self.mask.T
        band_err = abs(np.log10((1 + band_sim) / (1 + band_obs))).mean(axis=-1)
        scores = np.empty((6, 3, 6))
        for r in range(6):
            for wi, name in enumerate(WINDOWS):
                t, weights = self.profile[r, name]
                optical = dist[r, t] @ weights
                bands = band_err[r, t] @ weights
                vv = e.vapor[r, t]
                vok = np.isfinite(vv)
                vapor = (
                    float(
                        np.mean((result["vmodel"][r, t][vok] - vv[vok]) ** 2)
                        ** 0.5
                    )
                    if vok.any()
                    else np.nan
                )
                ci = e.cpi[r, t]
                use = ci.sum(axis=1) >= 40
                cpi = np.nan
                if use.any():
                    obs = ci[use]
                    mod = x[r, t[use], :8]
                    q = obs / obs.sum(axis=-1, keepdims=True)
                    qhat = mod / np.maximum(
                        mod.sum(axis=-1, keepdims=True), 1e-20
                    )
                    cpi = np.sqrt(
                        ((np.sqrt(q) - np.sqrt(qhat)) ** 2).sum(axis=-1) / 2
                    ).mean()
                plate = {1: 35, 5: 30}.get(r)
                habit = result["habit"][r] if plate in t else 0.0
                terms = np.array([optical, bands, vapor / 0.25, cpi + habit])
                channel_w = np.array([0.65, 0.20, 0.10, 0.05]) * np.isfinite(
                    terms
                )
                joint = np.nansum(terms * channel_w) / channel_w.sum()
                scores[r, wi] = [optical, bands, vapor, cpi, habit, joint]
        return scores
