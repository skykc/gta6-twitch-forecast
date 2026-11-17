"""gta6-twitch-ccu-forecast-01 entry: comparable-launch ratio forecast.

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2..24 of release day (2026-11-19 UTC) as five quantiles per hour.

Method (no network, stdlib only, fully deterministic):
  Each comparable past launch in the case file has hourly Twitch category
  averages for its own release day.  For a comparable, hour 1's print is the
  category's viewers at 00:00 UTC of its release day, and the launch's shape
  is the profile of ratios r_H = viewers(H) / viewers(1) for H = 2..24.
  The ratio profile depends strongly on how much of the world had already
  unlocked by 00:00 UTC: launches whose hour-1 print is small (few regions
  live yet) ramp many-fold as the unlock waves arrive, while launches whose
  hour-1 print is already huge barely grow.  Across the comparables,
  log(peak ratio) falls almost linearly in log10(hour-1 print) (r ~ -0.94).

  So for each forecast hour H we fit a Gaussian-kernel weighted regression
  of log r_H on log10(hour-1 print), with kernel bandwidth 1.0 dex centred
  on this case's (clamped) log10 hour-1 print: nearby launches count most,
  which also keeps the fit sane when this case's print lies beyond the
  comparables' range.  The forecast median is p1 * exp(mu_H) and the spread
  is the smoothed weighted residual standard deviation, halved (the raw
  residual spread overstates the conditional dispersion on the practice
  cases).  Quantiles are log-normal: q_tau = p1 * exp(mu_H + z_tau * s_H).

  If anything about the comparables is unusable (no valid comparable, a
  parse failure, a degenerate fit), the program falls back to the house
  persistence baseline (every hour at hour 1's print, spread widening as
  0.15*sqrt(h-1)) so the case is still valid.

  Practice validation (local, against SullyGnome's hourly averages):
  mean pinball loss about 0.19 over the four practice cases, versus about
  0.40 for the house baseline.  See gta6-twitch-validate.md (kept outside
  this repository).

Run: python3 main.py reads $MAKEFUN_INPUT_DIR (else /input) and writes
one <id>.out per <id>.in to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA_BASE = 0.15
BANDWIDTH = 1.0
SPREAD_MULT = 0.5
SPREAD_MIN = 0.12
SPREAD_MAX = 0.80


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2_half_up(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def hour1_print(case):
    """Hour-1 print from the known row, else the last history point, else 1."""
    for row in case.get("known", []):
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    hist = case.get("history") or []
    if hist:
        last = hist[-1].get("viewers")
        if last is not None:
            return max(1.0, float(last))
    return 1.0


def comparable_profiles(case):
    """(name, hour-1 print, {forecast hour: ratio}) for usable comparables.

    A comparable is usable when its release-day hour-1 print is positive
    and at least 23 of its 24 release-day hours have a positive point.
    """
    profiles = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours = {}
        for point in comp.get("hourly") or []:
            try:
                when = parse_time(point["from"])
                viewers = point["viewers"]
            except (KeyError, TypeError, ValueError):
                continue
            if viewers is None:
                continue
            h = int((when - day_start).total_seconds() // 3600)
            if 0 <= h < 24:
                hours[h] = float(viewers)
        first = hours.get(0)
        if not first or first <= 0:
            continue
        present = sum(
            1 for h in range(24) if h in hours and hours[h] is not None and hours[h] > 0
        )
        if present < 23:
            continue
        ratios = {}
        for hour in range(2, 25):
            value = hours.get(hour - 1)
            if value is not None and value > 0:
                ratios[hour] = value / first
        if len(ratios) < 22:
            continue
        profiles.append((str(comp.get("name", "")), first, ratios))
    profiles.sort(key=lambda item: item[0])
    return profiles


def weighted_ols(points, weights):
    """Weighted least squares y ~ a + b*x; returns (a, b, weighted resid sd)."""
    total = sum(weights)
    if total <= 0 or len(points) < 3:
        return None
    mean_x = sum(w * x for w, (x, _y) in zip(weights, points)) / total
    mean_y = sum(w * y for w, (_x, y) in zip(weights, points)) / total
    var_x = sum(w * (x - mean_x) ** 2 for w, (x, _y) in zip(weights, points))
    if var_x <= 0:
        return None
    slope = (
        sum(w * (x - mean_x) * (y - mean_y) for w, (x, y) in zip(weights, points))
        / var_x
    )
    intercept = mean_y - slope * mean_x
    resid_sq = sum(
        w * (y - (intercept + slope * x)) ** 2 for w, (x, y) in zip(weights, points)
    )
    dof = max(total - 2.0, 1.0)
    return intercept, slope, math.sqrt(resid_sq / dof)


def smooth(values, radius=1):
    """Centred moving average, radius 1 (edges use what is there)."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def model_quantiles(p1, profiles, levels):
    """Median + spread per forecast hour from the comparable profiles."""
    log_p1 = math.log10(p1)
    logs = [math.log10(first) for _name, first, _r in profiles]
    lo, hi = min(logs), max(logs)
    centre = clamp(log_p1, lo, hi)
    zvals = [statistics.NormalDist().inv_cdf(tau) for tau in levels]
    medians = {}
    spreads = []
    for hour in range(2, 25):
        points = []
        weights = []
        for _name, first, ratios in profiles:
            if hour not in ratios:
                continue
            x = math.log10(first)
            points.append((x, math.log(ratios[hour])))
            weights.append(math.exp(-0.5 * ((x - centre) / BANDWIDTH) ** 2))
        fit = weighted_ols(points, weights)
        if fit is None:
            return None
        intercept, slope, sd = fit
        medians[hour] = intercept + slope * centre
        spreads.append(sd)
    spreads = smooth(spreads)
    rows = []
    for hour in range(2, 25):
        spread = clamp(spreads[hour - 2] * SPREAD_MULT, SPREAD_MIN, SPREAD_MAX)
        quantiles = []
        previous = 0.0
        for z in zvals:
            q = p1 * math.exp(medians[hour] + z * spread)
            q = clamp(q, 0.01, 1e9)
            q = round2_half_up(q)
            if q < previous:
                q = previous
            quantiles.append(q)
            previous = q
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def baseline_quantiles(p1, levels):
    """House persistence baseline: every hour at hour 1's print."""
    zvals = [statistics.NormalDist().inv_cdf(tau) for tau in levels]
    rows = []
    for hour in range(2, 25):
        quantiles = []
        previous = 0.0
        for z in zvals:
            q = p1 * math.exp(z * SIGMA_BASE * math.sqrt(hour - 1))
            q = clamp(q, 0.01, 1e9)
            q = round2_half_up(q)
            if q < previous:
                q = previous
            quantiles.append(q)
            previous = q
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def forecast_case(case):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    p1 = hour1_print(case)
    try:
        profiles = comparable_profiles(case)
        if len(profiles) >= 5:
            forecast = model_quantiles(p1, profiles, levels)
            if forecast is not None:
                return forecast
    except Exception:
        pass
    return baseline_quantiles(p1, levels)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
