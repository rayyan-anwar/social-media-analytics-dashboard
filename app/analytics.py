"""Validation and analytics for imported social-media post datasets (pandas, framework-independent).

Required columns: post_id, timestamp, platform, text, likes, shares, comments. Optional: author, followers.
Timestamps are interpreted as UTC. Engagement = likes + shares + comments (unweighted).

Author: Rayyan Anwar
"""
from __future__ import annotations

import io
import math
import re
import warnings

import numpy as np
import pandas as pd

REQUIRED = ("post_id", "timestamp", "platform", "text", "likes", "shares", "comments")
MAX_ROWS = 200_000
POS = frozenset("""love loved great awesome amazing excellent fantastic happy proud thrilled excited good best wonderful
brilliant enjoy enjoyed delighted win winning congrats thanks thank""".split())
NEG = frozenset("""hate hated terrible awful horrible bad worst angry disappointed disappointing annoying sad broken fail
failed failing slow scam outage upset frustrated frustrating""".split())
NEGATORS = frozenset({"not", "never", "no"})
_TOK = re.compile(r"[a-z']+")
_TAG = re.compile(r"#\w+")


class InputError(ValueError):
    """Invalid dataset or parameters."""


def sentiment_score(text: str) -> float:
    """Lexicon sentiment in [-1, 1] with simple negation; 0 when no lexicon words are present."""
    toks = _TOK.findall(str(text).lower())
    p = n = 0
    for i, t in enumerate(toks):
        v = 1 if t in POS else -1 if t in NEG else 0
        if v and i and toks[i - 1] in NEGATORS:
            v = -v
        p += v > 0
        n += v < 0
    return (p - n) / (p + n) if p + n else 0.0


def sentiment_label(score: float) -> str:
    return "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"


def load_posts(source) -> tuple[pd.DataFrame, dict]:
    """Parse and validate CSV text or a DataFrame. Returns (clean_df, validation_report)."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        if not isinstance(source, str) or not source.strip():
            raise InputError("CSV is empty")
        try:
            df = pd.read_csv(io.StringIO(source.lstrip("\ufeff")))
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise InputError(f"could not parse CSV: {exc}") from None
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputError(f"missing required columns: {missing}")
    if len(df) > MAX_ROWS:
        raise InputError(f"too many rows ({len(df)}); limit is {MAX_ROWS}")
    rows_in = len(df)
    raw_ts = df["timestamp"]
    ts = pd.to_datetime(raw_ts, utc=True, errors="coerce", format="ISO8601")  # fast path for ISO-like values
    retry = ts.isna() & raw_ts.notna()
    if retry.any():  # other formats (e.g. "Jan 5 2025 10:00"): parsed element-wise; truly invalid values stay NaT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ts.loc[retry] = pd.to_datetime(raw_ts[retry], utc=True, errors="coerce", format="mixed")
    counts = {c: pd.to_numeric(df[c], errors="coerce") for c in ("likes", "shares", "comments")}
    bad_ts = ts.isna()
    bad_counts = pd.concat([(v.isna() | (v < 0)) for v in counts.values()], axis=1).any(axis=1) & ~bad_ts
    keep = ~(bad_ts | bad_counts)
    out = pd.DataFrame({
        "post_id": df["post_id"].astype(str),
        "timestamp": ts.dt.tz_localize(None),
        "platform": df["platform"].astype(str).str.strip().str.lower(),
        "text": df["text"].fillna("").astype(str),
        "likes": counts["likes"].fillna(0).astype(np.int64),
        "shares": counts["shares"].fillna(0).astype(np.int64),
        "comments": counts["comments"].fillna(0).astype(np.int64),
        "author": df["author"].astype(str) if "author" in df.columns else "unknown",
        "followers": pd.to_numeric(df["followers"], errors="coerce") if "followers" in df.columns else np.nan,
    })[keep]
    dup = out["post_id"].duplicated()
    out = out[~dup].reset_index(drop=True)
    if out.empty:
        raise InputError("no valid rows after validation")
    report = {"rows_in": rows_in, "rows_kept": int(len(out)),
              "dropped": {"invalid_timestamp": int(bad_ts.sum()), "invalid_or_negative_counts": int(bad_counts.sum()),
                          "duplicate_post_id": int(dup.sum())},
              "has_followers": bool(out["followers"].notna().any()),
              "date_range": [str(out["timestamp"].min().date()), str(out["timestamp"].max().date())]}
    return out, report


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["engagement"] = d["likes"] + d["shares"] + d["comments"]
    f = d["followers"].where(d["followers"] > 0)
    d["engagement_rate"] = d["engagement"] / f
    d["sentiment_score"] = d["text"].map(sentiment_score)
    d["sentiment"] = d["sentiment_score"].map(sentiment_label)
    d["date"] = d["timestamp"].dt.normalize()
    d["hour"] = d["timestamp"].dt.hour
    d["weekday"] = d["timestamp"].dt.dayofweek
    d["hashtags"] = d["text"].str.lower().str.findall(_TAG)
    return d


def platform_table(d: pd.DataFrame) -> list[dict]:
    g = d.groupby("platform")
    t = pd.DataFrame({"posts": g.size(), "total_engagement": g["engagement"].sum(),
                      "mean_engagement": g["engagement"].mean(), "median_engagement": g["engagement"].median(),
                      "mean_sentiment_score": g["sentiment_score"].mean(),
                      "mean_engagement_rate": g["engagement_rate"].mean()}).reset_index()
    return t.sort_values("total_engagement", ascending=False).to_dict("records")


def sentiment_summary(d: pd.DataFrame) -> dict:
    counts = d["sentiment"].value_counts()
    return {"counts": {k: int(counts.get(k, 0)) for k in ("positive", "neutral", "negative")},
            "share": {k: float(counts.get(k, 0) / len(d)) for k in ("positive", "neutral", "negative")},
            "mean_engagement_by_sentiment": {k: float(v) for k, v in d.groupby("sentiment")["engagement"].mean().items()}}


def daily_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("date")
    t = pd.DataFrame({"posts": g.size(), "engagement": g["engagement"].sum(), "mean_sentiment": g["sentiment_score"].mean()})
    t = t.reindex(pd.date_range(t.index.min(), t.index.max(), freq="D"))
    t[["posts", "engagement"]] = t[["posts", "engagement"]].fillna(0)
    t["rolling7_engagement"] = t["engagement"].rolling(7, min_periods=1).mean()
    t.index.name = "date"
    return t


def trend(daily: pd.DataFrame, min_days: int = 14) -> dict:
    """Linear slope of daily engagement plus Spearman rank correlation with time (a heuristic, not a forecast)."""
    y = daily["engagement"].to_numpy(float)
    if len(y) < min_days:
        return {"direction": "insufficient_data", "days": int(len(y)), "min_days": min_days}
    x = np.arange(len(y))
    slope = float(np.polyfit(x, y, 1)[0])
    rel = slope / y.mean() if y.mean() else 0.0
    rho = float(pd.Series(y).corr(pd.Series(x), method="spearman")) if np.ptp(y) > 0 else 0.0
    direction = "increasing" if rho > 0.3 and rel > 0.005 else "decreasing" if rho < -0.3 and rel < -0.005 else "flat"
    return {"direction": direction, "slope_per_day": slope, "relative_slope_per_day": rel, "spearman_rho": rho,
            "days": int(len(y))}


def spikes(daily: pd.DataFrame, threshold: float = 3.5) -> list[dict]:
    """Days whose engagement has a robust z-score (median/MAD) above `threshold`."""
    y = daily["engagement"].to_numpy(float)
    if len(y) < 7:
        return []
    med = np.median(y)
    scale = 1.4826 * np.median(np.abs(y - med))
    if scale == 0:
        scale = float(np.std(y))
    if scale == 0:
        return []
    z = (y - med) / scale
    return [{"date": str(daily.index[i].date()), "engagement": float(y[i]), "robust_z": float(z[i])}
            for i in np.argsort(-z) if z[i] > threshold]


def hashtag_table(d: pd.DataFrame, k: int = 10, min_posts: int = 1) -> list[dict]:
    e = d[["hashtags", "engagement"]].explode("hashtags").dropna(subset=["hashtags"])
    if e.empty:
        return []
    g = e.groupby("hashtags")["engagement"]
    t = pd.DataFrame({"posts": g.size(), "mean_engagement": g.mean()}).reset_index().rename(columns={"hashtags": "hashtag"})
    t = t[t["posts"] >= min_posts]
    return t.sort_values(["posts", "mean_engagement"], ascending=False).head(k).to_dict("records")


def best_times(d: pd.DataFrame, by: str, min_posts: int = 5, top: int = 3) -> dict:
    g = d.groupby(by)["engagement"]
    t = pd.DataFrame({"posts": g.size(), "mean_engagement": g.mean()}).reset_index()
    ok = t[t["posts"] >= min_posts].sort_values("mean_engagement", ascending=False)
    return {"table": t.to_dict("records"), "best": ok.head(top)[by].tolist(), "min_posts": min_posts}


def top_posts(d: pd.DataFrame, k: int = 5) -> list[dict]:
    cols = ["post_id", "platform", "engagement", "sentiment", "text"]
    return d.sort_values("engagement", ascending=False).head(k)[cols].assign(text=lambda x: x["text"].str[:140]).to_dict("records")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if (math.isnan(x) or math.isinf(x)) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return x


def analyze(df: pd.DataFrame, validation: dict | None = None) -> dict:
    d = enrich(df)
    daily = daily_table(d)
    rows = [{"date": str(i.date()), "posts": int(r.posts), "engagement": float(r.engagement),
             "mean_sentiment": r.mean_sentiment, "rolling7_engagement": float(r.rolling7_engagement)}
            for i, r in daily.iterrows()]
    return jsonable({
        "validation": validation or {},
        "summary": {"posts": len(d), "total_engagement": int(d["engagement"].sum()),
                    "mean_engagement": float(d["engagement"].mean()), "median_engagement": float(d["engagement"].median()),
                    "mean_engagement_rate": float(d["engagement_rate"].mean()) if d["engagement_rate"].notna().any() else None,
                    "platforms": int(d["platform"].nunique())},
        "platforms": platform_table(d), "sentiment": sentiment_summary(d), "daily": rows,
        "trend": trend(daily), "spikes": spikes(daily), "hashtags": hashtag_table(d),
        "hashtags_by_engagement": sorted(hashtag_table(d, 50, min_posts=max(5, len(d) // 100)),
                                         key=lambda h: -h["mean_engagement"])[:10],
        "best_hours": best_times(d, "hour", min_posts=max(5, len(d) // 100)),
        "best_weekdays": best_times(d, "weekday", min_posts=max(5, len(d) // 100)), "top_posts": top_posts(d),
        "assumptions": ["Engagement = likes + shares + comments, unweighted.", "Timestamps interpreted as UTC.",
                        "Sentiment is a small English lexicon with negation: a rough signal, not a validated classifier.",
                        "Trend and spike flags are descriptive heuristics, not forecasts or causal claims.",
                        "Best-time estimates ignore content type and are shown only for slots with enough posts."]})


def report_markdown(a: dict) -> str:
    s, v, t = a["summary"], a["validation"], a["trend"]
    L = ["# Social media analytics report", "", "Generated by Social Media Analytics Dashboard (Rayyan Anwar).", "",
         f"- Posts analysed: {s['posts']} (rows in: {v.get('rows_in', 'n/a')}; dropped: {v.get('dropped', {})})",
         f"- Total engagement: {s['total_engagement']}; mean per post: {s['mean_engagement']:.1f}; median: {s['median_engagement']:.1f}",
         f"- Trend: {t['direction']}" + (f" (slope {t['slope_per_day']:.1f}/day, Spearman rho {t['spearman_rho']:.2f})" if 'slope_per_day' in t else ""),
         f"- Sentiment shares: " + ", ".join(f"{k} {100 * x:.0f}%" for k, x in a["sentiment"]["share"].items()), "",
         "## Platforms", "", "| Platform | Posts | Total engagement | Mean | Median |", "|---|---:|---:|---:|---:|"]
    L += [f"| {p['platform']} | {p['posts']} | {p['total_engagement']} | {p['mean_engagement']:.1f} | {p['median_engagement']:.1f} |" for p in a["platforms"]]
    L += ["", "## Engagement spikes (robust z > 3.5)", ""] + ([f"- {x['date']}: {x['engagement']:.0f} (z={x['robust_z']:.1f})" for x in a["spikes"]] or ["- none"])
    L += ["", "## Top hashtags", ""] + ([f"- {h['hashtag']}: {h['posts']} posts, mean engagement {h['mean_engagement']:.1f}" for h in a["hashtags"]] or ["- none"])
    L += ["", "## Hashtags by mean engagement (enough posts only)", ""] + (
        [f"- {h['hashtag']}: mean {h['mean_engagement']:.1f} over {h['posts']} posts" for h in a["hashtags_by_engagement"]] or ["- none"])
    L += ["", f"## Best posting hours (UTC, slots with at least {a['best_hours']['min_posts']} posts)", "", ", ".join(map(str, a["best_hours"]["best"])) or "n/a"]
    L += ["", "## Assumptions and limitations", ""] + [f"- {x}" for x in a["assumptions"]]
    return "\n".join(L) + "\n"
