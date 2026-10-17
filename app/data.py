"""Seeded synthetic post generator with PLANTED ground-truth signals for validating the analytics.

Planted: upward engagement trend, one spike day, best posting hour, one high-engagement hashtag.
Text uses generic positive/negative/neutral phrases and may include negation, so sentiment checks are not
perfectly trivial, but the generator and the lexicon share an author: treat those scores as a sanity check.

Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLATFORMS = ["x", "instagram", "linkedin", "facebook"]
PLATFORM_MULT = {"x": 0.8, "instagram": 1.4, "linkedin": 0.9, "facebook": 1.0}
POS_P = ["love the new update", "great launch today", "amazing support from the team", "so happy with this", "excellent results this week"]
NEG_P = ["terrible outage again", "so disappointed with the delay", "awful experience today", "this is broken and slow", "hate the new layout"]
NEU_P = ["new post is live", "meeting notes shared", "schedule for next week", "photo from the event", "reading list updated"]
TAGS = ["#news", "#tips", "#team", "#product", "#event", "#launch"]
BEST_HOUR = 19
SPIKE_MULT = 4.0
LAUNCH_MULT = 2.5
TREND_TOTAL_GROWTH = 0.8  # engagement baseline grows 80% across the period


def generate_posts(n: int = 3000, days: int = 60, seed: int = 0, start: str = "2025-03-01") -> tuple[pd.DataFrame, dict]:
    if n < 10 or days < 14:
        raise ValueError("need n >= 10 and days >= 14")
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    day = rng.integers(0, days, n)
    hours = rng.choice(24, n, p=_hour_probs())
    spike_day = int(rng.integers(days // 4, days - 3))
    authors = [f"user{i}" for i in range(40)]
    followers = {a: int(rng.integers(500, 50_000)) for a in authors}
    rows = []
    for i in range(n):
        label = rng.choice(["positive", "negative", "neutral"], p=[0.4, 0.2, 0.4])
        phrase = rng.choice({"positive": POS_P, "negative": NEG_P, "neutral": NEU_P}[label])
        if label != "neutral" and rng.random() < 0.15:  # negation flips the stated sentiment
            phrase = "not " + phrase
            label = "negative" if label == "positive" else "positive"
        tags = [t for t in TAGS if rng.random() < (0.10 if t == "#launch" else 0.18)]
        platform = str(rng.choice(PLATFORMS))
        author = str(rng.choice(authors))
        mean = 40 * PLATFORM_MULT[platform] * (1 + TREND_TOTAL_GROWTH * day[i] / days)
        mean *= 2.0 if hours[i] == BEST_HOUR else 1.0
        mean *= LAUNCH_MULT if "#launch" in tags else 1.0
        mean *= SPIKE_MULT if day[i] == spike_day else 1.0
        likes = int(rng.negative_binomial(4, 4 / (4 + mean * 0.7)))
        shares = int(rng.negative_binomial(2, 2 / (2 + mean * 0.1)))
        comments = int(rng.negative_binomial(2, 2 / (2 + mean * 0.2)))
        ts = start_ts + pd.Timedelta(days=int(day[i]), hours=int(hours[i]), minutes=int(rng.integers(0, 60)))
        rows.append((f"p{seed}_{i}", ts.isoformat(), platform, f"{phrase} {' '.join(tags)}".strip(), likes, shares, comments,
                     author, followers[author], label))
    df = pd.DataFrame(rows, columns=["post_id", "timestamp", "platform", "text", "likes", "shares", "comments",
                                     "author", "followers", "true_sentiment"])
    truth = {"spike_date": str((start_ts + pd.Timedelta(days=spike_day)).date()), "best_hour": BEST_HOUR,
             "top_engagement_hashtag": "#launch", "trend": "increasing"}
    return df, truth


def _hour_probs() -> np.ndarray:
    base = np.ones(24)
    base[[12, 13]] += 1.5
    base[[18, 19, 20]] += 2.5
    base[0:6] *= 0.3
    return base / base.sum()
