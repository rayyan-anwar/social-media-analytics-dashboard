"""Tests for validation, analytics maths, generator ground truth, report output and (if installed) the HTTP API.

Author: Rayyan Anwar
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.analytics import (InputError, analyze, best_times, daily_table, enrich, hashtag_table, load_posts,
                           platform_table, report_markdown, sentiment_label, sentiment_score, spikes, trend)
from app.data import generate_posts

HEAD = "post_id,timestamp,platform,text,likes,shares,comments"


def csv(*rows):
    return "\n".join((HEAD,) + rows)


class TestValidation(unittest.TestCase):
    def test_valid_csv_and_columns_normalised(self):
        df, rep = load_posts(csv("1,2025-01-01T10:00:00,X,hello #a,5,1,2", "2,2025-01-02 11:00,Instagram,bye,0,0,0"))
        self.assertEqual(rep["rows_kept"], 2)
        self.assertEqual(list(df["platform"]), ["x", "instagram"])
        self.assertEqual(rep["date_range"], ["2025-01-01", "2025-01-02"])
        self.assertFalse(rep["has_followers"])

    def test_mixed_timestamp_formats_are_all_parsed(self):
        df, rep = load_posts(csv("1,2025-01-01T10:00:00Z,x,t,1,1,1", "2,2025-01-02 11:00,x,t,1,1,1",
                                 "3,Jan 5 2025 09:30,x,t,1,1,1", "4,garbage,x,t,1,1,1"))
        self.assertEqual((rep["rows_kept"], rep["dropped"]["invalid_timestamp"]), (3, 1))
        self.assertEqual(df["timestamp"].dt.day.tolist(), [1, 2, 5])

    def test_missing_columns_empty_and_garbage(self):
        for bad in ("a,b\n1,2", "", "   ", HEAD, "\x00\x01"):
            with self.assertRaises(InputError, msg=repr(bad)):
                load_posts(bad)
        with self.assertRaises(InputError):
            load_posts("post_id,timestamp\n1,2025-01-01")

    def test_bad_rows_are_dropped_and_counted(self):
        df, rep = load_posts(csv("1,2025-01-01,x,t,1,1,1", "2,nonsense,x,t,1,1,1", "3,2025-01-01,x,t,-1,0,0",
                                 "4,2025-01-01,x,t,abc,0,0", "1,2025-01-03,x,dup,9,9,9", "5,2025-01-01,x,,2,0,0"))
        self.assertEqual(rep["dropped"], {"invalid_timestamp": 1, "invalid_or_negative_counts": 2, "duplicate_post_id": 1})
        self.assertEqual(sorted(df["post_id"]), ["1", "5"])
        self.assertEqual(df.loc[df["post_id"] == "5", "text"].iloc[0], "")

    def test_all_invalid_raises_and_row_limit(self):
        with self.assertRaises(InputError):
            load_posts(csv("1,bad,x,t,1,1,1"))
        import app.analytics as a
        old, a.MAX_ROWS = a.MAX_ROWS, 2
        try:
            with self.assertRaises(InputError):
                load_posts(csv(*[f"{i},2025-01-01,x,t,1,1,1" for i in range(3)]))
        finally:
            a.MAX_ROWS = old

    def test_bom_and_optional_followers(self):
        text = "\ufeff" + HEAD + ",followers\n1,2025-01-01,x,t,1,1,1,100"
        df, rep = load_posts(text)
        self.assertTrue(rep["has_followers"])
        self.assertEqual(df["followers"].iloc[0], 100)


class TestSentiment(unittest.TestCase):
    def test_scores(self):
        self.assertGreater(sentiment_score("love this, great"), 0.9)
        self.assertLess(sentiment_score("terrible and broken"), -0.9)
        self.assertLess(sentiment_score("not good"), 0)
        self.assertGreater(sentiment_score("not bad"), 0)
        self.assertEqual(sentiment_score("meeting at noon"), 0.0)
        self.assertEqual(sentiment_score(None), 0.0)
        self.assertEqual([sentiment_label(x) for x in (0.5, 0.0, -0.5)], ["positive", "neutral", "negative"])


def tiny():
    df, _ = load_posts(csv(
        "1,2025-01-01T10:00:00,x,love it #a #B,10,0,0", "2,2025-01-01T11:00:00,x,awful #a,2,2,0",
        "3,2025-01-03T10:00:00,ig,ok #b,0,0,4", "4,2025-01-03T23:00:00,ig,great,6,0,0"))
    return df


class TestAnalytics(unittest.TestCase):
    def test_enrich_and_engagement_rate(self):
        d = enrich(tiny())
        self.assertEqual(list(d["engagement"]), [10, 4, 4, 6])
        self.assertTrue(d["engagement_rate"].isna().all())
        raw = pd.DataFrame({"post_id": ["a", "b"], "timestamp": ["2025-01-01"] * 2, "platform": "x", "text": "t",
                            "likes": [10, 10], "shares": 0, "comments": 0, "followers": [100, 0]})
        e = enrich(load_posts(raw)[0])
        self.assertAlmostEqual(e["engagement_rate"].iloc[0], 0.1)
        self.assertTrue(np.isnan(e["engagement_rate"].iloc[1]))

    def test_platform_table_hand_computed(self):
        t = {r["platform"]: r for r in platform_table(enrich(tiny()))}
        self.assertEqual(t["x"]["posts"], 2)
        self.assertEqual(t["x"]["total_engagement"], 14)
        self.assertAlmostEqual(t["ig"]["mean_engagement"], 5.0)

    def test_daily_fills_gaps_and_rolls(self):
        d = daily_table(enrich(tiny()))
        self.assertEqual(len(d), 3)
        self.assertEqual(list(d["engagement"]), [14, 0, 10])
        self.assertAlmostEqual(d["rolling7_engagement"].iloc[2], 8.0)

    def test_hashtags_case_insensitive(self):
        h = {x["hashtag"]: x for x in hashtag_table(enrich(tiny()))}
        self.assertEqual(h["#b"]["posts"], 2)
        self.assertEqual(h["#a"]["posts"], 2)
        self.assertAlmostEqual(h["#a"]["mean_engagement"], 7.0)

    def test_best_times_respects_min_posts(self):
        d = enrich(tiny())
        self.assertEqual(best_times(d, "hour", min_posts=3)["best"], [])  # no hour has three posts
        self.assertEqual(best_times(d, "hour", min_posts=2)["best"], [10])  # only hour 10 has two posts
        self.assertEqual(best_times(d, "hour", min_posts=1)["best"][0], 10)

    def test_trend_variants(self):
        idx = pd.date_range("2025-01-01", periods=30)
        mk = lambda v: pd.DataFrame({"engagement": v}, index=idx)
        self.assertEqual(trend(mk(np.arange(30) * 2.0 + 10))["direction"], "increasing")
        self.assertEqual(trend(mk(100 - np.arange(30) * 2.0))["direction"], "decreasing")
        self.assertEqual(trend(mk(np.full(30, 5.0)))["direction"], "flat")
        self.assertEqual(trend(mk(np.arange(30) * 1.0).iloc[:5])["direction"], "insufficient_data")
        noisy = mk(np.random.default_rng(0).normal(100, 5, 30))
        self.assertEqual(trend(noisy)["direction"], "flat")

    def test_spikes(self):
        idx = pd.date_range("2025-01-01", periods=30)
        y = np.full(30, 100.0) + np.random.default_rng(1).normal(0, 3, 30)
        y[12] = 900
        s = spikes(pd.DataFrame({"engagement": y}, index=idx))
        self.assertEqual([x["date"] for x in s], ["2025-01-13"])
        self.assertEqual(spikes(pd.DataFrame({"engagement": np.full(30, 5.0)}, index=idx)), [])
        const_with_one = np.full(30, 5.0)
        const_with_one[3] = 50
        self.assertEqual(len(spikes(pd.DataFrame({"engagement": const_with_one}, index=idx))), 1)
        self.assertEqual(spikes(pd.DataFrame({"engagement": [1, 2, 3]}, index=pd.date_range("2025-01-01", periods=3))), [])


class TestGeneratorAndReport(unittest.TestCase):
    def test_generator_deterministic_with_truth(self):
        (a, ta), (b, tb) = generate_posts(300, 30, 5), generate_posts(300, 30, 5)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(ta, tb)
        self.assertFalse(a.equals(generate_posts(300, 30, 6)[0]))
        with self.assertRaises(ValueError):
            generate_posts(5, 30)

    def test_planted_signals_recovered_on_default_dataset(self):
        df, truth = generate_posts(3000, 60, 0)
        a = analyze(*load_posts(df))
        self.assertIn(truth["spike_date"], [s["date"] for s in a["spikes"]])
        self.assertEqual(a["trend"]["direction"], "increasing")
        self.assertEqual(a["best_hours"]["best"][0], truth["best_hour"])
        self.assertEqual(a["hashtags_by_engagement"][0]["hashtag"], "#launch")

    def test_analysis_is_json_safe_and_report_complete(self):
        df, _ = generate_posts(400, 30, 2)
        a = analyze(*load_posts(df))
        json.dumps(a, allow_nan=False)
        md = report_markdown(a)
        for section in ("## Platforms", "## Engagement spikes", "## Top hashtags", "## Assumptions"):
            self.assertIn(section, md)

    def test_sample_csv_loads(self):
        text = (Path(__file__).resolve().parent.parent / "data" / "sample_posts.csv").read_text()
        df, rep = load_posts(text)
        self.assertEqual(rep["rows_kept"], 500)
        self.assertEqual(analyze(df, rep)["summary"]["posts"], 500)


try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    HAVE_API = True
except ImportError:
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed - API tests are skipped, not passed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.c = TestClient(app)

    def test_basics(self):
        self.assertEqual(self.c.get("/health").json()["author"], "Rayyan Anwar")
        self.assertIn("Rayyan Anwar", self.c.get("/").text)

    def test_sample_then_analyze_roundtrip(self):
        sample = self.c.get("/api/sample", params={"n": 300}).text
        r = self.c.post("/api/analyze", content=sample, headers={"content-type": "text/csv"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["summary"]["posts"], 300)
        self.assertIn("markdown", r.json())

    def test_errors(self):
        self.assertEqual(self.c.post("/api/analyze", content=b"a,b\n1,2", headers={"content-type": "text/csv"}).status_code, 422)
        self.assertEqual(self.c.post("/api/analyze", content=b"\xff\xfe\x00", headers={"content-type": "text/csv"}).status_code, 422)
        self.assertEqual(self.c.post("/api/analyze", content=b"x" * 5_000_001, headers={"content-type": "text/csv"}).status_code, 413)
        self.assertEqual(self.c.get("/api/sample", params={"n": 5}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
