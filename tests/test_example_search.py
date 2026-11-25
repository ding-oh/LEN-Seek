"""The example search of examples/search.sh: the four example queries against the example database return, for every
query, the ten sites of data/example/expected_hits.tsv in the same order, at the distances listed there."""
import csv
import os
import subprocess
import sys

import pytest

from conftest import EXAMPLES, ROOT


def expected():
    out = {}
    for row in csv.DictReader(open(EXAMPLES / "expected_hits.tsv"), delimiter="\t"):
        out.setdefault(row["query"], []).append((int(row["rank"]), row["site_id"], float(row["chamfer_distance_paper"])))
    return out


@pytest.mark.assets
def test_example_search_reproduces_expected_hits(main_checkpoint, tmp_path):
    out = tmp_path / "hits.tsv"
    cmd = [sys.executable, "-m", "lenseek.cli.search", str(EXAMPLES / "queries"), "--embeddings",
           str(EXAMPLES / "embeddings"), "--db", str(EXAMPLES / "db" / "example_db.npz"), "--checkpoint",
           main_checkpoint, "--top-k", "10", "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert p.returncode == 0, p.stderr
    got = {}
    for row in csv.DictReader(open(out), delimiter="\t"):
        got.setdefault(row["query"], []).append(row)
    exp = expected()
    assert sorted(got) == sorted(exp)
    for q, hits in exp.items():
        assert [int(r["rank"]) for r in got[q]] == [h[0] for h in hits]
        assert [r["site_id"] for r in got[q]] == [h[1] for h in hits], q
        for r, (_, _, d) in zip(got[q], hits):
            assert float(r["chamfer_distance"]) == pytest.approx(d, abs=0.05 + 2e-3 * d)
