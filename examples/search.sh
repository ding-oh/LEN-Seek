#!/bin/bash
# Search the four example binding sites against the 500-site example database (CPU; no download beyond the model checkpoint).
#
#   examples/search.sh [top_k]
#
# Each hit list matches data/example/expected_hits.tsv, the paper's search of the full database, in the same order.
set -euo pipefail
cd "$(dirname "$0")/.."
lenseek-search data/example/queries --embeddings data/example/embeddings --db data/example/db/example_db.npz --top-k "${1:-10}"
