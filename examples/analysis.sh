#!/bin/bash
# The figures and tables of the manuscript that are computed from intermediate results (analysis/run.py), and the
# steps of scripts/ that write the intermediate files they read and that no other example script runs.
#
#   bash examples/analysis.sh inputs             # site annotation and MMseqs2 searches of the database chains,
#                                                #   identity, structure and ligand strata of the evaluation queries,
#                                                #   per-site checks of the BsitePDB files and the re-encoded sites,
#                                                #   top-100 lists of the sequence-identity and random baselines
#   bash examples/analysis.sh query-sets         # data/query_sets/ again (query set, sequence clusters)
#   bash examples/analysis.sh pfam               # SIFTS chain-to-Pfam and Pfam clan tables, downloaded (read only
#                                                #   when $LENSEEK_REFERENCE_DIR/pfam/ is absent)
#   bash examples/analysis.sh timing STEP        # one timing run of Table 2 and Supplementary Table S17: probis-prep,
#                                                #   probis, glosa, lenseek-cpu, lenseek-gpu, prep-cpu, prep-gpu,
#                                                #   deeplytough, dt-net, dt-search, dt-prep, folddisco (16 cores;
#                                                #   *-gpu on a GPU; examples/timing.sh runs the CPU steps one after
#                                                #   another on one node)
#   bash examples/analysis.sh                    # every result whose inputs are present, then a table of what ran,
#                                                #   what was skipped (with the missing inputs) and what failed
#   bash examples/analysis.sh table1 fig3 cases  # the named results only
#   python analysis/run.py --list                # the result names and the manuscript item of each
#
# Produces  inputs: $LENSEEK_OUT/cache/site_annot/, site_annotation/ (protein_strata.tsv, query_strata.csv,
#           strata/), database_scan/, baselines/top100/; query-sets:
#           $LENSEEK_OUT/site_annotation/query_sets/; pfam: $LENSEEK_OUT/cache/pfam/; timing: $LENSEEK_OUT/timing/;
#           the results: one folder per result below $LENSEEK_OUT (default out/ in the repository):
#           database_derivation/, table1/, score_agreement/, fig4/<model>/, model_figures/, cases/,
#           table2/, ablations/, factorial/, filter_retrain/, table3_two_stage/, holdout_score/, ...
# Inputs    see `python analysis/run.py -h` for the results and the docstrings of the scripts that inputs, query-sets
#           and timing run; read here: LENSEEK_CKPT (timing lenseek-*, prep-*), DT_PYTHON (deeplytough, dt-net,
#           dt-search), DT_RUNS (dt-prep) and FOLDDISCO (folddisco; the rewritten files of examples/labels.sh
#           fd-prepare). pfam: SIFTS is updated weekly; the copy in $LENSEEK_REFERENCE_DIR/pfam/
#           is the one the results used. labels.sh, holdout.sh and scaling.sh produce the inputs of the label-based
#           results, the results of the binding sites from later PDB entries and Supplementary Table S16; results
#           whose inputs are missing are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export LENSEEK_OUT=$(realpath -m "${LENSEEK_OUT:-$PWD/out}")
[ $# -gt 0 ] || exec python analysis/run.py --all
T="python -u scripts/timing.py"
case "$1" in
  inputs)
    python scripts/site_annotation.py annotate
    python scripts/site_annotation.py db-search
    python scripts/site_annotation.py identity-strata
    python scripts/site_annotation.py strata
    for s in sites coords; do python scripts/database_scan.py $s --workers 8; done
    python scripts/database_scan.py patch
    python scripts/baseline_features.py top100 ;;
  query-sets)
    python scripts/site_annotation.py query-sets
    python scripts/site_annotation.py clusters ;;
  pfam)
    P=$LENSEEK_OUT/cache/pfam E=https://ftp.ebi.ac.uk/pub/databases
    mkdir -p "$P"
    curl -fsSL -o "$P/Pfam-A.clans.tsv.gz" $E/Pfam/releases/Pfam38.2/Pfam-A.clans.tsv.gz
    curl -fsSL -o "$P/pdb_chain_pfam.tsv.gz" $E/msd/sifts/flatfiles/tsv/pdb_chain_pfam.tsv.gz
    md5sum "$P"/*.tsv.gz ;;
  timing)
    L=$LENSEEK_OUT/single_vector/cache
    case "${2:?timing step, see the header}" in
      probis-prep) $T probis-prep ;;
      probis)      $T probis ;;
      glosa)       $T glosa ;;
      lenseek-cpu) $T lenseek --checkpoint "${LENSEEK_CKPT:?}" --latent-db "$L" --device cpu ;;
      lenseek-gpu) $T lenseek --checkpoint "${LENSEEK_CKPT:?}" --latent-db "$L" --device cuda --threads 6 ;;
      prep-cpu)    $T lenseek-prep --checkpoint "${LENSEEK_CKPT:?}" --device cpu --sample 500 ;;
      prep-gpu)    $T lenseek-prep --checkpoint "${LENSEEK_CKPT:?}" --device cuda --threads 6 ;;
      deeplytough|dt-net|dt-search)   # DeeplyTough environment, from a scratch directory (DeeplyTough writes ./cache)
                   S=$PWD/scripts/timing.py W=$(mktemp -d "${TMPDIR:-/tmp}/dt_XXXX"); trap 'rm -rf "$W"' EXIT
                   (cd "$W" && "${DT_PYTHON:?}" -u "$S" "$2") ;;
      dt-prep)     $T dt-prep --runs ${DT_RUNS:?DeeplyTough run folders} ;;
      folddisco)   $T folddisco ;;
      *) echo "unknown timing step $2" >&2; exit 2 ;;
    esac ;;
  *) for r in "$@"; do python analysis/run.py "$r"; done ;;
esac
