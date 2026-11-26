# Trained checkpoints

The checkpoints are in the Zenodo record LEN-Seek data ([doi:10.5281/zenodo.24491227](https://doi.org/10.5281/zenodo.24491227));
`lenseek-fetch <name>` downloads each of them (`checkpoint_v1_0` comes from the repository of version 1.0). Files are
stored in `$LENSEEK_HOME/assets/` (default `~/.cache/lenseek/assets/`); `LENSEEK_ASSET_MIRROR` can point to a folder
that holds a local copy instead. `lenseek-fetch` and
`lenseek-search` check each file's SHA-256 against `lenseek/configs/assets.yaml` and recognise the model from it. Each `.pth` holds the model and optimizer states after
the last of 200 epochs; there is no validation step, so no epoch was selected with evaluation-set quantities.

| Checkpoint | Training sites | Model in the paper | Configuration | Epoch |
|---|---|---|---|---|
| `main_s0` | all 114,661 database sites | LEN-Seek (default) | run `main_s0` of `data/runs.yaml` | 200 |
| `main_s1` | all 114,661 database sites | the same recipe, seed 1 | run `main_s1` of `data/runs.yaml` | 200 |
| `main_s2` | all 114,661 database sites | the same recipe, seed 2 | run `main_s2` of `data/runs.yaml` | 200 |
| `strict30_s0` | 101,977 sites (5,963 entries with ≥ 30 % identity to an evaluation protein removed) | the model trained without homologous entries | run `strict30_s0` of `data/runs.yaml` | 200 |

Each checkpoint is evaluated the same way: `scripts/evaluate.py` encodes the 481 sites of the evaluation data and the
database and writes the distances, and the analysis scripts read that folder.

```bash
python scripts/evaluate.py --checkpoint lenseek_main_s0.pth --features <database features> --queries <query cache> \
    --out $LENSEEK_EVAL_ROOT/out_main_s0
LENSEEK_EVAL_DIR=$LENSEEK_EVAL_ROOT/out_main_s0 python analysis/run.py table1
```

The label-based results (Table 1, Figures 4 and 5 and most Supplementary tables) also need the ProBiS and G-LoSA scans
of `examples/labels.sh`, which the LEN-Seek data record holds as asset `labels_probis_glosa`.

`main_s0` gives the LEN-Seek rows of the paper, `main_s1` and `main_s2` the seed-to-seed spread against which the
variants are read, and `strict30_s0` the rows of the model trained without homologous entries in the stratum tables. The
database features are in the data record (asset `train_features`); the query cache is made with `lenseek-featurize` from
the evaluation queries (asset `eval_queries`). The checkpoints of the other variants (the node-feature variants, the
model trained without homologous entries at 50 % identity and the model trained without the ligand filters) are not
released; `data/runs.yaml` trains each of them, and their evaluation outputs are in the LEN-Seek data record (asset
`derived_tables`).

The released latent-space dataset (`latentdb_main_s0`, 2.8 GB, and its standard-edition patch
`latentdb_main_s0_standard`) is the 114,661 database sites encoded with `main_s0`; `lenseek-search` uses it by default.
Queries searched against it must be encoded with the same checkpoint.

`checkpoint_v1_0` (`--model v1.0`) is the model of the bioRxiv preprint (v1); the v2.0 model supersedes it.

The other files the analysis scripts read (evaluation outputs, reference tables, labels, comparison-method outputs,
timing logs and scoring outputs of the binding sites from later PDB entries, all in the LEN-Seek data record) are
listed by `lenseek-fetch --list` with their SHA-256 in `lenseek/configs/assets.yaml`.

| File | SHA-256 |
|---|---|
| `lenseek_main_s0.pth` | `392d113bca4309e9ac2044cd0b186c682318fc22c2e397b5a5b74b66f034b290` |
| `lenseek_main_s1.pth` | `2c009f9017e395429582536589edb81641f6870d29b21d5019f039a60fbff4fe` |
| `lenseek_main_s2.pth` | `acd4ae48088e739880aa2d675dc9baa2a93f5d62602168c532a5c03c93d8e70e` |
| `lenseek_strict30_s0.pth` | `1cedc0d07ad32f845c9f7a74adfecc294c60693e837972f57dbdedb8b00e8da3` |
| `lenseek_latentdb_main_s0_paper.tar` | `eeae8b98a44aa3f38d14c2ad6e7a42c85dcfef391757520d03b25f86ac2ada8f` |
| `lenseek_latentdb_main_s0_standard.npz` | `e8b2631cf0c20877403ef1ed92b08da9a9422ad44962ea33052523076d4827f9` |
| `lenseek_v1.0.pth` | `8264cbee9c0a667c95a7ddb2d4543a9d4b814c8e272cad668ba2da62bb7e3555` |
