# AGENTS.md — SFE (Speech Feature Extraction / Vowel Beat Detector)

## Setup

- Conda environment: `rtm` (from `env.yml`). Create with `conda env create -f env.yml`.
- Slurm scripts use `rtm_clone` instead — differs from `env.yml`. Verify which is active.

## Entry point

```
python src/main.py -in <IN_DIR> -out <OUT_DIR> -env <bark|vocallic_energy> -sr <RATE>
```

**Main script is `src/main.py`**.

Required arguments: `-in`, `-out`, `-sr`, `-env`.

## Core modules

- `src/feat_extractor.py` — `sfe` class with two envelope extraction strategies (`bark`, `vocallic_energy`). Also handles pitch (Praat), wavelet, eGeMAPS (opensmile), and plotting.
- `src/mfcc.py` — MFCC extraction around beat frames.
- `src/main.py` — CLI pipeline that orchestrates batch processing into LMDB.

## Output format

- LMDB file `feats.lmdb` at output dir. Keys are `"dirname_filename.wav"` (last two path components joined with `_`).
- Values are `pickle.dumps()` of a dict. Load with:
  ```python
  env = lmdb.open(path, readonly=True)
  entry = pickle.loads(txn.get(key.encode()))
  ```
- A `metadata.txt` summary is also written.

## Resumability

The pipeline skips files whose identifier already exists in the target LMDB. If you delete or rename the LMDB, all files are reprocessed.

## Testing / verification

No test framework, no lint/format config. The only test-like artifacts are:
- `test/test_samples.sh` — one-off script that runs `main.py` against a specific data path on the cluster.
- `test/inspect_lmdb.py` — ad-hoc script to peek into an LMDB file (hardcoded path).

## Slurm

Batch job scripts live in `.slurm/`. They use `conda activate rtm_clone` and run `python src/main.py` against different datasets on the cluster filesystem (`/hadatasets/...`). Update conda env name and paths if running elsewhere.
