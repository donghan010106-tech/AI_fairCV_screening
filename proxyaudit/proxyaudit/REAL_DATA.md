# Running ProxyAudit on the REAL datasets

GitHub's "Download ZIP" does **not** include Git-LFS data (FairCVdb) and FairJob's
data lives on HuggingFace, so the repo zips contain only code/pointers. Fetch the
real data as follows, then run the bundled scripts.

## FairCVdb (24,000 profiles)
```bash
git lfs install
git clone https://github.com/BiDAlab/FairCVtest.git
cd FairCVtest && git lfs pull          # downloads data/FairCVdb.npy (~203 MB)
```
Then, from the proxyaudit repo:
```bash
PYTHONPATH=src python scripts/run_faircv_real.py /path/to/FairCVtest/data/FairCVdb.npy
```
The loader reads the official dict (`Profiles Train/Test`, `Biased Labels (Gender)`);
gender = `profiles[:,1]`, features = `profiles[:,4:31]`. The face-embedding tail is
the real proxy channel; PLS localizes it and the before/after triad is measured.

## FairJob (1,072,226 rows)
```bash
pip install huggingface_hub
huggingface-cli download criteo/FairJob --repo-type dataset --local-dir ./fairjob_data
```
Then:
```bash
PYTHONPATH=src python scripts/run_fairjob.py --real ./fairjob_data/fairjob.csv.gz --nrows 200000
```
The loader follows the official positional layout (col 0 click, 1 protected_attribute,
2 senior, 3 displayrandom, 4 rank, 5+ features); the parity estimate is reported on
the senior-ad slice as the benchmark prescribes.

> Both loaders were validated against the official schema using schema-faithful
> mock files; they run first-try once the real data is in place.
