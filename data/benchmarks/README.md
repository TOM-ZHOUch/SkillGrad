# Datasets

SkillGrad's training and evaluation scripts both read SpreadsheetBench
Verified from `data/benchmarks/spreadsheetbench/`. The dataset is not
bundled with this repository.

## Expected layout

```
data/benchmarks/spreadsheetbench/
├── dataset.json
└── spreadsheet/
    ├── <sample_id>/
    │   ├── 1_<sample_id>_init.xlsx       # input workbook
    │   ├── 1_<sample_id>_golden.xlsx     # reference answer
    │   └── ...                            # additional test cases (if any)
    └── ...
```

`data/load.py` accepts the three naming conventions that ship in
SpreadsheetBench releases (`*_input/_answer`, `*_init/_golden`, and the
bare `initial.xlsx/golden.xlsx` outliers), so any of the official
distributions will work.

## Fetching the data

SpreadsheetBench is released by Ma et al. (2024). Follow the upstream
instructions to download the verified subset and place the files under
`data/benchmarks/spreadsheetbench/` as shown above.

- Upstream: <https://spreadsheetbench.github.io/>
- The "verified-400" subset used in the SkillGrad paper covers 400 tasks;
  the canonical 200/200 evolution/held-out split is then produced
  on-the-fly by `data/split.py` from a fixed `--master-seed` /
  `--heldout-seed` pair.
