# DEAP EEG Emotion Recognition -- Comparative Analysis

Subject-independent emotion classification on the DEAP dataset: a classical-ML
pipeline on Differential Entropy features (Track A) vs. a deep-learning
pipeline on raw EEG windows (Track B), plus an eighteen-section investigation
into why subject-independent performance is weak, and a cross-corpus
validation against SEED-IV. See `Project_Report.pdf` / `Project_Report.tex`
for the full write-up.

## Datasets (not included in this repo)

The raw EEG datasets are large (multi-GB) and gated/licensed, so they are
**not committed to Git** (`deap-dataset/`, `seed-dataset/`, and the derived
`processed_cache/` are gitignored). Download them yourself and place them at
the paths below.

| Dataset | Link | Place at |
|---|---|---|
| DEAP (preprocessed Python `.dat` files) | <http://www.eecs.qmul.ac.uk/mmv/datasets/deap/> (registration/EULA required) | `deap-dataset/data_preprocessed_python/` |
| SEED-IV (Kaggle mirror, no agreement required) | <https://www.kaggle.com/datasets/phhasian0710/seed-iv> | `seed-dataset/` |
| SEED (original, 3-class) -- not used here | requires a signed BCMI data-use agreement | -- |
| DREAMER -- not used here | <https://ieee-dataport.org/open-access/dreamer> | -- |
| AMIGOS -- not used here | <http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/> | -- |

`processed_cache/` (cached `.npz` feature/window builds) is regenerated
automatically the first time the relevant notebook cell or script runs against
the raw data -- expect several minutes on a cache miss.

## Repository layout

```
Project_Report.tex / .pdf      -- the full report (methodology, results, figures/tables)
DEAP_Comparative_Analysis.ipynb -- main notebook (Track A/B, Sections 1-18)
scripts/                        -- standalone companion experiments (Track A/B, cross-corpus)
results/
  tables/                       -- CSV outputs from the notebook and scripts
  figures/                      -- PNG figures from the notebook and scripts
papers_EEG/                     -- literature-review corpus (32 PDFs)
citations.txt                   -- paper citation list
literature_log.txt              -- per-paper attribution/summary log
deap-dataset/, seed-dataset/    -- raw data (gitignored; download separately, see above)
processed_cache/                -- cached feature builds (gitignored; regenerated automatically)
```
