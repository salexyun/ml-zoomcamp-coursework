# ML Zoomcamp Coursework

My homework solutions for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Structure

Each module has its own folder:

```
01-intro/
├── homework.ipynb   # all questions with answers
├── q1.py … q7.py    # one script per question
└── data/            # dataset (not committed)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.14 is pinned in `.python-version`.

```bash
uv sync
```

Download the dataset for a module, e.g. for `01-intro`:

```bash
curl -L --create-dirs -o 01-intro/data/car_fuel_efficiency_2026.csv \
  https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

## Usage

Run a single question:

```bash
uv run python 01-intro/q1.py
```

Open the notebook (run from inside the module folder so data paths resolve):

```bash
cd 01-intro && uv run jupyter lab homework.ipynb
```

## Progress

| Module | Topic |
|---|---|
| [01-intro](01-intro/) | Introduction to Machine Learning |
| [02-regression](02-regression/) | Regression |
