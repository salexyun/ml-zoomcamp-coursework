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

`05-deployment` has its own pinned environment (Python 3.11.15, `uv.lock`) to match the frozen model artifact, so sync and run it from inside that folder:

```bash
cd 05-deployment && uv sync --locked
uv run python q3.py
uv run --with jupyter jupyter lab homework.ipynb
```

`q4.py` starts the API locally and `q6.py` builds and runs the Docker container (Docker must be running).

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
| [03-classification](03-classification/) | Classification |
| [04-evaluation](04-evaluation/) | Evaluation |
| [05-deployment](05-deployment/) | Deployment |
