import json
import pickle
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).parent
URL = "http://localhost:9696"

LEAD_Q3 = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

LEAD_Q4 = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}


def load_pipeline():
    with open(HERE / "pipeline.bin", "rb") as f:
        return pickle.load(f)


def health():
    with urlopen(f"{URL}/health") as response:
        return json.load(response)


def predict(lead):
    request = Request(
        f"{URL}/predict",
        data=json.dumps(lead).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.load(response)


def wait_for_health(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return health()
        except (URLError, ConnectionError):
            time.sleep(0.5)
    raise TimeoutError(f"API at {URL} did not become healthy")


@contextmanager
def local_server():
    proc = subprocess.Popen(
        ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"],
        cwd=HERE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_health()
        yield
    finally:
        proc.terminate()
        proc.wait()


@contextmanager
def container(tag="zoomcamp-model:2026-hw5", name="hw5"):
    subprocess.run(["docker", "build", "-q", "-t", tag, str(HERE)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", "9696:9696", tag],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    try:
        wait_for_health()
        yield
    finally:
        subprocess.run(["docker", "stop", name], check=True, stdout=subprocess.DEVNULL)
