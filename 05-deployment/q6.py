import json

from common import HERE, LEAD_Q4, container, health, predict

expected_sha = json.loads((HERE / "model_metadata.json").read_text())["model_sha256"]

with container():
    status = health()
    result = predict(LEAD_Q4)

assert status["status"] == "ok"
assert status["model_sha256"] == expected_sha
print("Health:", status)
print("Response:", result)
print("Q6. Conversion probability:", round(result["conversion_probability"], 3))
