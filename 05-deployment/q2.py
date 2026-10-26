import tomllib

from common import HERE

lock = tomllib.loads((HERE / "uv.lock").read_text())
sklearn = next(p for p in lock["package"] if p["name"] == "scikit-learn")
print("Q2. Locked scikit-learn version:", sklearn["version"])
