from common import HERE

from_line = next(line for line in (HERE / "Dockerfile").read_text().splitlines() if line.startswith("FROM"))
image = from_line.split()[1].split("@")[0]
print("Q5. Base image:", image)
