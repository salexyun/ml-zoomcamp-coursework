import subprocess

version = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True).stdout.strip()
print("Q1. uv version:", version)
