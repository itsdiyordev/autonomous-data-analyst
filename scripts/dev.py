"""Start both development services. Run with: python scripts/dev.py."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
uv = shutil.which("uv")
npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
if not uv or not npm:
    sys.exit("Install uv and Node.js 22.12+ first. See README.md.")

subprocess.run([uv, "sync", "--python", "3.12"], cwd=root / "backend", check=True)
subprocess.run([npm, "install"], cwd=root / "frontend", check=True, shell=os.name == "nt")
processes = [
    subprocess.Popen([uv, "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"], cwd=root / "backend"),
    subprocess.Popen([npm, "run", "dev"], cwd=root / "frontend", shell=os.name == "nt"),
]
print("\nAnalytiq → http://localhost:5173 | API docs → http://localhost:8000/docs\n")
try:
    processes[0].wait()
except KeyboardInterrupt:
    pass
finally:
    for process in processes:
        process.terminate()
