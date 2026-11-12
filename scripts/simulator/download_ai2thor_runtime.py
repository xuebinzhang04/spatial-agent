"""Download the AI2-THOR Unity build matched to the installed Python package.

Run from the SpatialAgent root with 02_Simulator/AI2THOR/runtime/python/bin/python.
This does not launch Unity.
"""

from pathlib import Path

from ai2thor.build import Build, COMMIT_ID
from ai2thor.platform import CloudRendering


if not COMMIT_ID:
    raise RuntimeError("Installed ai2thor package has no build commit ID")

root = Path(__file__).resolve().parents[2]
releases_dir = root / "02_Simulator" / "AI2THOR" / "runtime" / "releases"
build = Build(CloudRendering, COMMIT_ID, False, releases_dir=str(releases_dir))
print(f"Downloading {build.name} to {releases_dir}", flush=True)
build.download()
print(build.executable_path, flush=True)
