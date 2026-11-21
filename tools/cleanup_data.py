"""Delete personal recordings once the project is done.

Removes everything under data/ (face videos, slow-mo clips, landmark recordings, calibration
sessions, per-user trained models, profiles). Keeps reports/ (aggregate numbers and plots of
synthetic data) and the code. Lists what it will delete and asks for confirmation.

    python tools/cleanup_data.py            # dry run + confirm
    python tools/cleanup_data.py --yes      # no prompt
"""
import shutil
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    if not DATA.exists():
        print("nothing to delete: data/ does not exist"); return
    files = [p for p in DATA.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"data/ contains {len(files)} files, {size:.0f} MB:")
    for p in sorted(files)[:40]:
        print("  ", p.relative_to(DATA))
    if len(files) > 40:
        print(f"   ... and {len(files) - 40} more")
    if "--yes" not in sys.argv and input("Permanently delete all of data/? [y/N] ").strip().lower() != "y":
        print("kept"); return
    shutil.rmtree(DATA)
    print("deleted data/")


if __name__ == "__main__":
    main()
