import sys
import shutil
from pathlib import Path

def copy_models(sig, frame):
    out_dir = Path("/mnt/DATA/models/" + RUN_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pth in Path("./models/").glob("*.pth"):
        print(f"Copying {pth} to {out_dir}")
        shutil.move(pth, out_dir)

    print("Done copying.")

    sys.exit()


def check_or_make_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)
