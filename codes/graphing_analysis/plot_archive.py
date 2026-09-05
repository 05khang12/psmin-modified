from datetime import date
from pathlib import Path
from shutil import copy2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / f"graphs_{date.today().isoformat()}"


def archive_plot(path, archive_dir=None):
    src = Path(path).resolve()
    outdir = Path(archive_dir).resolve() if archive_dir else DEFAULT_ARCHIVE
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / src.name
    if target.exists() and target.resolve() != src:
        target = outdir / f"{src.parent.name}_{src.name}"
    copy2(src, target)
    return target
