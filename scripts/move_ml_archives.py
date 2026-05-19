from pathlib import Path
import shutil

project_root = Path(__file__).resolve().parents[1]
ml_dir = project_root / "3_ml"
archive = ml_dir / "archive"
archive.mkdir(parents=True, exist_ok=True)

moved = []
skipped = []

for p in ml_dir.iterdir():
    if p.is_file():
        name = p.name
        lower = name.lower()
        # Move parquet files and checkpoint files
        if p.suffix == ".parquet" or "checkpoint" in lower:
            dest = archive / name
            try:
                shutil.move(str(p), str(dest))
                moved.append(name)
            except Exception as e:
                skipped.append((name, str(e)))

print("Moved files:\n", "\n".join(moved))
if skipped:
    print("Skipped with errors:\n", skipped)