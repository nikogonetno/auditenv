"""Test doubles for A's transform: permute_and_rename(episode_dir, out_dir, salt). Never used outside tests."""
import csv
import json
import shutil
from pathlib import Path


def copy_unchanged(episode_dir, out_dir, salt):
    """The identity transform: the copy keeps every row, order and id, so every paired check must pass."""
    shutil.copytree(episode_dir, out_dir, dirs_exist_ok=True)


def drop_one_issue(episode_dir, out_dir, salt):
    """A broken transform: the copy's truth loses its first issue, so the transform check must fail."""
    copy_unchanged(episode_dir, out_dir, salt)
    path = Path(out_dir) / "private" / "truth.json"
    truth = json.loads(path.read_text(encoding="utf-8"))
    truth["issues"] = truth["issues"][1:]
    path.write_text(json.dumps(truth, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rename_without_truth(episode_dir, out_dir, salt):
    """A broken transform: ledger row ids are renamed (reversed) but truth keeps the old ones."""
    copy_unchanged(episode_dir, out_dir, salt)
    path = Path(out_dir) / "public" / "ledger.csv"
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, list(reader)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "row_id": row["row_id"][::-1]})
