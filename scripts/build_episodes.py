"""Build the open train or dev episodes from a versioned config file (docs/tasks/B_next.md items 9-10, case §09).

    python3 scripts/build_episodes.py {train,dev} [--out episodes] [--version v2] [--generator module:function]

Reads configs/<split>_<version>.json, generates every (seed, config) into <out>/<split>/<seed>/ with A's generator,
checks that each episode loads and its truth parses, and writes <out>/<split>/index.json with seed, episode_id,
config_sha256, clean and the number of issues. Regenerating is byte-identical, so a rebuild only shows real changes.
Without a generator module it prints SKIP and exits 0.
"""
import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditenv.config import config_sha256, normalize_config  # noqa: E402
from auditenv.episode import load_episode  # noqa: E402
from auditenv.grader import parse_truth  # noqa: E402

DEFAULT_GENERATOR = "auditenv.generate:generate_episode"
SEED_RANGES = {"train": range(1000, 2000), "dev": range(2000, 3000)}


def load_generator(spec):
    module_name, _, func = spec.partition(":")
    try:
        return getattr(importlib.import_module(module_name), func)
    except ImportError:
        return None


def load_split(split, version="v2"):
    """The (seed, normalized config) list of a split. ValueError on a seed outside the split's range."""
    path = ROOT / "configs" / f"{split}_{version}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for entry in doc["episodes"]:
        seed = entry["seed"]
        if type(seed) is not int or seed not in SEED_RANGES[split]:
            raise ValueError(f"{path.name}: seed {seed!r} outside the {split} range")
        rows.append((seed, normalize_config(entry["config"])))
    return doc["version"], rows


def build(split, out, generator, version="v2"):
    version_name, rows = load_split(split, version)
    split_dir = Path(out) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for seed, config in rows:
        episode_dir = split_dir / str(seed)
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        generator(seed, config, str(episode_dir))
        episode = load_episode(episode_dir)
        truth = episode.load_truth()
        parse_truth(truth, episode.aliases)
        index.append({"seed": seed, "episode_id": episode.episode_id, "config_sha256": config_sha256(config),
                      "clean": truth["clean"], "issues": len(truth["issues"])})
    doc = {"version": version_name, "split": split, "episodes": index}
    (split_dir / "index.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("split", choices=sorted(SEED_RANGES))
    parser.add_argument("--out", default=str(ROOT / "episodes"))
    parser.add_argument("--version", default="v2")
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    args = parser.parse_args(argv)
    generator = load_generator(args.generator)
    if generator is None:
        print(json.dumps({"build": "SKIP", "reason": f"{args.generator} not available yet"}))
        return 0
    doc = build(args.split, args.out, generator, args.version)
    clean = sum(e["clean"] for e in doc["episodes"])
    print(json.dumps({"build": "OK", "split": args.split, "version": doc["version"], "episodes": len(doc["episodes"]),
                      "clean": clean, "issues": sum(e["issues"] for e in doc["episodes"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
