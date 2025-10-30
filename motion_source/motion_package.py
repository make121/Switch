import argparse
import pickle
import re
import sys
from pathlib import Path

import joblib
import torch


def load_pkl(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e1:
        try:
            print(f"Pickle failed, trying joblib for {pkl_path}")
            return joblib.load(pkl_path)
        except Exception as e2:
            try:
                print(f"Joblib failed, trying torch.load for {pkl_path}")
                return torch.load(pkl_path, map_location="cpu")
            except Exception as e3:
                raise RuntimeError(f"Failed to load {pkl_path} as pickle, joblib, or torch.\nErrors:\n- pickle: {e1}\n- joblib: {e2}\n- torch: {e3}")


def merge_motion_files(pkl_paths, failed_list_path=None, min_len=None, max_len=None):
    merged = {}
    key_sources = {}  # key -> source path
    dropped = []  # dropped motions
    pattern = re.compile(r"^(?:\S+)\s+(.+)\s+([0-9]*\.?[0-9]+)$")

    if failed_list_path:
        failed_file_stems = set()
        with open(failed_list_path, "r") as f:
            for line in f:
                line = line.strip()
                m = pattern.match(line)
                if not m:
                    print(f"Skipping invalid line: {line}")
                    continue
                filepath = m.group(1)
                score_str = m.group(2)
                try:
                    score = float(score_str)
                except ValueError:
                    print(f"Skipping line with invalid score: {line}")
                    continue
                if score < 0.8:
                    failed_file_stems.add(Path(filepath).name)
        print(f"Loaded {len(failed_file_stems)} failed motion file stems with score < 0.8.")

    for pkl_path in pkl_paths:
        pkl_path = Path(pkl_path)
        print(f"Loading: {pkl_path}")
        data = load_pkl(pkl_path)

        if not isinstance(data, dict):
            raise ValueError(f"{pkl_path} does not contain a dict")

        for k, motion in data.items():
            if not isinstance(motion, dict) or "dof" not in motion:
                print(f"Skipping key '{k}' in {pkl_path}: missing 'dof'")
                continue

            motion_len = motion["dof"].shape[0]
            if (min_len and motion_len < min_len) or (max_len and motion_len > max_len):
                dropped.append((k, motion_len, str(pkl_path)))
                continue

            # if pkl_path.name in failed_file_stems:
            #     print(f"Skipping failed motion file: {pkl_path.name}")
            #     continue

            new_key = k 

            key_sources[new_key] = str(k)
            merged[new_key] = motion

    print(f"Dropped {len(dropped)} motions due to length constraints.")
    for k, l, path in dropped:
        print(f"  - Dropped {k} ({l} frames) from {path}")

    print(f"After filtering, total motions count: {len(merged)}")

    return merged, key_sources

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder with .pkl motion files")
    parser.add_argument("--min_len", type=int, default=0, help="Minimum allowed motion length (frames)")
    parser.add_argument(
        "--max_len",
        type=int,
        default=5000,
        help="Maximum allowed motion length (frames)",
    )
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    pkl_paths = sorted(folder_path.glob("**/*.pkl"))
    num_pkl = len(pkl_paths)

    if num_pkl == 0:
        print(f"No .pkl files found in {folder_path}")
        sys.exit(1)
    else:
        print(f"Found {num_pkl} .pkl files in {folder_path}")

    print(f"Found {len(pkl_paths)} pkl files.")

    merged_data, key_sources = merge_motion_files(pkl_paths, failed_list_path=None, min_len=args.min_len, max_len=args.max_len)

    output_path = folder_path / "merged_motion.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(merged_data, f)

    source_txt_path = folder_path / "merged_motion_keys.txt"
    with open(source_txt_path, "w") as f:
        for i, (key, path) in enumerate(key_sources.items()):
            f.write(f"{i}\t{path}\n")

    print(f"Merged motion saved to: {output_path.resolve()}")
    print(f"Key source mapping saved to: {source_txt_path.resolve()}")


if __name__ == "__main__":
    main()