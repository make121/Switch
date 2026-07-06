#!/usr/bin/env python3
"""
Build a Skill Graph (SG) from PBHC motion data.

Based on "Switch: Learning Agile Skills Switching for Humanoid Robots"
(arXiv:2604.14834)

Usage:
    # From a merged .pkl file (auto-splits by skill)
    python SG_build/build_sg.py --merged example/motion_data/merged_two.pkl

    # From individual skill files
    python SG_build/build_sg.py --files pose.pkl punch.pkl kick.pkl \\
        --labels horse_pose horse_punch roundhouse_kick

    # With custom thresholds
    python SG_build/build_sg.py --files a.pkl b.pkl \\
        --threshold 25.0 --topk 3 --buffer-base 2.5
"""

import argparse
import sys
from pathlib import Path

# Add project root
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from SG_build.skill_graph import (
    SkillGraphBuilder,
    build_skill_graph,
    load_motion_pkl,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build Skill Graph for multi-skill humanoid motion control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input sources
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--files", nargs="+", help="Individual .pkl files (one per skill)"
    )
    group.add_argument(
        "--merged", type=str, help="Single merged .pkl file (auto-splits by key)"
    )
    group.add_argument(
        "--folder", type=str, help="Directory of .pkl files (one per skill)"
    )

    # Options
    parser.add_argument("--labels", nargs="+", help="Skill name labels")
    parser.add_argument(
        "-o", "--output", default="./sg_output", help="Output directory"
    )
    parser.add_argument(
        "--threshold", type=float, default=30.0,
        help="Max L1 distance for cross-skill edges (default: 30.0)",
    )
    parser.add_argument(
        "--topk", type=int, default=5,
        help="Top-K nearest neighbors per frame (default: 5)",
    )
    parser.add_argument(
        "--buffer-base", type=float, default=1.0,
        help="L1 distance per buffer node (default: 1.0)",
    )
    parser.add_argument(
        "--max-buffer", type=int, default=30,
        help="Max buffer nodes per edge (default: 30)",
    )
    parser.add_argument(
        "--subsample", type=int, default=3,
        help="Subsample stride for source frames (default: 3)",
    )
    parser.add_argument(
        "--exclude-boundary", type=int, default=10,
        help="Exclude transitions whose src or dst falls within N frames "
             "of any skill boundary (default: 10)",
    )
    parser.add_argument(
        "--max-trajectories", type=int, default=10,
        help="Max trajectories to collect per skill pair (default: 10)",
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Override FPS for all motions"
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Don't merge original motions into output",
    )

    args = parser.parse_args()

    # Resolve input files
    if args.files:
        pkl_paths = args.files
    elif args.merged:
        print(f"Loading merged file: {args.merged}")
        data = load_motion_pkl(args.merged)
        print(f"  Found {len(data)} motion entries")
        # Write each entry to a temp file so SkillGraphBuilder can load them
        import tempfile
        import pickle

        tmpdir = Path(tempfile.mkdtemp(prefix="sg_split_"))
        pkl_paths = []
        for i, (key, motion) in enumerate(data.items()):
            fpath = tmpdir / f"skill_{i:03d}.pkl"
            with open(fpath, "wb") as f:
                pickle.dump({key: motion}, f)
            pkl_paths.append(str(fpath))
        print(f"  Split into {len(pkl_paths)} individual files in {tmpdir}")
    elif args.folder:
        folder = Path(args.folder)
        pkl_paths = sorted([str(p) for p in folder.glob("*.pkl")])
        print(f"Found {len(pkl_paths)} .pkl files in {folder}")

    # Labels
    labels = args.labels
    if labels is None:
        labels = [f"skill_{i:02d}" for i in range(len(pkl_paths))]
    if len(labels) != len(pkl_paths):
        print(f"Warning: {len(labels)} labels for {len(pkl_paths)} files; padding")
        while len(labels) < len(pkl_paths):
            labels.append(f"skill_{len(labels):02d}")

    print(f"\nSkills: {list(zip(labels, [Path(p).name for p in pkl_paths]))}")

    # Build
    builder = SkillGraphBuilder(
        motion_files=pkl_paths,
        skill_labels=labels,
        cross_skill_threshold=args.threshold,
        cross_skill_topk=args.topk,
        buffer_base_threshold=args.buffer_base,
        max_buffer_nodes=args.max_buffer,
        subsample_stride=args.subsample,
        exclude_boundary_frames=args.exclude_boundary,
        max_trajectories_per_pair=args.max_trajectories,
        fps=args.fps,
    )

    graph, augmented = builder.build()
    builder.print_stats()
    builder.save(args.output, merge_with_original=not args.no_merge)

    return 0


if __name__ == "__main__":
    sys.exit(main())
