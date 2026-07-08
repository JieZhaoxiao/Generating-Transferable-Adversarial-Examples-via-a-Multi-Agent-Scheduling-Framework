import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bundled or generated adversarial examples.")
    parser.add_argument("--input_dir", default="data", help="Directory with labels.csv and clean images/.")
    parser.add_argument("--adv_dir", default="results/vit_b_agentic_adv", help="Directory with adversarial images.")
    parser.add_argument("--output_dir", default="results/vit_b_eval", help="Where CSV metrics are written.")
    parser.add_argument("--attack_name", default="Ours")
    parser.add_argument("--source_model", default="vit_base_patch16_224")
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--max_images", default=None, type=int)
    return parser.parse_args()


def run(cmd):
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        completed.check_returncode()


def format_paper_table(frame):
    table = frame.copy()
    for column in ["Res-50", "VGG-19", "ViT-B", "Swin-B", "Avg."]:
        table[column] = pd.to_numeric(table[column], errors="coerce").round(1)
    table["SSIM"] = pd.to_numeric(table["SSIM"], errors="coerce").round(4)
    table["L2"] = pd.to_numeric(table["L2"], errors="coerce").round(2)
    table["PSNR"] = pd.to_numeric(table["PSNR"], errors="coerce").round(2)
    return table


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_csv = output_dir / "attack_success.csv"
    quality_csv = output_dir / "quality_metrics.csv"
    summary_csv = output_dir / "summary.csv"

    base_args = [
        "--input_dir",
        args.input_dir,
        "--attack_name",
        args.attack_name,
        "--batchsize",
        str(args.batchsize),
        "--num_workers",
        str(args.num_workers),
        "--GPU_ID",
        args.GPU_ID,
    ]
    if args.max_images is not None:
        base_args.extend(["--max_images", str(args.max_images)])

    run(
        [
            sys.executable,
            "evaluate_attack.py",
            *base_args,
            "--adv_dir",
            args.adv_dir,
            "--source_model",
            args.source_model,
            "--output_csv",
            str(attack_csv),
            "--quiet",
        ]
    )

    run(
        [
            sys.executable,
            "evaluate_quality.py",
            *base_args,
            "--adv_dir",
            args.adv_dir,
            "--output_csv",
            str(quality_csv),
            "--skip_lpips",
            "--quiet",
        ]
    )

    frames = []
    if attack_csv.is_file():
        frames.append(pd.read_csv(attack_csv))
    if quality_csv.is_file():
        frames.append(pd.read_csv(quality_csv))
    if frames:
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="Attacks", how="outer")
        display_columns = [
            "Attacks",
            "Surrogate",
            "Res-50",
            "VGG-19",
            "ViT-B",
            "Swin-B",
            "Avg.",
            "SSIM",
            "L2",
            "PSNR",
        ]
        for column in display_columns:
            if column not in merged.columns:
                merged[column] = ""
        merged = format_paper_table(merged[display_columns])
        merged.to_csv(summary_csv, index=False)
        print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
