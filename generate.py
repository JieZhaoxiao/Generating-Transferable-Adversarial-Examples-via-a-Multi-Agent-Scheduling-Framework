import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial examples with the Qwen3.6-Plus agent.")
    parser.add_argument("--input_dir", default="data")
    parser.add_argument("--output_dir", default="results/generated_vit_b_adv")
    parser.add_argument("--source_model", default="vit_base_patch16_224")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--eps", default=16 / 255, type=float)
    parser.add_argument("--batchsize", default=16, type=int)
    parser.add_argument("--GPU_ID", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_images", default=None, type=int)
    parser.add_argument("--attack_config", default="configs/attack_vit_b.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set. Set it before running generation.")

    cmd = [
        sys.executable,
        "main.py",
        "--attack",
        "agentic_schedule",
        "--model",
        args.source_model,
        "--input_dir",
        args.input_dir,
        "--output_dir",
        args.output_dir,
        "--attack_config",
        args.attack_config,
        "--epoch",
        str(args.epoch),
        "--eps",
        str(args.eps),
        "--batchsize",
        str(args.batchsize),
        "--GPU_ID",
        args.GPU_ID,
    ]
    if args.max_images is not None:
        cmd.extend(["--max_images", str(args.max_images)])
    print("=> " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
