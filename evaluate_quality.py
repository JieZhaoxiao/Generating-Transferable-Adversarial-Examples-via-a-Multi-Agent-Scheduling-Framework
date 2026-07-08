import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: scikit-image. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


QUALITY_COLUMNS = ["Attacks", "SSIM", "LPIPS", "L1", "L2", "Linf", "PSNR"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image quality for adversarial images.")
    parser.add_argument("--input_dir", default="./data", type=str, help="Directory with labels.csv and clean images/.")
    parser.add_argument("--adv_dir", required=True, type=str, help="Directory containing adversarial images.")
    parser.add_argument("--attack_name", default="Ours", type=str)
    parser.add_argument("--output_csv", default="results/quality_metrics.csv", type=str)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--GPU_ID", default="0", type=str)
    parser.add_argument("--max_images", default=None, type=int)
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS when the package/weights are unavailable.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser.parse_args()


def load_filenames(input_dir):
    frame = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(row.filename) for row in frame.itertuples(index=False)]


class QualityPairDataset(Dataset):
    def __init__(self, filenames, adv_dir, original_dir, image_size):
        self.filenames = filenames
        self.adv_dir = Path(adv_dir)
        self.original_dir = Path(original_dir)
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        adv = Image.open(self.adv_dir / filename).convert("RGB")
        original = Image.open(self.original_dir / filename).convert("RGB")
        size = (self.image_size, self.image_size)
        adv = adv.resize(size)
        original = original.resize(size)
        adv_np = np.asarray(adv)
        original_np = np.asarray(original)
        return self.to_tensor(adv), self.to_tensor(original), adv_np, original_np


def collate_quality(batch):
    adv_tensors, original_tensors, adv_arrays, original_arrays = zip(*batch)
    return torch.stack(adv_tensors), torch.stack(original_tensors), list(adv_arrays), list(original_arrays)


def load_lpips(device, skip_lpips):
    if skip_lpips:
        return None
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print(f"=> LPIPS unavailable, writing NaN for LPIPS: {exc}")
        return None


def compute_quality(filenames, adv_dir, original_dir, args, device):
    dataset = QualityPairDataset(filenames, adv_dir, original_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_quality,
    )
    lpips_model = load_lpips(device, args.skip_lpips)

    psnr_values = []
    ssim_values = []
    lpips_values = []
    l1_values = []
    l2_values = []
    linf_values = []
    with torch.no_grad():
        for adv, original, adv_arrays, original_arrays in loader:
            diff = (adv - original).flatten(1)
            mse = diff.pow(2).mean(dim=1).clamp_min(1e-12)
            psnr_values.extend((10.0 * torch.log10(1.0 / mse)).cpu().tolist())
            l1_values.extend(diff.abs().sum(dim=1).cpu().tolist())
            l2_values.extend(diff.norm(p=2, dim=1).cpu().tolist())
            linf_values.extend(diff.abs().amax(dim=1).cpu().tolist())

            for adv_np, original_np in zip(adv_arrays, original_arrays):
                ssim_values.append(ssim(adv_np, original_np, data_range=255.0, channel_axis=-1))

            if lpips_model is not None:
                adv = adv.to(device, non_blocking=True)
                original = original.to(device, non_blocking=True)
                lpips_batch = lpips_model(adv, original, normalize=True)
                lpips_values.extend(lpips_batch.detach().cpu().view(-1).tolist())

    return {
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)) if lpips_values else float("nan"),
        "L1": float(np.mean(l1_values)),
        "L2": float(np.mean(l2_values)),
        "Linf": float(np.mean(linf_values)),
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = load_filenames(args.input_dir)
    if args.max_images is not None:
        filenames = filenames[: args.max_images]
    original_dir = Path(args.input_dir) / "images"
    adv_dir = Path(args.adv_dir)
    missing = [name for name in filenames if not (adv_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{adv_dir} is missing {len(missing)} files; first missing: {missing[0]}")

    metrics = compute_quality(filenames, adv_dir, original_dir, args, device)
    row = {"Attacks": args.attack_name, **metrics}
    frame = pd.DataFrame([row], columns=QUALITY_COLUMNS)
    for column in ["SSIM", "LPIPS", "L1", "L2", "PSNR"]:
        frame[column] = frame[column].round(4)
    frame["Linf"] = frame["Linf"].round(6)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    if not args.quiet:
        print(
            f"=> {args.attack_name}: PSNR={metrics['PSNR']:.4f}, SSIM={metrics['SSIM']:.4f}, "
            f"LPIPS={metrics['LPIPS']:.4f}, L1={metrics['L1']:.4f}, L2={metrics['L2']:.4f}, "
            f"Linf={metrics['Linf']:.6f}"
        )
        print(f"=> Wrote {output_csv}")


if __name__ == "__main__":
    main()
