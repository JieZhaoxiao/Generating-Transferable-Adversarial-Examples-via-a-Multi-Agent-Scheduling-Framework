import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torchvision.models as models
import timm

from transferattack.utils import AdvDataset, _local_timm_weight_path, wrap_model


EVAL_MODELS = [
    ("Res-50", "resnet50", "cnn"),
    ("VGG-19", "vgg19", "cnn"),
    ("ViT-B", "vit_base_patch16_224", "vit"),
    ("Swin-B", "swin_tiny_patch4_window7_224", "vit"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate transfer attack success rates.")
    parser.add_argument("--input_dir", default="./data", type=str)
    parser.add_argument("--adv_dir", required=True, type=str)
    parser.add_argument("--attack_name", default="Attack", type=str)
    parser.add_argument("--output_csv", default="results/attack_eval.csv", type=str)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--GPU_ID", default="0", type=str)
    parser.add_argument("--max_images", default=None, type=int)
    parser.add_argument("--source_model", default="resnet50", type=str)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser.parse_args()


def load_eval_model(model_id, kind):
    if kind == "cnn":
        try:
            model = models.__dict__[model_id](weights="DEFAULT")
            return model, "pretrained"
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load pretrained weights for {model_id}. "
                "Check the network connection or the local torchvision cache."
            ) from exc

    local_weight_path = _local_timm_weight_path(model_id)
    if local_weight_path is not None:
        model = timm.create_model(
            model_id,
            pretrained=True,
            pretrained_cfg_overlay={"file": local_weight_path},
        )
        return model, "pretrained_local"

    try:
        model = timm.create_model(model_id, pretrained=True)
        return model, "pretrained_download_or_cache"
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load pretrained weights for {model_id}. "
            f"Check the network connection or place weights under pretrained/timm/{model_id}/model.pth."
        ) from exc


def eval_model(model, dataloader):
    correct, total = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            pred = logits.argmax(dim=1).detach().cpu()
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return (1.0 - correct / max(1, total)) * 100.0


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.adv_dir, targeted=False, eval=True)
    if args.max_images is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(len(dataset), args.max_images))))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = {"Attacks": args.attack_name, "Surrogate": args.source_model}
    black_box_asrs = []
    expected_black_box = sum(1 for _, model_id, _ in EVAL_MODELS if model_id != args.source_model)
    name_to_title = {model_id: title for title, model_id, _ in EVAL_MODELS}
    ordered_titles = [name_to_title[model_id] for _, model_id, _ in EVAL_MODELS]
    status_rows = []
    for _, model_id, kind in EVAL_MODELS:
        title = name_to_title[model_id]
        if not args.quiet:
            print(f"=> Evaluating {title} ({model_id})")
        model, weight_status = load_eval_model(model_id, kind)
        status_rows.append({"Model": title, "Model ID": model_id, "Weight status": weight_status})
        if model is None:
            rows[title] = ""
            continue
        model = wrap_model(model.eval().to(device))
        for param in model.parameters():
            param.requires_grad = False
        asr = eval_model(model, loader)
        rows[title] = round(asr, 1)
        if model_id != args.source_model:
            black_box_asrs.append(asr)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows["Avg."] = round(sum(black_box_asrs) / len(black_box_asrs), 1) if len(black_box_asrs) == expected_black_box else ""
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rows], columns=["Attacks", "Surrogate"] + ordered_titles + ["Avg."]).to_csv(output_csv, index=False)
    status_csv = output_csv.with_name("model_status.csv")
    pd.DataFrame(status_rows).to_csv(status_csv, index=False)
    if not args.quiet:
        print(f"=> Wrote {output_csv}")
        print(f"=> Wrote {status_csv}")


if __name__ == "__main__":
    main()
