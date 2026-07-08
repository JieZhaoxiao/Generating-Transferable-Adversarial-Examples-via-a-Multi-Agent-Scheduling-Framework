import argparse
import json
import os
from pathlib import Path

import torch
import tqdm
import transferattack
from transferattack.utils import *


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='mifgsm', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=32, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet50', type=str, help='the source surrogate model')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--attack_config', default=None, type=str, help='optional JSON file with attack kwargs')
    parser.add_argument('--max_images', default=None, type=int, help='maximum number of images to process')
    return parser.parse_args()


def load_attack_config(config_path):
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f'attack_config not found: {config_path}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if torch.cuda.is_available():
        visible_devices = torch.cuda.device_count()
        print(f'=> Using {visible_devices} visible CUDA device(s): {args.GPU_ID}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    if args.max_images is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(len(dataset), args.max_images))))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    if not args.eval:
        if args.ensemble or len(args.model.split(',')) > 1:
            args.model = args.model.split(',')
        attack_kwargs = load_attack_config(args.attack_config)
        attacker = transferattack.load_attack_class(args.attack)(
            model_name=args.model,
            epsilon=args.eps,
            epoch=args.epoch,
            decay=args.momentum,
            random_start=args.random_start,
            targeted=args.targeted,
            **attack_kwargs,
        )

        for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
            if args.attack in ['ttp', 'm3d']: 
                for idx, target_class in enumerate(generation_target_classes):
                    perturbations = attacker(images, labels, idx)
                    new_output_dir = os.path.join(args.output_dir, str(target_class))
                    if not os.path.exists(new_output_dir):
                        os.makedirs(new_output_dir)
                    save_images(new_output_dir, images + perturbations.cpu(), filenames)
            else:
                perturbations = attacker(images, labels)
                save_images(args.output_dir, images + perturbations.cpu(), filenames)
    else:
        res = '|'
        for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
            model = wrap_model(model.eval().cuda())
            for p in model.parameters():
                p.requires_grad = False
                
            if args.attack in ['ttp', 'm3d']: 
                asr = 0
                for idx, target_class in enumerate(generation_target_classes):
                    new_output_dir = os.path.join(args.output_dir, str(target_class))
                    new_dataset = AdvDataset(input_dir=args.input_dir, output_dir=new_output_dir, targeted=True, target_class=target_class, eval=args.eval)
                    new_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
                    asr += eval(model, new_dataloader, True)
                asr /= 10

            else:
                asr = eval(model, dataloader, args.targeted)
            print(f'{model_name}: {asr:.1f}')
            res += f' {asr:.1f} |'

        print(res)
        with open('results_eval.txt', 'a') as f:
            f.write(args.output_dir + res + '\n')
                
                
def eval(model, dataloader, is_targeted):
    correct, total = 0, 0
    for images, labels, _ in dataloader:
        if is_targeted:
            labels = labels[1]
        pred = model(images.cuda())
        correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
        total += labels.shape[0]
    if is_targeted:
        # correct: pred == target_label
        asr = (correct / total) * 100
    else:
        # correct: pred == original_label
        asr = (1 - correct / total) * 100
    return asr


if __name__ == '__main__':
    main()
