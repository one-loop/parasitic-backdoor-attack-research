#!/usr/bin/env python3
"""
Simplified parasitic backdoor attack pipeline (notebook phases 1-2 + export).

Trains a CIFAR ResNet18, runs TracIn host selection, co-adapts the trigger,
and writes BackdoorBench-compatible artifacts to bb_export/parasitic_attack/.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run parasitic attack and export for BackdoorBench")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-class", type=int, default=3)
    parser.add_argument("--k-hosts", type=int, default=1000)
    parser.add_argument("--poison-budget", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=16 / 255)
    parser.add_argument("--alpha-trigger", type=float, default=4 / 255)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--epochs-base", type=int, default=100)
    parser.add_argument("--epochs-coadapt", type=int, default=10)
    parser.add_argument("--trigger-steps", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="./BackdoorBench/data")
    parser.add_argument(
        "--bb-root",
        type=str,
        default="./BackdoorBench",
        help="BackdoorBench root (for export module)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./bb_export/parasitic_attack",
        help="Repo-root export directory used by run_defenses.sh",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_resnet18(device: torch.device) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.linear = model.fc
    return model.to(device)


def evaluate_metrics(model, dataloader, device, trigger=None, poison_target=None):
    model.eval()
    cda_correct, cda_total = 0, 0
    class_correct = [0] * 10
    class_total = [0] * 10
    asr_correct, asr_total = 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            cda_total += targets.size(0)
            cda_correct += preds.eq(targets).sum().item()

            for label, pred in zip(targets.tolist(), preds.tolist()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

            if trigger is not None and poison_target is not None:
                non_target_mask = targets != poison_target
                if non_target_mask.any():
                    p_inputs = inputs[non_target_mask] + trigger
                    p_preds = model(p_inputs).argmax(dim=1)
                    asr_total += p_inputs.size(0)
                    asr_correct += (p_preds == poison_target).sum().item()

    metrics = {"cda_overall": 100.0 * cda_correct / cda_total}
    metrics["class_cda"] = {
        i: 100.0 * class_correct[i] / class_total[i] if class_total[i] else 0.0
        for i in range(10)
    }
    if trigger is not None:
        metrics["asr"] = 100.0 * asr_correct / asr_total if asr_total else 0.0
    return metrics


class ParasiticDataset(Dataset):
    def __init__(self, base_dataset, hosts, poisons, target_class):
        self.base = base_dataset
        self.host_set = set(hosts)
        self.poison_set = set(poisons)
        self.target_class = target_class

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        flag = 0
        if idx in self.host_set:
            flag = 1
        elif idx in self.poison_set:
            flag = 2
            label = self.target_class
        return img, label, flag


def train_base_model(trainset, args, device):
    logging.info("Phase 1: training base model")
    model = get_resnet18(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    checkpoints = []
    model.train()
    for epoch in range(args.epochs_base):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            checkpoints.append(copy.deepcopy(model.state_dict()))
            logging.info("Base epoch %d/%d complete", epoch + 1, args.epochs_base)
    return checkpoints


def select_hosts(trainset, checkpoints, target_class, k_hosts, device):
    logging.info("Phase 1: TracIn host selection")
    criterion = nn.CrossEntropyLoss()
    target_indices = [i for i, label in enumerate(trainset.targets) if label == target_class]
    eval_loader = DataLoader(
        Subset(trainset, target_indices),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    influence_scores = {idx: 0.0 for idx in target_indices}

    for state_dict in checkpoints:
        temp_model = get_resnet18(device)
        temp_model.load_state_dict(state_dict)
        temp_model.eval()
        for idx, (inputs, targets) in zip(target_indices, eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            temp_model.zero_grad()
            loss = criterion(temp_model(inputs), targets)
            loss.backward()
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in temp_model.fc.parameters()
                if p.grad is not None
            )
            influence_scores[idx] += grad_norm

    host_indices = [
        x[0]
        for x in sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:k_hosts]
    ]
    logging.info("Selected %d hosts for class %d", len(host_indices), target_class)
    return host_indices


def co_adapt(trainset, checkpoints, host_indices, args, device):
    logging.info("Phase 2: trigger co-adaptation")
    non_target_indices = [i for i, label in enumerate(trainset.targets) if label != args.target_class]
    poison_indices = np.random.choice(non_target_indices, args.poison_budget, replace=False)

    unified_loader = DataLoader(
        ParasiticDataset(trainset, host_indices, poison_indices, args.target_class),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    delta = torch.randn((1, 3, 32, 32), device=device) * 1e-3
    delta.requires_grad = True

    model = get_resnet18(device)
    model.load_state_dict(checkpoints[-1])
    model.eval()
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epochs_coadapt):
        running_loss = 0.0
        penalty_accum = 0.0
        batches_with_penalty = 0

        for inputs, targets, flags in unified_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask_h = flags == 1
            mask_p = flags == 2

            if mask_h.any() and mask_p.any():
                loss_h = criterion(model(inputs[mask_h]), targets[mask_h])
                g_h_tuple = torch.autograd.grad(loss_h, model.fc.parameters())
                g_h = torch.cat([g.flatten() for g in g_h_tuple]).detach()

                for _ in range(args.trigger_steps):
                    x_p = inputs[mask_p] + delta
                    loss_p = criterion(model(x_p), targets[mask_p])
                    g_p_tuple = torch.autograd.grad(loss_p, model.fc.parameters(), create_graph=True)
                    g_p = torch.cat([g.flatten() for g in g_p_tuple])
                    penalty = F.mse_loss(g_p, -g_h)
                    penalty.backward()
                    with torch.no_grad():
                        delta -= args.alpha_trigger * delta.grad.sign()
                        delta.clamp_(-args.epsilon, args.epsilon)
                    delta.grad.zero_()

                penalty_accum += penalty.item()
                batches_with_penalty += 1

            optimizer.zero_grad()
            x_train = inputs.clone()
            if mask_p.any():
                x_train[mask_p] = x_train[mask_p] + delta.detach()
            loss = criterion(model(x_train), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_penalty = penalty_accum / batches_with_penalty if batches_with_penalty else 0.0
        logging.info(
            "Co-adapt round %d/%d | penalty=%.6f | loss=%.4f",
            epoch + 1,
            args.epochs_coadapt,
            avg_penalty,
            running_loss / len(unified_loader),
        )

    return model, delta, poison_indices


def export_for_backdoorbench(model, delta, poison_indices, args):
    bb_root = os.path.abspath(args.bb_root)
    export_dir = os.path.abspath(args.export_dir)
    if not os.path.isfile(os.path.join(bb_root, "train_and_export_parasitic.py")):
        raise FileNotFoundError(f"BackdoorBench not found at {bb_root}")

    sys.path.insert(0, bb_root)
    os.chdir(bb_root)
    from train_and_export_parasitic import export_attack_result

    record_path = "record/bb_export/parasitic_attack"
    export_attack_result(
        model,
        delta,
        poison_indices,
        save_path=record_path,
        target_class=args.target_class,
    )

    os.makedirs(export_dir, exist_ok=True)
    src = os.path.join(bb_root, record_path, "attack_result.pt")
    dst = os.path.join(export_dir, "attack_result.pt")
    if os.path.abspath(src) != os.path.abspath(dst):
        import shutil
        shutil.copy2(src, dst)

    logging.info("Exported attack_result.pt to %s", dst)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logging.info("Using device: %s", device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    data_root = os.path.join(args.data_dir, "cifar10")
    trainset = CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    testset = CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    checkpoints = train_base_model(trainset, args, device)
    host_indices = select_hosts(trainset, checkpoints, args.target_class, args.k_hosts, device)
    model, delta, poison_indices = co_adapt(trainset, checkpoints, host_indices, args, device)

    metrics = evaluate_metrics(
        model, testloader, device, trigger=delta.detach(), poison_target=args.target_class
    )
    logging.info("Pre-unlearning CDA: %.2f%%", metrics["cda_overall"])
    logging.info("Pre-unlearning ASR: %.2f%%", metrics["asr"])
    logging.info(
        "Target class %d (%s) CDA: %.2f%%",
        args.target_class,
        CIFAR_CLASSES[args.target_class],
        metrics["class_cda"][args.target_class],
    )

    export_for_backdoorbench(model, delta, poison_indices, args)
    logging.info("Done. Run BackdoorBench defenses with: sbatch BackdoorBench/run_defenses.sh")


if __name__ == "__main__":
    main()
