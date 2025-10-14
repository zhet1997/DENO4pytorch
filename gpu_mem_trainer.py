#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple infinite training script that reserves ~target GiB on chosen GPUs "
            "with safety margin to avoid OOM."
        )
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="4",
        help=(
            "Comma-separated CUDA device IDs to use for memory reservation (default: 4,6). "
            "Training runs on the first available GPU from this list."
        ),
    )
    parser.add_argument(
        "--target-gib",
        type=float,
        default=18.0,
        help="Approximate GiB to reserve per selected GPU (default: 18.0).",
    )
    parser.add_argument(
        "--safety-gib",
        type=float,
        default=2.0,
        help=(
            "Extra safety margin to leave free on each GPU to reduce OOM risk (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--min-chunk-mib",
        type=int,
        default=64,
        help="Minimum allocation chunk size in MiB when probing available memory (default: 64).",
    )
    parser.add_argument(
        "--max-chunk-mib",
        type=int,
        default=512,
        help="Initial allocation chunk size in MiB when reserving memory (default: 512).",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.1,
        help="Sleep seconds between training steps to reduce heat/noise (default: 0.1).",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=50,
        help="Steps between status prints (default: 50).",
    )
    return parser.parse_args()


def get_device_list(gpus_arg: str) -> List[int]:
    try:
        device_list = [int(x.strip()) for x in gpus_arg.split(",") if x.strip() != ""]
    except ValueError:
        print(f"[Error] Invalid --gpus value: {gpus_arg}", file=sys.stderr)
        sys.exit(1)
    return device_list


def format_gib(bytes_value: int) -> str:
    return f"{bytes_value / GIB:.2f} GiB"


def reserve_memory_on_device(
    device_id: int,
    target_gib: float,
    safety_gib: float,
    max_chunk_mib: int,
    min_chunk_mib: int,
) -> Tuple[List[torch.Tensor], int]:
    """
    Best-effort memory reservation on a specific CUDA device, returning the held tensors
    and the number of bytes successfully reserved. Allocation is done in decreasing chunk
    sizes to avoid OOM.
    """
    if not torch.cuda.is_available():
        print("[Warn] CUDA not available. Skipping reservation.")
        return [], 0

    if device_id < 0 or device_id >= torch.cuda.device_count():
        print(f"[Warn] Device cuda:{device_id} not present. Skipping reservation.")
        return [], 0

    reservation_tensors: List[torch.Tensor] = []
    with torch.cuda.device(device_id):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device_id)
        desired_bytes = int(target_gib * GIB)
        margin_bytes = int(safety_gib * GIB)
        # Leave margin beyond the target when free memory is tight.
        available_bytes = max(free_bytes - margin_bytes, 0)
        to_reserve = min(desired_bytes, available_bytes)

        if to_reserve <= 0:
            print(
                f"[Info][cuda:{device_id}] Not reserving (target={target_gib}GiB, "
                f"free={format_gib(free_bytes)}, margin={safety_gib}GiB)"
            )
            return [], 0

        chunk_bytes = max(max_chunk_mib * 1024 * 1024, 1)
        min_chunk_bytes = max(min_chunk_mib * 1024 * 1024, 1)

        reserved_bytes = 0
        print(
            f"[Info][cuda:{device_id}] total={format_gib(total_bytes)}, free={format_gib(free_bytes)}, "
            f"attempting to reserve ~{target_gib:.2f} GiB (<= {format_gib(to_reserve)}), margin={safety_gib}GiB"
        )

        # Use uint8 for one-byte elements to control allocation sizes precisely.
        dtype = torch.uint8
        while reserved_bytes < to_reserve and chunk_bytes >= min_chunk_bytes:
            remaining = to_reserve - reserved_bytes
            this_alloc = min(chunk_bytes, remaining)
            try:
                numel = max(this_alloc // torch.tensor([], dtype=dtype).element_size(), 1)
                tensor = torch.empty(numel, dtype=dtype, device=f"cuda:{device_id}")
                reservation_tensors.append(tensor)
                reserved_bytes += tensor.numel() * tensor.element_size()
            except RuntimeError as e:
                # Reduce chunk size on OOM or allocator failure
                chunk_bytes = chunk_bytes // 2
                continue

        print(
            f"[Info][cuda:{device_id}] reserved ~{format_gib(reserved_bytes)}; "
            f"free now ~{format_gib(torch.cuda.mem_get_info(device=device_id)[0])}"
        )

    return reservation_tensors, reserved_bytes


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.head(x)


def build_tiny_model_and_optimizer(device: torch.device) -> Tuple[nn.Module, optim.Optimizer]:
    model = TinyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def infinite_training_loop(
    device: torch.device,
    sleep_sec: float,
    print_interval: int,
) -> None:
    torch.backends.cudnn.benchmark = True
    model, optimizer = build_tiny_model_and_optimizer(device)
    criterion = nn.CrossEntropyLoss()

    step = 0
    while True:
        inputs = torch.randn(32, 3, 64, 64, device=device)
        targets = torch.randint(0, 10, (32,), device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        step += 1
        if step % print_interval == 0:
            alloc = torch.cuda.memory_allocated(device=device.index)
            reserved = torch.cuda.memory_reserved(device=device.index)
            free, total = torch.cuda.mem_get_info(device=device.index)
            print(
                f"[Train][{time.strftime('%H:%M:%S')}] step={step} loss={loss.item():.4f} "
                f"alloc={format_gib(alloc)}, reserved={format_gib(reserved)}, free={format_gib(free)} / total={format_gib(total)}"
            )

        if sleep_sec > 0:
            time.sleep(sleep_sec)


def main() -> None:
    args = parse_args()
    devices = get_device_list(args.gpus)

    if not torch.cuda.is_available():
        print("[Error] CUDA is not available. Exiting.", file=sys.stderr)
        sys.exit(1)

    if len(devices) == 0:
        print("[Error] No GPUs specified via --gpus.", file=sys.stderr)
        sys.exit(1)

    print(
        f"[Config] gpus={devices}, target_gib={args.target_gib}, safety_gib={args.safety_gib}, "
        f"chunk_mib=[{args.min_chunk_mib}, {args.max_chunk_mib}], sleep_sec={args.sleep_sec}"
    )

    # Reserve memory on all specified devices (best-effort)
    hold: Dict[int, List[torch.Tensor]] = {}
    for dev_id in devices:
        tensors, reserved = reserve_memory_on_device(
            device_id=dev_id,
            target_gib=args.target_gib,
            safety_gib=args.safety_gib,
            max_chunk_mib=args.max_chunk_mib,
            min_chunk_mib=args.min_chunk_mib,
        )
        if reserved > 0:
            hold[dev_id] = tensors

    # Pick first available device for training
    train_device: torch.device = None  # type: ignore
    for dev_id in devices:
        if 0 <= dev_id < torch.cuda.device_count():
            train_device = torch.device(f"cuda:{dev_id}")
            break
    if train_device is None:
        print("[Error] None of the requested GPUs exist on this machine.", file=sys.stderr)
        sys.exit(1)

    print(f"[Info] Starting infinite training on {train_device}...")
    try:
        infinite_training_loop(
            device=train_device,
            sleep_sec=args.sleep_sec,
            print_interval=args.print_interval,
        )
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user. Releasing memory and exiting.")
        # Release references explicitly
        hold.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


