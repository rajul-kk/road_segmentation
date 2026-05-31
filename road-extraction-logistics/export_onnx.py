"""
ONNX Export Script

Exports a trained DeepLabV3 road segmentation model to ONNX format for fast
CPU inference without a full PyTorch installation.

Usage:
    python export_onnx.py
    python export_onnx.py --backbone resnet101 --image-size 512 --output models/road_seg.onnx
    python export_onnx.py --verify   # export then validate with onnxruntime

Requirements:
    pip install onnx onnxruntime
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

import config as cfg
from models.architecture import build_model


class _SegWrapper(nn.Module):
    """Thin wrapper so torch.onnx.export receives a plain tensor output."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']  # (B, 2, H, W)


def parse_args():
    p = argparse.ArgumentParser(description="Export trained model to ONNX")
    p.add_argument("--checkpoint",  default=cfg.FINAL_MODEL_PATH, help="Path to trained .pth checkpoint")
    p.add_argument("--backbone",    default=cfg.BACKBONE,         choices=["resnet50", "resnet101"])
    p.add_argument("--output",      default="models/road_seg.onnx", help="Output .onnx file path")
    p.add_argument("--image-size",  type=int, default=512,         help="Input image size (square)")
    p.add_argument("--opset",       type=int, default=17,          help="ONNX opset version")
    p.add_argument("--verify",      action="store_true",           help="Validate export with onnxruntime")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cpu")  # ONNX export always on CPU for portability
    print(f"Loading {args.backbone} from {args.checkpoint}...")

    base = build_model(backbone=args.backbone)
    ckpt = torch.load(args.checkpoint, map_location=device)
    base.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    base.eval()

    model = _SegWrapper(base)

    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Exporting to {args.output}  (opset {args.opset}, input {args.image_size}x{args.image_size})...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"✅ Exported: {args.output}")

    if args.verify:
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx.checker.check_model(args.output)
            print("✅ ONNX model check passed")

            sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
            inp  = dummy.numpy()
            out  = sess.run(None, {"image": inp})[0]
            print(f"✅ onnxruntime inference OK — output shape: {out.shape}")

            # Sanity-check: PyTorch vs ONNX outputs should match closely
            with torch.no_grad():
                pt_out = model(dummy).numpy()
            max_diff = float(np.abs(pt_out - out).max())
            print(f"   Max abs diff PyTorch vs ONNX: {max_diff:.6f}")
            if max_diff > 1e-3:
                print("⚠️  Difference is larger than expected — check opset compatibility")

        except ImportError as e:
            print(f"⚠️  Verification skipped — missing package: {e}")
            print("    Install with: pip install onnx onnxruntime")


if __name__ == "__main__":
    main()
