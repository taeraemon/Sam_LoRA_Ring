import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from segment_anything import build_sam_vit_b, SamPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="SAM inference with flexible box prompt")
    parser.add_argument('--image', type=str, required=True, help="Image file path")
    parser.add_argument('--checkpoint', type=str, default='sam_vit_b_01ec64.pth', help="SAM checkpoint path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--box_xyxy', nargs=4, type=int, metavar=('x1', 'y1', 'x2', 'y2'),
                        help="Box by x1 y1 x2 y2 (top-left and bottom-right)")
    group.add_argument('--box_cxcywh', nargs=4, type=int, metavar=('cx', 'cy', 'w', 'h'),
                        help="Box by center x/y and width/height (cx cy w h)")
    group.add_argument('--box_xywh', nargs=4, type=int, metavar=('x', 'y', 'w', 'h'),
                        help="Box by x/y (top-left) and width/height (x y w h)")
    return parser.parse_args()

def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    return [x1, y1, x2, y2]

def box_xywh_to_xyxy(box):
    x, y, w, h = box
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)
    return [x1, y1, x2, y2]

def main():
    args = parse_args()

    image_path = args.image
    checkpoint = args.checkpoint

    # 1. 이미지 로드
    image = np.array(Image.open(image_path).convert("RGB"))

    # 2. 박스 좌표 변환
    if args.box_xyxy:
        box = args.box_xyxy
        box_type = "xyxy"
    elif args.box_cxcywh:
        box = box_cxcywh_to_xyxy(args.box_cxcywh)
        box_type = "cxcywh"
    elif args.box_xywh:
        box = box_xywh_to_xyxy(args.box_xywh)
        box_type = "xywh"
    else:
        print("박스 프롬프트를 반드시 한 가지 방식으로 넣어야 해요!", file=sys.stderr)
        return

    # 3. SAM 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam_vit_b(checkpoint=checkpoint)
    predictor = SamPredictor(sam)
    sam.to(device)
    sam.eval()

    # 4. 이미지 셋업
    predictor.set_image(image)

    # 5. 박스 프롬프트로 추론
    input_box = np.array(box)[None, :]  # shape: [1, 4]
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=True
    )
    mask = masks[0]

    # 6. 시각화
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"Input Image ({box_type})")
    plt.imshow(image)
    x1, y1, x2, y2 = box
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2, linestyle='--')
    )
    plt.text(x1, y1 - 5, f"Box ({box_type})", color='red')

    plt.subplot(1, 3, 2)
    plt.title("Pred Mask")
    plt.imshow(mask, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(image)
    plt.imshow(mask, cmap="jet", alpha=0.5)
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2, linestyle='--')
    )
    plt.text(x1, y1 - 5, f"Box ({box_type})", color='red')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
