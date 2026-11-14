#!/usr/bin/env python3
"""GeoSearcher-4B: single-image inference with anchor and target visualization.

This script preserves the original CLI: --model, --image, --output.
Red dashed rectangles are intermediate reference anchors (<ref_bbox>).
Green solid rectangle is the final target (bbox_2d).
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

DEFAULT_QUERY = (
    'Find the specific object described as '
    '"The storage tank is on the left of the storage tank on the far right" '
    'in this aerial photo.'
)

# Match integer or decimal coordinates normalized to the 0–1000 range.
_NUMBER = r"-?\d+(?:\.\d+)?"
_FOUR_NUMBERS = (
    rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*"
    rf"({_NUMBER})\s*,\s*({_NUMBER})\s*\]"
)
_REF_PATTERN = re.compile(
    rf"<ref_bbox>\s*{_FOUR_NUMBERS}\s*</ref_bbox>", re.IGNORECASE | re.DOTALL
)
_TARGET_PATTERN = re.compile(
    rf"[\"']?bbox_2d[\"']?\s*:\s*{_FOUR_NUMBERS}", re.IGNORECASE | re.DOTALL
)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_reference_boxes(text: str) -> List[List[float]]:
    """Extract all explicit reference anchors from the generated reasoning."""
    return [[float(x) for x in m.groups()] for m in _REF_PATTERN.finditer(text)]


def extract_target_box(text: str) -> Optional[List[float]]:
    """Prefer the bbox_2d inside the final <answer> section."""
    answers = _ANSWER_PATTERN.findall(text)
    for section in reversed(answers):
        matches = _TARGET_PATTERN.findall(section)
        if matches:
            return [float(x) for x in matches[-1]]
    matches = _TARGET_PATTERN.findall(text)
    return [float(x) for x in matches[-1]] if matches else None


def denormalize_box(box: List[float], width: int, height: int) -> Optional[List[int]]:
    """Convert a valid normalized xyxy box to image pixel coordinates."""
    if not all(0 <= v <= 1000 for v in box) or box[0] >= box[2] or box[1] >= box[3]:
        return None
    x1, y1, x2, y2 = box
    return [
        round(x1 * width / 1000),
        round(y1 * height / 1000),
        round(x2 * width / 1000),
        round(y2 * height / 1000),
    ]


def draw_dashed_line(draw: ImageDraw.ImageDraw, start, end, color, width=3, dash=12, gap=7):
    """Draw one axis-aligned dashed line using Pillow."""
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:  # Vertical side
        direction = 1 if y2 >= y1 else -1
        length = abs(y2 - y1)
        for offset in range(0, length + 1, dash + gap):
            a = min(offset, length)
            b = min(offset + dash, length)
            draw.line([(x1, y1 + direction * a), (x1, y1 + direction * b)], fill=color, width=width)
    else:  # Horizontal side
        direction = 1 if x2 >= x1 else -1
        length = abs(x2 - x1)
        for offset in range(0, length + 1, dash + gap):
            a = min(offset, length)
            b = min(offset + dash, length)
            draw.line([(x1 + direction * a, y1), (x1 + direction * b, y1)], fill=color, width=width)


def draw_dashed_box(draw: ImageDraw.ImageDraw, box: List[int], color, width=3):
    """Draw a true dashed rectangle, not a solid rectangle."""
    x1, y1, x2, y2 = box
    dash = max(8, width * 4)
    gap = max(5, width * 2)
    draw_dashed_line(draw, (x1, y1), (x2, y1), color, width, dash, gap)
    draw_dashed_line(draw, (x2, y1), (x2, y2), color, width, dash, gap)
    draw_dashed_line(draw, (x2, y2), (x1, y2), color, width, dash, gap)
    draw_dashed_line(draw, (x1, y2), (x1, y1), color, width, dash, gap)


def get_font(image_height: int):
    # Prefer a readable TrueType font and fall back to the Pillow default.
    font_size = max(14, round(image_height / 45))
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, font_size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, box: List[int], text: str, color, font):
    # Place the label above the box, or inside if there is not enough space.
    x1, y1, _, _ = box
    rect = draw.textbbox((0, 0), text, font=font)
    text_w = rect[2] - rect[0]
    text_h = rect[3] - rect[1]
    pad = 4
    label_y = y1 - text_h - 2 * pad - 3
    if label_y < 0:
        label_y = y1 + 3
    draw.rectangle(
        (x1, label_y, x1 + text_w + 2 * pad, label_y + text_h + 2 * pad),
        fill=color,
    )
    draw.text((x1 + pad, label_y + pad - rect[1]), text, fill="white", font=font)


def save_visualization(image: Image.Image, references, target, output: Path) -> bool:
    """Draw all valid red dashed anchors and the final green target."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    width, height = vis.size
    line_width = max(3, round(min(width, height) / 240))
    font = get_font(height)
    count = 0

    for idx, ref in enumerate(references, start=1):
        pixel_box = denormalize_box(ref, width, height)
        if pixel_box is None:
            print(f"Warning: invalid reference bbox #{idx}: {ref}; skipping.")
            continue
        draw_dashed_box(draw, pixel_box, color="#E53935", width=line_width)
        draw_label(draw, pixel_box, f"Reference {idx}", "#C62828", font)
        count += 1

    if target is not None:
        pixel_box = denormalize_box(target, width, height)
        if pixel_box is None:
            print(f"Warning: invalid final bbox_2d: {target}; skipping.")
        else:
            draw.rectangle(pixel_box, outline="#24D04B", width=line_width)
            draw_label(draw, pixel_box, "Target", "#138D35", font)
            count += 1

    if count == 0:
        print("No valid boxes to visualize; image not saved.")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output)
    print(f"Visualization saved to: {output}")
    return True


def main():
    # Import model dependencies at runtime to keep box utilities standalone.
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    parser = argparse.ArgumentParser(description="GeoSearcher single-image inference and visualization")
    # Support the original CLI flags and the extended _path aliases.
    parser.add_argument("--model", "--model_path", dest="model", default="wangdianyu/GeoSearcher-4B")
    parser.add_argument("--image", "--image_path", dest="image", default="example/00014.jpg")
    parser.add_argument("--output", "--output_path", dest="output", default="outputs/00014_prediction.jpg")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=1536)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    # Use the Qwen3-VL model class; Qwen2.5-VL has incompatible weights.
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map="auto", trust_remote_code=True
    ).eval()

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.query},
        ]}
    ]

    # Let the processor build the multimodal prompt and image tokens.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # Decode only newly generated tokens and retain reasoning/anchor markers.
    continuation = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        continuation,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    print("\n=== GeoSearcher output ===\n")
    print(response)

    references = extract_reference_boxes(response)
    target = extract_target_box(response)
    print(f"\nDetected reference anchors: {len(references)}")
    for idx, ref in enumerate(references, start=1):
        print(f"Reference bbox #{idx} (0-1000): {ref}")
    if target is not None:
        print("Predicted normalized bbox_2d (0-1000):", target)
    else:
        print("Warning: bbox_2d not found in model output.")

    save_visualization(image, references, target, Path(args.output))


if __name__ == "__main__":
    main()
