<div align="center">

# GeoSearcher

### Anchor-Guided Progressive Reasoning for Remote Sensing Visual Grounding with Process Supervision

**Accepted by _Information Fusion_ (2026)**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.01050-b31b1b.svg)](https://arxiv.org/abs/2607.01050)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-GeoSearcher--4B-yellow)](https://huggingface.co/wangdianyu/GeoSearcher-4B)
[![Journal](https://img.shields.io/badge/Information%20Fusion-Accepted-2f72b7.svg)](https://www.sciencedirect.com/journal/information-fusion)

</div>

## 🔍 Overview

<p align="center"><img src="GeoSearcher_framework.png" width="95%" alt="GeoSearcher framework"></p>

<!-- 中文：GeoSearcher 将遥感视觉定位重构为锚点引导的渐进式推理过程，而不是直接一步生成目标坐标。 -->
**GeoSearcher** is an anchor-guided progressive reasoning framework for **remote sensing visual grounding (RSVG)**. Instead of directly generating target coordinates in a single step, GeoSearcher uses explicit reference-object anchors to progressively integrate location, spatial-relation, and attribute clues before localizing the final target.

<!-- 中文：模型基于 Qwen3-VL-4B，使用两阶段后训练：锚点中心推理监督微调（ACR-SFT）及过程忠实的组相对策略优化（PF-GRPO）。 -->
Built upon **Qwen3-VL-4B**, GeoSearcher is trained in two stages:

1. **Anchor-Centric Reasoning Supervised Fine-Tuning (ACR-SFT):** constructs and learns from anchor-centric reasoning trajectories. Intermediate reference-object coordinates are explicitly represented using `<ref_bbox>...</ref_bbox>`.
2. **Process-Faithful Group Relative Policy Optimization (PF-GRPO):** improves grounding and intermediate reasoning behavior using a **Process-Aware Reward (PAR)** and a **Reasoning-Informative Sample Selector (RISS)**.

## 🤗 Model Weights

<!-- 中文：模型仓库已创建，公开权重及推理配置以 Hugging Face 页面为准。 -->
The official model repository is available on Hugging Face. Please check the repository for the latest checkpoint and processor/tokenizer files.

| Model | Backbone | Training | Download |
|:--|:--|:--|:--|
| **GeoSearcher-4B** | Qwen3-VL-4B | ACR-SFT + PF-GRPO | [🤗 Hugging Face](https://huggingface.co/wangdianyu/GeoSearcher-4B) |

## ⚙️ Environment Setup

<!-- 中文：单图推理推荐直接使用 Transformers，不需要安装 LLaMA-Factory 或 vLLM。需要 PyTorch、Transformers、Accelerate 和 Pillow，GPU 版本的 PyTorch 应根据设备和 CUDA 环境安装。 -->
### Option 1: Transformers (single-image inference)

For single-image inference, **Transformers is sufficient as the inference framework**; LLaMA-Factory and vLLM are not required. Install PyTorch for your CUDA environment, then install the Python dependencies:

```bash
pip install -U "transformers>=4.57.3" accelerate pillow
# Install a CUDA-compatible PyTorch build if torch is not installed.
```

<!-- 中文：多图批量推理推荐使用 LLaMA-Factory 的官方 vLLM 批量脚本，在独立环境安装，避免 vLLM 与现有 Transformers / PyTorch 版本冲突。 -->
### Option 2: LLaMA-Factory + vLLM (batch inference)

For faster batch inference, we recommend the official [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) vLLM inference script. Use a separate Python environment if necessary.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install -U vllm
```

> **Note:** Compatible versions of PyTorch, CUDA, Transformers, LLaMA-Factory, and vLLM depend on the environment. 

## 🖼️ Single-Image Inference (Transformers)

<!-- 中文：示例使用仓库中的 example/00014.jpg。只将用户问题作为模型输入，不能把标注答案传给模型。 -->
A sample aerial image is provided at [`example/00014.jpg`](example/00014.jpg).

**Query:**

> Find the specific object described as "The storage tank is on the left of the storage tank on the far right" in this aerial photo.

<!-- 中文：以下是样例的真实标注坐标，不是预先声称的模型预测结果。坐标范围为 0–1000。 -->
**Reference annotation (not a model prediction):** `{"bbox_2d": [555, 624, 706, 771]}` (normalized to 0–1000).

Run the provided single-image inference script from the GeoSearcher repository root:

```bash
python scripts/infer_single.py \
  --model wangdianyu/GeoSearcher-4B \
  --image example/00014.jpg \
  --output outputs/00014_prediction.jpg
```

<!-- 中文：脚本将打印模型原始生成内容、提取最终目标框 bbox_2d，并将 0–1000 坐标映射回原图像素，在输出图片上画框。 -->
The script prints the complete generated response, extracts the final `bbox_2d` field, and saves a visualization with the predicted box. To inspect the intermediate anchors, it preserves special tokens in the raw generated text.

<!-- 中文：PIL 图像会被 Processor 转换成视觉输入，脚本会自动插入图像 token。用户文本不能再额外写一个文字形式的 <image>，否则可能重复。 -->
**Important:** The image is passed as a structured multimodal input. Do not manually prepend the literal `<image>` token to the text in the Transformers example; the processor inserts the required image tokens automatically.

## 🚀 Batch Inference (LLaMA-Factory + vLLM)

<!-- 中文：LLaMA-Factory 官方批量入口是 scripts/vllm_infer.py。llamafactory-cli chat 是交互式推理入口，不是数据集批量预测命令。 -->
LLaMA-Factory provides the official [`scripts/vllm_infer.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/vllm_infer.py) for dataset-level batch generation.

### 1. Prepare the dataset

<!-- 中文：我们提供了 data/geosearcher_demo_source.json（OpenAI 风格 messages + images），和对应的 data/dataset_info.json 注册文件。 -->
The sample data is in `data/geosearcher_demo_source.json`, using the `messages` and `images` fields. The dataset is registered in `data/dataset_info.json`. For additional images, add entries using the same format.

<!-- 中文：执行下面的预处理，将图片相对路径改为绝对路径，避免从 LLaMA-Factory 目录执行命令时找不到 example/00014.jpg。 -->
From the **GeoSearcher repository root**, prepare absolute image paths:

```bash
python scripts/prepare_vllm_dataset.py \
  --input data/geosearcher_demo_source.json \
  --output data/geosearcher_demo.json \
  --image-root .
```

### 2. Run batch generation

<!-- 中文：切换到 LLaMA-Factory 目录后执行。请把 GEOSEARCHER_ROOT 改成你电脑中 GeoSearcher 仓库的绝对路径。 -->
From the **LLaMA-Factory repository root**, run:

```bash
# Replace this with the absolute path to your GeoSearcher repository.
# 中文：修改为 GeoSearcher 仓库的绝对路径。
export GEOSEARCHER_ROOT=/absolute/path/to/GeoSearcher

CUDA_VISIBLE_DEVICES=0 python scripts/vllm_infer.py \
  --model_name_or_path wangdianyu/GeoSearcher-4B \
  --template qwen3_vl \
  --dataset geosearcher_demo \
  --dataset_dir "$GEOSEARCHER_ROOT/data" \
  --cutoff_len 4096 \
  --max_new_tokens 1536 \
  --image_max_pixels 1048576 \
  --batch_size 16 \
  --temperature 0 \
  --top_p 1.0 \
  --skip_special_tokens False \
  --save_name "$GEOSEARCHER_ROOT/outputs/batch_predictions.jsonl"
```

<!-- 中文：运行结果保存为 JSONL，每行通常有 prompt、predict 和 label 三个字段。请注意以上推理超参数是开源演示的初始建议，不一定与论文正式评测配置完全一致。 -->
The generated file contains the input prompt, model prediction, and reference label (when provided) for each sample. The settings above are suggested for the public inference example and **should not be assumed to exactly reproduce the paper's evaluation configuration**.

<!-- 中文：如果显存不足，可先降低 batch_size、image_max_pixels，或配置 vLLM 的 gpu_memory_utilization/max_num_seqs。 -->
If you encounter GPU out-of-memory errors, reduce the batch size or image resolution, or configure vLLM memory utilization. Batch generation supports multiple samples by adding further entries to `data/geosearcher_demo_source.json`.

## 📁 Benchmarks

<!-- 中文：在 DIOR-RSVG、OPT-RSVG 和 VRS-Bench 上评估。下面数据来自最终论文；完整复现还需要相同图像预处理、提示词、checkpoint 和坐标解析协议。 -->
GeoSearcher is evaluated on three RSVG benchmarks:

| Benchmark | Pr@0.5 (%) | Pr@0.7 (%) | mIoU (%) |
|:--|--:|--:|--:|
| DIOR-RSVG | **83.2** | **72.4** | **73.5** |
| OPT-RSVG | **81.2** | **65.5** | **69.4** |
| VRS-Bench | **80.3** | **64.4** | **69.2** |

Please obtain benchmark data from their original sources and follow the respective data licenses. The scripts above demonstrate inference; full benchmark reproduction requires the evaluation protocol used in the paper.

## 🙏 Acknowledgments

GeoSearcher builds upon the [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) model family. We thank the developers of the DIOR-RSVG, OPT-RSVG, and VRS-Bench datasets and the open-source research community.

## 📄 License

<!-- 中文：代码和权重许可可能不同。待确认机构授权、基础模型和训练数据许可后正式填写。 -->
The code and model-weight licenses will be specified when the corresponding resources are released. Please also review the terms of the underlying base model and datasets.
