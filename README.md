<div align="center">

# GeoSearcher

### Anchor-Guided Progressive Reasoning for Remote Sensing Visual Grounding with Process Supervision

**Accepted by _Information Fusion_ (2026)**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.01050-b31b1b.svg)](https://arxiv.org/abs/2607.01050)
[![Journal](https://img.shields.io/badge/Information%20Fusion-Accepted-2f72b7.svg)](https://www.sciencedirect.com/journal/information-fusion)

[**Paper**](https://arxiv.org/abs/2607.01050) 

</div>

## 🔍 Overview

**GeoSearcher** is an anchor-guided progressive reasoning framework for **remote sensing visual grounding (RSVG)**. Instead of directly generating target coordinates in a single step, GeoSearcher uses explicit reference-object anchors to progressively integrate location, spatial-relation, and attribute clues before localizing the final target.

Built upon **Qwen3-VL-4B**, GeoSearcher is trained in two stages:

1. **Anchor-Centric Reasoning Supervised Fine-Tuning (ACR-SFT):** constructs and learns from anchor-centric reasoning trajectories. Intermediate reference-object coordinates are explicitly represented using `<ref_bbox>...</ref_bbox>`.
2. **Process-Faithful Group Relative Policy Optimization (PF-GRPO):** improves grounding and intermediate reasoning behavior using a **Process-Aware Reward (PAR)** and a **Reasoning-Informative Sample Selector (RISS)**.

The explicit anchors guide **reasoning-level spatial focus**; they do not require inference-time image cropping or restricted visual feature extraction.

Once a framework image is uploaded, uncomment the following line:
<p align="center"><img src="framework.pdf" width="95%" alt="GeoSearcher framework"></p>

## 🤗 Model Weights

We are preparing the release of the **GeoSearcher-4B** model checkpoint.

| Model | Backbone | Training | Availability |
|:--|:--|:--|:--|
| **GeoSearcher-4B** | Qwen3-VL-4B | ACR-SFT + PF-GRPO | **Coming soon** |

The download link will be added here after the weights and inference configuration have been verified. The final model release will include the required processor/tokenizer configuration.

## 📁 Benchmarks

GeoSearcher is evaluated on three RSVG benchmarks:

- **DIOR-RSVG:** remote sensing images with language-guided object localization.
- **OPT-RSVG:** visual grounding with complex aerial scenes.
- **VRS-Bench:** remote sensing vision-language benchmark with grounding evaluation.

Please obtain the benchmark data from their respective original sources and follow their data licenses. Evaluation instructions and scripts will be documented when available.

## 🙏 Acknowledgments

GeoSearcher builds upon the [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) model family. We also thank the developers of the DIOR-RSVG, OPT-RSVG, and VRS-Bench datasets and the open-source research community.

## 📄 License

The code and model-weight licenses will be specified when the corresponding resources are released. Please also review the terms of the underlying base model and datasets.
