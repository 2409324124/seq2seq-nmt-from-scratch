# Seq2Seq Neural Machine Translation from Scratch  
从零实现的 Seq2Seq + Bahdanau Attention 神经机器翻译（德语 → 英语，Multi30k 数据集）

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![BLEU](https://img.shields.io/badge/BLEU-56.3-brightgreen)](https://huggingface.co/spaces/xu2409324124/lstm-translator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="model_architecture_bahdanau_lstm.png" alt="Bahdanau Attention + LSTM Seq2Seq Architecture" width="800"/>
  <br>
  <em>Bahdanau (加性) Attention + LSTM Seq2Seq 模型架构图（包含源句子反转 + input feeding 机制）</em>
</p>

## 项目亮点

- 忠实复刻 Sutskever et al. (2014) 核心技巧：**源句子反转** + 加性注意力（Bahdanau Attention）
- LSTM 编码器/解码器（hidden=256~512 可调，dropout=0.4，label smoothing=0.1）
- 训练优化：动态 teacher forcing 衰减、梯度裁剪、AdamW、学习率调度、早停 + 验证监控
- **注意力热图可视化**（Matplotlib/Seaborn），支持解释性分析与错误诊断
- **Beam Search** 解码（size=3~5，支持长度惩罚）
- sacreBLEU 标准化评估
- **实时 Gradio 翻译界面**（已部署 Hugging Face Spaces）

## 最终性能（Multi30k test set, sacreBLEU）

| 配置                          | BLEU 分数 | 备注                              |
|-------------------------------|-----------|-----------------------------------|
| Greedy decoding (best epoch)  | **56.3** | Epoch 35，最佳 checkpoint         |
| Beam search (size=3~5)        | ~54–57   | 实际略波动，可进一步调优          |
| 无源句子反转 baseline         | ~30–35   | 验证 reverse trick 提升显著       |

> 在仅 ~29k 训练句对的 Multi30k 上达到 56+ BLEU，已显著超越大多数开源 seq2seq 教程和 2014–2017 年基准实现。

## Loss 曲线（训练 30+ epoch）

<p align="center">
  <img src="loss_curve_lstm.png" alt="Training & Validation Loss Curve" width="700"/>
  <br>
  <em>蓝色：训练 Loss　　红色：验证 Loss　　最佳验证 Loss：4.3435（早停触发）</em>
</p>

## 注意力热图示例（Greedy + Source Reversed）

<p align="center">
  <img src="attention_heatmap_example.jpeg" alt="Attention Heatmap Example" width="700"/>
  <br>
  <em>横轴：德语源句（已反转）　　纵轴：生成的英语句子　　颜色深度表示关注权重</em>
</p>

## 实时翻译演示

浏览器直接试用（支持任意德语句子输入）：

👉 **[LSTM Translator on Hugging Face Spaces](https://huggingface.co/spaces/xu2409324124/lstm-translator)**

## 如何运行

### 要求
- Python 3.8+
- PyTorch 2.0+（CUDA 推荐）
- GPU：RTX 4060 或以上（8GB+ 显存支持 batch=64~128）

### 步骤
1. 克隆仓库
   ```bash
   git clone https://github.com/2409324124/seq2seq-nmt-from-scratch.git
   cd seq2seq-nmt-from-scratch
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 训练
   ```bash
   python train.py
   ```

4. 测试 & BLEU 计算
   ```bash
   python translate.py --mode test --beam 5
   ```

5. 启动 Gradio 界面
   ```bash
   python translate_gradio.py
   ```

数据集自动从 Hugging Face 下载 Multi30k (en-de)。

## 自学故事 & 鸣谢

本项目由**非 CS 专业（社会学背景）零基础自学者**完成，通过与大语言模型的多轮对话式指导，从环境搭建 → PyTorch 入门 → MNIST CNN → GRU Seq2Seq → LSTM + Attention 全链路实现。

鸣谢：
- PyTorch 官方文档 & 教程
- bentrevett/pytorch-seq2seq（经典参考）
- Hugging Face Datasets & Spaces
- sacreBLEU 库

欢迎 fork、star、提 issue！也欢迎讨论优化方向（如 bidirectional encoder、多头 attention、pretrained embeddings）。

Happy translating! 🚀
```


——————**2026.1.26更新**——————


Bilibili视频已更新：[bilibili](https://www.bilibili.com/video/BV1ezzxBFEA4/?spm_id_from=333.1387.homepage.video_card.click&vd_source=46eef21c98a84797a917421ea20dc08a)
具体的安装步骤和训练流程会逐步以视频形式更新。
