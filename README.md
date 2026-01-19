# Seq2Seq Neural Machine Translation from Scratch

从零实现的 Seq2Seq + Bahdanau Attention 神经机器翻译（德语 → 英语，Multi30k 数据集）

## 项目亮点
- 完整复刻 Sutskever 2014 论文核心（Seq2Seq + Attention + source reversal）
- LSTM 模型训练（hidden=256，dropout=0.4，label smoothing）
- 动态 teacher forcing + 梯度裁剪 + AdamW + 学习率调度
- 早停机制 + 验证集监控
- 注意力热图可视化（Matplotlib）
- Beam Search 解码（提升翻译质量）
- 批量 BLEU 评估（sacrebleu）

## Loss 曲线（训练 30+ epoch）
![Loss Curve](loss_curve_lstm.png)

**训练 Loss**：蓝色曲线  
**验证 Loss**：红色曲线  
**最佳验证 Loss**：4.3435（早停触发）

## 注意力热图示例（Greedy + Source Reversed）
![Attention Heatmap](attention_heatmap_example)

- 横轴：德语源句（已反转）
- 纵轴：生成的英语句子
- 亮点表示模型在生成该英语词时关注的德语位置

## 实时翻译界面（Gradio）
运行 `translate_gradio.py` 即可启动浏览器界面，支持实时输入德语句子 → 输出英语翻译。

## 最终模型
- 最佳 checkpoint：**Epoch30**（BLEU 值为58.45）
- 权重文件：`encoder_lstm_epoch30.pt` / `decoder_lstm_epoch30.pt`（已上传或链接）

## 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. 下载数据集：自动从 Hugging Face 下载 Multi30k
3. 训练：`python train.py`
4. 测试翻译：`python translate.py`
5. 实时界面：`python translate_gradio.py`

欢迎 fork / star！项目记录了完整的 debug 过程，从环境坑到过拟合分析。

感谢 PyTorch 官方教程 + bentrevett/pytorch-seq2seq 仓库的启发。