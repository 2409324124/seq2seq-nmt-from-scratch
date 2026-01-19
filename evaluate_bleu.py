# evaluate_bleu_parallel.py - 批量并行评估所有 epoch 的 BLEU（最优版 + 源序列反转）
import torch
from torch.utils.data import DataLoader
import sacrebleu
from tqdm import tqdm
import csv
import time
import multiprocessing as mp
from utils import prepare_data, TranslationDataset, collate_fn
from models import EncoderLSTM, AttnDecoderLSTM

# ------------------- 参数 -------------------
hidden_size = 256
batch_size = 3072         # 测试时 batch 越大越快（4060 8GB 可承受 256~512）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 多进程并行数量（根据 CPU 核心调整，建议 2~4）
num_processes = 2

# 评估范围
start_epoch = 1
end_epoch = 37  # 你的最大 epoch

# 加载词表（只加载一次）
input_lang, output_lang, _ = prepare_data(max_length=25, min_freq=2)

# 加载 test set（只加载一次）
_, _, test_pairs = prepare_data(max_length=25, min_freq=2)
test_dataset = TranslationDataset(test_pairs, input_lang, output_lang)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 评估单个 epoch 的函数（供多进程调用）
def evaluate_epoch(epoch):
    start_time = time.time()
    print(f"进程启动 - 开始评估 Epoch {epoch}...")

    try:
        encoder = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words, dropout=0.4).to(device)

        encoder.load_state_dict(torch.load(f"encoder_lstm_epoch{epoch}.pt", map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(f"decoder_lstm_epoch{epoch}.pt", map_location=device, weights_only=True))

        encoder.eval()
        decoder.eval()
        references = []
        hypotheses = []

        with torch.no_grad():
            for src, tgt in tqdm(test_loader, desc=f"Epoch {epoch}", leave=False):
                src = src.to(device)
                tgt = tgt.to(device)

                # 关键修改：反转源序列（Sutskever 2014 技巧）
                src_reversed = torch.flip(src, dims=[1])  # 反转序列维度（batch, seq_len → seq_len 倒序）

                encoder_outputs, (encoder_hidden, encoder_cell) = encoder(src_reversed)

                for i in range(src.size(0)):
                    decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
                    decoder_hidden_i = encoder_hidden[:, i:i+1, :]
                    decoder_cell_i = encoder_cell[:, i:i+1, :]

                    translated = []
                    for _ in range(50):
                        output, decoder_hidden_i, decoder_cell_i, _ = decoder(
                            decoder_input, decoder_hidden_i, decoder_cell_i, encoder_outputs[i:i+1]
                        )
                        topv, topi = output.topk(1)
                        if topi.item() == output_lang.word2index["<EOS>"]:
                            break
                        translated.append(output_lang.index2word[topi.item()])
                        decoder_input = topi.detach()

                    hypotheses.append(" ".join(translated))

                    ref_tokens = [output_lang.index2word[idx.item()] for idx in tgt[i] if idx.item() not in [0, 1, 2]]
                    if not ref_tokens:
                        ref_tokens = ['<EMPTY>']
                    ref = " ".join(ref_tokens)
                    references.append([ref])

        bleu = sacrebleu.corpus_bleu(hypotheses, references)
        print(f"Epoch {epoch} BLEU: {bleu.score:.2f} (1-gram: {bleu.precisions[0]:.2f}, 2-gram: {bleu.precisions[1]:.2f}, 3-gram: {bleu.precisions[2]:.2f}, 4-gram: {bleu.precisions[3]:.2f})")
        print(f"Epoch {epoch} 评估时间: {time.time() - start_time:.0f} 秒")

        return (epoch, bleu.score, bleu.precisions[0], bleu.precisions[1], bleu.precisions[2], bleu.precisions[3])

    except Exception as e:
        print(f"Epoch {epoch} 评估失败: {str(e)}")
        return None

# ------------------- 主程序（多进程并行） -------------------
if __name__ == '__main__':
    print(f"启动 {num_processes} 个进程并行评估 Epoch {start_epoch} 到 {end_epoch}...")

    with mp.Pool(processes=num_processes) as pool:
        async_results = [pool.apply_async(evaluate_epoch, (epoch,)) for epoch in range(start_epoch, end_epoch + 1)]
        results = [res.get() for res in async_results if res.get() is not None]

    # 保存结果到 CSV
    with open('bleu_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'BLEU', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'])
        for row in results:
            writer.writerow(row)

    print("\n评估完成！结果已保存到 bleu_results.csv")
    print("打开 CSV 查看所有 epoch 的 BLEU 分数，找到最高的那一轮作为最终模型")