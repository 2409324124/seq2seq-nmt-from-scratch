# utils.py
import re
import unicodedata
from collections import defaultdict
import spacy
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

# 加载 spacy 分词器（全局）
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

def tokenize_de(text):
    """德语分词"""
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """英语分词"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def unicode_to_ascii(s):
    """把 unicode 转成 ascii"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    """标准化字符串：小写、去除多余符号、加空格"""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

class Lang:
    """语言类：管理词表、索引、计数"""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = defaultdict(int)
        self.index2word = {}
        self.n_words = 0

        # 立即添加特殊 token（索引固定 0,1,2）
        self.add_word("<SOS>")
        self.add_word("<EOS>")
        self.add_word("<PAD>")

    def add_word(self, word):
        """添加单个词"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.word2count[word] += 1

    def add_sentence(self, sentence):
        """添加整句"""
        for word in sentence:
            self.add_word(word)

def prepare_data(max_length=25, min_freq=2):
    """加载 Multi30k 数据集，构建词表和 pair"""
    print("正在加载 Multi30k 数据集...")
    dataset = load_dataset("bentrevett/multi30k")

    train_data = dataset['train']

    input_lang = Lang("de")   # 德语 → 输入
    output_lang = Lang("en")  # 英语 → 输出

    # 强制确保特殊 token 存在（保险）
    for special in ["<SOS>", "<EOS>", "<PAD>"]:
        input_lang.add_word(special)
        output_lang.add_word(special)

    pairs = []
    for example in train_data:
        de_sent = normalize_string(example['de'])
        en_sent = normalize_string(example['en'])

        de_tokens = tokenize_de(de_sent)
        en_tokens = tokenize_en(en_sent)

        # 过滤过长句子
        if len(de_tokens) > max_length or len(en_tokens) > max_length:
            continue

        pairs.append((list(reversed(de_tokens)), en_tokens))  # ← 关键！反转德语句子
        input_lang.add_sentence(list(reversed(de_tokens)))    # 也要反转加进词表
        output_lang.add_sentence(en_tokens)

    # 可选：过滤低频词（这里先不删，只打印）
    print(f"数据集准备完成！")
    print(f"德语词汇量: {input_lang.n_words} (含特殊token)")
    print(f"英语词汇量: {output_lang.n_words} (含特殊token)")
    print(f"示例对数: {len(pairs)}")

    # 调试打印：确认特殊 token 存在
    print("\n特殊 token 检查（德语输入）：")
    print(" <SOS> index:", input_lang.word2index.get("<SOS>", "缺失"))
    print(" <EOS> index:", input_lang.word2index.get("<EOS>", "缺失"))
    print(" <PAD> index:", input_lang.word2index.get("<PAD>", "缺失"))

    return input_lang, output_lang, pairs

class TranslationDataset(Dataset):
    """自定义 Dataset"""
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]

        # 词转索引，未知词用 PAD=2
        src_indices = [self.input_lang.word2index.get(w, 2) for w in src_tokens]
        tgt_indices = [self.output_lang.word2index.get(w, 2) for w in tgt_tokens]

        # 添加 SOS 和 EOS，使用 get() 防万一
        sos_idx = self.input_lang.word2index.get("<SOS>", 0)
        eos_idx = self.input_lang.word2index.get("<EOS>", 1)
        src_tensor = torch.tensor([sos_idx] + src_indices + [eos_idx])

        sos_idx_tgt = self.output_lang.word2index.get("<SOS>", 0)
        eos_idx_tgt = self.output_lang.word2index.get("<EOS>", 1)
        tgt_tensor = torch.tensor([sos_idx_tgt] + tgt_indices + [eos_idx_tgt])

        return src_tensor, tgt_tensor

def collate_fn(batch):
    """batch padding 处理"""
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=2)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=2)

    return src_padded, tgt_padded