import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from torchtext.vocab import vocab
import spacy
import io

# 1. Load Spacy Tokenizer
spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# 2. Xây dựng từ điển (Vocabulary)
def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    # Tạo vocab object, thêm special tokens
    v = vocab(counter, min_freq=2, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    v.set_default_index(v['<unk>'])
    return v

# 3. Custom Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
        self.src_data = open(src_path, encoding='utf-8').readlines()
        self.trg_data = open(trg_path, encoding='utf-8').readlines()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_text = self.src_data[idx].strip()
        trg_text = self.trg_data[idx].strip()
        
        # Convert text -> List of IDs (Numericalize)
        src_indices = [self.src_vocab[token] for token in self.src_tokenizer(src_text)]
        trg_indices = [self.trg_vocab[token] for token in self.trg_tokenizer(trg_text)]
        
        return torch.tensor(src_indices), torch.tensor(trg_indices)

# 4. Collate Function (Để Padding)
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        # Thêm <sos> và <eos> vào mỗi câu
        src_batch.append(torch.cat([torch.tensor([src_vocab['<sos>']]), src_item, torch.tensor([src_vocab['<eos>']])], dim=0))
        trg_batch.append(torch.cat([torch.tensor([trg_vocab['<sos>']]), trg_item, torch.tensor([trg_vocab['<eos>']])], dim=0))
    
    # Pad cho bằng nhau (Padding value = index của <pad>)
    src_batch = pad_sequence(src_batch, padding_value=src_vocab['<pad>'])
    trg_batch = pad_sequence(trg_batch, padding_value=trg_vocab['<pad>'])
    
    return src_batch, trg_batch