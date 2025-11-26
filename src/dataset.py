import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from torchtext.vocab import vocab
import spacy
import io

# 1. Load Spacy Tokenizer
# Đảm bảo bạn đã chạy lệnh python -m spacy download ...
try:
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')
except OSError:
    print("Đang tải model ngôn ngữ Spacy...")
    from spacy.cli import download
    download('en_core_web_sm')
    download('fr_core_news_sm')
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
        
        # Convert text -> List of IDs
        src_indices = [self.src_vocab[token] for token in self.src_tokenizer(src_text)]
        trg_indices = [self.trg_vocab[token] for token in self.trg_tokenizer(trg_text)]
        
        return torch.tensor(src_indices), torch.tensor(trg_indices)

# 4. Class Collator (Để xử lý Padding trong DataLoader)
# Chúng ta dùng Class thay vì hàm lẻ để truyền được vocab vào trong
class Collator:
    def __init__(self, src_pad_idx, trg_pad_idx, src_sos_idx, src_eos_idx, trg_sos_idx, trg_eos_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_sos_idx = src_sos_idx
        self.src_eos_idx = src_eos_idx
        self.trg_sos_idx = trg_sos_idx
        self.trg_eos_idx = trg_eos_idx

    def __call__(self, batch):
        src_batch, trg_batch = [], []
        for src_item, trg_item in batch:
            # Thêm <sos> và <eos>
            src_p = torch.cat([torch.tensor([self.src_sos_idx]), src_item, torch.tensor([self.src_eos_idx])], dim=0)
            trg_p = torch.cat([torch.tensor([self.trg_sos_idx]), trg_item, torch.tensor([self.trg_eos_idx])], dim=0)
            src_batch.append(src_p)
            trg_batch.append(trg_p)
        
        # Pad sequence (Mặc định batch_first=False -> [seq_len, batch_size])
        src_batch = pad_sequence(src_batch, padding_value=self.src_pad_idx)
        trg_batch = pad_sequence(trg_batch, padding_value=self.trg_pad_idx)
        
        return src_batch, trg_batch

# 5. Hàm chính để gọi từ Main (Helper function)
def build_vocab_and_tokenizers():
    # Thay đổi đường dẫn này nếu file bạn nằm chỗ khác
    # Lưu ý: file train dùng để build vocab
    src_filepath = 'data/raw/train.en'
    trg_filepath = 'data/raw/train.fr'
    
    print("Đang xây dựng từ điển (Vocabulary)...")
    src_vocab = build_vocab(src_filepath, tokenize_en)
    trg_vocab = build_vocab(trg_filepath, tokenize_fr)
    
    return src_vocab, trg_vocab, tokenize_en, tokenize_fr

def get_data_loaders(batch_size=32):
    # 1. Lấy Vocab và Tokenizer
    src_vocab, trg_vocab, src_tokenizer, trg_tokenizer = build_vocab_and_tokenizers()
    
    # 2. Định nghĩa đường dẫn file (Đổi tên file thành train.en, test.en như mình dặn)
    train_src, train_trg = 'data/raw/train.en', 'data/raw/train.fr'
    val_src, val_trg = 'data/raw/val.en', 'data/raw/val.fr'
    test_src, test_trg = 'data/raw/test.en', 'data/raw/test.fr'
    
    # 3. Tạo Dataset
    print("Đang tạo Dataset...")
    train_dataset = TranslationDataset(train_src, train_trg, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer)
    valid_dataset = TranslationDataset(val_src, val_trg, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer)
    test_dataset = TranslationDataset(test_src, test_trg, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer)
    
    # 4. Tạo Collator
    collator = Collator(
        src_pad_idx=src_vocab['<pad>'], trg_pad_idx=trg_vocab['<pad>'],
        src_sos_idx=src_vocab['<sos>'], src_eos_idx=src_vocab['<eos>'],
        trg_sos_idx=trg_vocab['<sos>'], trg_eos_idx=trg_vocab['<eos>']
    )
    
    # 5. Tạo DataLoader
    print("Đang tạo DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    return train_loader, valid_loader, test_loader