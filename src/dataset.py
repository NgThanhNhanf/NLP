import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import spacy
import io
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# 3. Custom Dataset Class
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
        # 1. Lấy câu thô
        src_text = self.src_data[idx].strip().lower()
        trg_text = self.trg_data[idx].strip().lower()
        
        # 2. Tokenize
        src_tokens = self.src_tokenizer(src_text) 
        trg_tokens = self.trg_tokenizer(trg_text)
        
        # 3. Chuyển sang indices 
        src_indices = [self.src_vocab[token] for token in src_tokens] 
        trg_indices = [self.trg_vocab[token] for token in trg_tokens] 
        
        # 4. Trả về tensor
        return torch.tensor(src_indices, dtype=torch.long), \
               torch.tensor(trg_indices, dtype=torch.long)

# 4. Class Collator (Để xử lý Padding trong DataLoader)
class Collator:
    def __init__(self, src_pad_idx, trg_pad_idx, src_sos_idx, src_eos_idx, trg_sos_idx, trg_eos_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_sos_idx = src_sos_idx
        self.src_eos_idx = src_eos_idx
        self.trg_sos_idx = trg_sos_idx
        self.trg_eos_idx = trg_eos_idx

    def __call__(self, batch):
        src_batch, trg_batch, src_lengths = [], [], []
        
        for src_item, trg_item in batch:
            # Thêm <sos> và <eos> cho source và target
            src_p = torch.cat([
                torch.tensor([self.src_sos_idx]), 
                src_item, 
                torch.tensor([self.src_eos_idx])
            ], dim=0)
            
            trg_p = torch.cat([
                torch.tensor([self.trg_sos_idx]), 
                trg_item, 
                torch.tensor([self.trg_eos_idx])
            ], dim=0)
            
            src_batch.append(src_p)
            trg_batch.append(trg_p)
            src_lengths.append(len(src_p))
        
        # Pad sequence với batch_first=True
        src_batch = pad_sequence(src_batch, padding_value=self.src_pad_idx, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=self.trg_pad_idx, batch_first=True)
        
        # Chuyển src_lengths từ list sang tensor
        # print(tuple(src_lengths))
        src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        
        return src_batch, trg_batch, src_lengths
    
def yield_tokens(filepath, tokenizer):
    with io.open(filepath, encoding='utf-8') as f:
        for line in f:
            yield tokenizer(line.strip().lower())

# 5. Hàm chính để gọi từ Main (Helper function)
def build_vocab_and_tokenizers(en_tokenizer, fr_tokenizer):
    src_filepath = 'data/raw/train.en'
    trg_filepath = 'data/raw/train.fr'
    
    print("Đang xây dựng từ điển (Vocabulary)...")
    src_vocab = build_vocab_from_iterator(yield_tokens(src_filepath, en_tokenizer), min_freq=2, specials=['<unk>', '<pad>', '<sos>', '<eos>'],
                                          special_first=True, max_tokens=10000)
    trg_vocab = build_vocab_from_iterator(yield_tokens(trg_filepath, fr_tokenizer), min_freq=2, specials=['<unk>', '<pad>', '<sos>', '<eos>'],
                                          special_first=True, max_tokens=10000)
    src_vocab.set_default_index(src_vocab['<unk>'])
    trg_vocab.set_default_index(trg_vocab['<unk>'])  
    
    return src_vocab, trg_vocab

def get_data_loaders(batch_size=32, en_tokenizer=None, fr_tokenizer=None, src_vocab=None, trg_vocab=None):
    
    # Định nghĩa đường dẫn file
    train_src, train_trg = 'data/raw/train.en', 'data/raw/train.fr'
    val_src, val_trg = 'data/raw/val.en', 'data/raw/val.fr'
    test_src, test_trg = 'data/raw/test_2016_flickr.en', 'data/raw/test_2016_flickr.fr'
    
    # Tạo Dataset
    print("Đang tạo Dataset...")
    train_dataset = TranslationDataset(train_src, train_trg, src_vocab, trg_vocab, en_tokenizer, fr_tokenizer)
    valid_dataset = TranslationDataset(val_src, val_trg, src_vocab, trg_vocab, en_tokenizer, fr_tokenizer)
    test_dataset = TranslationDataset(test_src, test_trg, src_vocab, trg_vocab, en_tokenizer, fr_tokenizer)
    
    # Tạo Collator
    collator = Collator(
        src_pad_idx=src_vocab['<pad>'], trg_pad_idx=trg_vocab['<pad>'],
        src_sos_idx=src_vocab['<sos>'], src_eos_idx=src_vocab['<eos>'],
        trg_sos_idx=trg_vocab['<sos>'], trg_eos_idx=trg_vocab['<eos>']
    )
    
    # Tạo DataLoader
    print("Đang tạo DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    return train_loader, valid_loader, test_loader
    
