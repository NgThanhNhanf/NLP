import torch
import spacy
import nltk
import matplotlib.pyplot as plt

# Kiểm tra xem có nhận GPU không (Nếu dùng Colab nhớ bật GPU: Runtime -> Change runtime type -> T4 GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Thiết bị đang sử dụng: {device}")

# Kiểm tra Spacy load được model chưa
try:
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')
    print("Spacy: Đã load thành công model Anh - Pháp!")
except OSError:
    print("LỖI: Chưa tải model Spacy. Hãy chạy lệnh 'python -m spacy download...'")

print("Mọi thứ đã sẵn sàng để code!")