import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        # 1. Lớp Embedding: Biến ID từ thành vector
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # 2. Lớp LSTM:
        # input_size: kích thước vector nhúng
        # hidden_size: kích thước vector ẩn
        # num_layers: số lớp chồng lên nhau
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [src_len, batch_size]
        
        # Bước 1: Qua Embedding
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src_len, batch_size, emb_dim]
        
        # Bước 2: Qua LSTM
        # outputs: chứa hidden state của TẤT CẢ các thời điểm (dùng cho Attention sau này)
        # hidden, cell: chỉ là trạng thái của thời điểm CUỐI CÙNG (cái ta cần cho context vector)
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # Trả về hidden và cell để ném sang cho Decoder
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 1. Embedding
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # 2. LSTM
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        # 3. Linear (Fully Connected): Biến vector ẩn thành xác suất dự đoán từ
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input shape: [batch_size] (Chỉ là 1 từ tại thời điểm t)
        # hidden, cell: Lấy từ thời điểm t-1
        
        # Thêm 1 chiều (unsqueeze) để khớp với yêu cầu của LSTM [1, batch_size]
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        # embedded shape: [1, batch_size, emb_dim]
        
        # Qua LSTM
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # Dự đoán từ tiếp theo
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: câu tiếng Anh [src_len, batch_size]
        # trg: câu tiếng Pháp [trg_len, batch_size]
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor để chứa kết quả dự đoán
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 1. Mã hóa câu tiếng Anh (Lấy Context Vector cuối cùng)
        hidden, cell = self.encoder(src)
        
        # Input đầu tiên cho Decoder luôn là token <sos> (Start of Sentence)
        input = trg[0, :]
        
        # 2. Vòng lặp giải mã từng từ
        for t in range(1, trg_len):
            
            # Chạy 1 bước Decoder
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Lưu dự đoán
            outputs[t] = output
            
            # Teacher Forcing: Quyết định xem nên dùng từ dự đoán hay từ đúng để train tiếp?
            # teacher_force = True -> dùng từ đúng (ground truth)
            # teacher_force = False -> dùng từ model vừa đoán
            best_guess = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            
            input = trg[t] if teacher_force else best_guess
            
        return outputs