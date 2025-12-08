import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout, batch_first=True, bidirectional=False) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, hidden_size]

        # print(f"Encoder received src_len shape: {src_len.shape}")
        # print(f"Encoder received src_len values: {src_len}")
        
        # Đảm bảo src_len là tensor 1D
        if src_len.dim() == 0:  # Nếu là scalar (batch_size=1)
            src_len = src_len.unsqueeze(0)  # Chuyển thành [1]
        
        # Đảm bảo không có giá trị 0
        if (src_len == 0).any():
            print(f"Warning: Found zero length in src_len: {src_len}")
            # Thay thế giá trị 0 bằng 1
            src_len = torch.clamp(src_len, min=1)
        
        # Packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        # hidden, cell: [num_layers, batch_size, hidden_size]
        
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, hidden_size)

        # LSTM nhận: embedding + context từ encoder
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=num_layers, 
                           dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell, encoder_hidden):
        # input: [batch_size, 1] - từ hiện tại
        # encoder_hidden: [num_layers, batch_size, hidden_size] - từ encoder
        
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, hidden_size]
        
        # Lấy hidden state cuối cùng từ encoder
        encoder_context = encoder_hidden[-1].unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Kết hợp embedding với context
        lstm_input = torch.cat([embedded, encoder_context], dim=2)  # [batch_size, 1, hidden_size*2]
        
        # LSTM xử lý
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch_size, 1, hidden_size]
        
        # Dự đoán từ tiếp theo
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, source_lengths, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Tensor để lưu outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Encoder forward
        encoder_hidden, encoder_cell = self.encoder(source, source_lengths)
        
        # Input đầu tiên cho decoder là <sos> token
        input = target[:, 0]  # [batch_size]
        
        # Khởi tạo hidden, cell cho decoder bằng hidden, cell từ encoder
        hidden, cell = encoder_hidden, encoder_cell
        
        for t in range(1, target_len):
            # Decoder dự đoán từ tiếp theo
            # Dùng encoder_hidden làm context (không đổi)
            output, hidden, cell = self.decoder(
                input.unsqueeze(1),      # [batch_size, 1]
                hidden, cell,            # Hidden, cell từ bước trước
                encoder_hidden           # ← CONTEXT TỪ ENCODER (quan trọng!)
            )
            
            # Lưu dự đoán
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)  # Từ có xác suất cao nhất
            
            # Chọn input cho bước tiếp theo
            input = target[:, t] if teacher_force else top1
        
        return outputs