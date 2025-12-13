import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout, batch_first=True, bidirectional=False) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))

        # Xử lý src_len như code cũ của bạn
        if src_len.dim() == 0:
            src_len = src_len.unsqueeze(0)
        if (src_len == 0).any():
            src_len = torch.clamp(src_len, min=1)
        
        # Packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        
        # --- THAY ĐỔI QUAN TRỌNG CHO ATTENTION ---
        # Unpack sequence để lấy outputs của tất cả các bước thời gian
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, src_len, hidden_size]
        # hidden, cell: [num_layers, batch_size, hidden_size]
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # W kết hợp hidden state của decoder và encoder output
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size] (hidden state hiện tại của decoder)
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        src_len = encoder_outputs.shape[1]
        
        # Lặp lại hidden state của decoder src_len lần để khớp kích thước với encoder_outputs
        # hidden: [batch_size, src_len, hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Tính năng lượng (energy) - mức độ phù hợp
        # torch.cat -> [batch_size, src_len, hidden_size * 2]
        # self.attn -> [batch_size, src_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Tính attention scores
        # self.v -> [batch_size, src_len, 1]
        attention = self.v(energy).squeeze(2) # [batch_size, src_len]
        
        # Softmax để ra xác suất (trọng số)
        return F.softmax(attention, dim=1)

class DecoderAttention(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(DecoderAttention, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.attention = Attention(hidden_size) # Thêm lớp Attention
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        # LSTM input = embedding + context_vector (từ attention)
        self.rnn = nn.LSTM(hidden_size * 2, hidden_size, num_layers=num_layers, 
                           dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size, 1]
        # hidden: [num_layers, batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        input = input.unsqueeze(1) # [batch_size, 1]
        embedded = self.dropout(self.embedding(input)) # [batch_size, 1, hidden_size]
        
        # --- TÍNH ATTENTION ---
        # Dùng hidden state lớp cuối cùng của decoder bước trước để tính attention
        # hidden[-1]: [batch_size, hidden_size]
        a = self.attention(hidden[-1], encoder_outputs) # [batch_size, src_len]
        
        # Tính Context Vector (Weighted sum của encoder outputs)
        a = a.unsqueeze(1) # [batch_size, 1, src_len]
        
        # bmm (batch matrix multiplication): [batch, 1, src_len] * [batch, src_len, hidden]
        context = torch.bmm(a, encoder_outputs) # [batch_size, 1, hidden_size]
        
        # --- KẾT HỢP VÀ ĐƯA VÀO LSTM ---
        # Kết hợp embedding của từ hiện tại với context vector
        rnn_input = torch.cat((embedded, context), dim=2) # [batch_size, 1, hidden_size * 2]
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Dự đoán từ
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, source_lengths, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Encoder trả về cả outputs (cho attention) và hidden/cell
        encoder_outputs, hidden, cell = self.encoder(source, source_lengths)
        
        input = target[:, 0] # Start token <sos>
        
        for t in range(1, target_len):
            # Truyền encoder_outputs vào decoder
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            
            outputs[:, t] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            input = target[:, t] if teacher_force else top1
        
        return outputs