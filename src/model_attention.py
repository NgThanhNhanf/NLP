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

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]

        src_len = encoder_outputs.size(1)

        dec = self.W_dec(decoder_hidden).unsqueeze(1)
        # [batch_size, 1, hidden_size]

        enc = self.W_enc(encoder_outputs)
        # [batch_size, src_len, hidden_size]

        energy = self.v(torch.tanh(dec + enc)).squeeze(2)
        # [batch_size, src_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(energy, dim=1)

        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        # [batch_size, hidden_size]

        return context, attn_weights

class DecoderAttention(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)

        self.lstm = nn.LSTM(
            hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, input, hidden, cell, encoder_outputs):
    #     # input: [batch_size, 1]
    #     embedded = self.dropout(self.embedding(input))
    #     # [batch_size, 1, hidden_size]

    #     decoder_hidden = hidden[-1]
    #     # [batch_size, hidden_size]

    #     context, attn_weights = self.attention(
    #         decoder_hidden, encoder_outputs
    #     )

    #     context = context.unsqueeze(1)

    #     lstm_input = torch.cat([embedded, context], dim=2)

    #     output, (hidden, cell) = self.lstm(
    #         lstm_input, (hidden, cell)
    #     )

    #     output = output.squeeze(1)
    #     context = context.squeeze(1)

    #     prediction = self.fc_out(
    #         torch.cat([output, context], dim=1)
    #     )

    #     return prediction, hidden, cell, attn_weights

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size, 1]
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, hidden_size]

        decoder_hidden = hidden[-1]
        # [batch_size, hidden_size]

        context, attn_weights = self.attention(
            decoder_hidden, encoder_outputs
        )
        # context: [batch_size, hidden_size]

        context = context.unsqueeze(1)
        # [batch_size, 1, hidden_size]

        lstm_input = torch.cat([embedded, context], dim=2)
        # [batch_size, 1, hidden_size * 2]

        output, (hidden, cell) = self.lstm(
            lstm_input, (hidden, cell)
        )
        # output: [batch_size, 1, hidden_size]

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(
            torch.cat([output, context], dim=1)
        )
        # [batch_size, output_size]

        return prediction, hidden, cell, attn_weights


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
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(
            source, source_lengths
        )

        hidden, cell = encoder_hidden, encoder_cell
        input = target[:, 0]

        for t in range(1, target_len):
            output, hidden, cell, attn = self.decoder(
                input.unsqueeze(1),
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1

        return outputs