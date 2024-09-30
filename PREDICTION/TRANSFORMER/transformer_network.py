import torch.nn as nn
import torch
import math
from gpu import gpu


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out


class TransAm(nn.Module):
    """使用了Transformer的编码器搭建的预测网络"""

    def __init__(self, n_inputs: int, d_model=250, num_layers=1, dropout=0.1):
        """
        初始化
        :param n_inputs:    输入数据的特征维度
        :param d_model:     词嵌入特征维度
        :param num_layers:  Transformer编码器块的个数
        :param dropout:     dropout概率，防止过拟合
        """
        super(TransAm, self).__init__()

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_inputs * d_model, nhead=5,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(n_inputs * d_model, 1)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        # src = src.reshape((src.shape[0], src.shape[1], -1))
        src = src.permute((1, 0, 2))  # 变换维度0和维度1进行交换，交换之后的shape为[seq_len, batch_size, feature]
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, self.src_mask)
        out = out.permute(1, 0, 2)
        out = self.decoder(out)
        return out

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
    )

    return mask


class Transformer(nn.Module):
    """标准的Transformer编码器-解码器结构"""

    def __init__(self, n_encoder_inputs, n_decoder_inputs, d_model=512, dropout=0.1, num_layer=8):
        """
        初始化
        :param n_encoder_inputs:    输入数据的特征维度
        :param n_decoder_inputs:    编码器输入的特征维度，其实等于编码器输出的特征维度
        :param d_model:             词嵌入特征维度
        :param dropout:             dropout
        :param num_layer:           Transformer块的个数
        """
        super(Transformer, self).__init__()

        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.input_projection = torch.nn.Linear(n_encoder_inputs, d_model)
        self.output_projection = torch.nn.Linear(n_decoder_inputs, d_model)

        self.linear = torch.nn.Linear(d_model, 1)

    def encode_in(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1))
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_out(self, tgt, memory):
        tgt_start = self.output_projection(tgt).permute(1, 0, 2)
        out_sequence_len, batch_size = tgt_start.size(0), tgt_start.size(1)
        pos_decoder = (torch.arange(0, out_sequence_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1))
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        tgt = tgt_start + pos_decoder
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        out = self.linear(out)
        return out

    def forward(self, src, target_in):
        src = self.encode_in(src)
        out = self.decode_out(tgt=target_in, memory=src)
        return out


class LSTMModel(nn.Module):
    def __init__(self, n_input, n_hidden=128, n_layers=3, drop_prob=0.1):
        """
        LSTM初始化函数
        :param n_input:     输入数据的特征维度
        :param n_hidden:    隐藏层特征维度
        :param n_layers:    LSTM块的个数
        :param drop_prob:   dropout的概率
        """
        super(LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, dropout=drop_prob)
        self.fc = nn.Linear(n_hidden, 1)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        _batch_size = x.size(0)
        _hidden = self.init_hidden(_batch_size, x.device)
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm(x, _hidden)
        # x = self.dropout(x)
        out = x.contiguous().view(-1, _batch_size, self.n_hidden)
        out = out.permute(1, 0, 2)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size, device):
        """ Initialize hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))

        return hidden


def lstm_init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    """ for test """
    data = torch.rand((1, 58, 1))  # [batch_size, seq_len, feature_size]
    tgt_in = torch.rand((1, 58, 1))
    print(f'input shape:{data.shape}')
    model = TransAm(n_inputs=1)
    output = model(data)
    print(f'TransAm shape:{output.shape}')
    model = LSTMModel(n_input=1)
    output = model(data)
    print(f'LSTM output shape:{output.shape}')
    model = Transformer(n_encoder_inputs=1, n_decoder_inputs=1)
    output = model(data, tgt_in)
    print(f'Transformer output shape:{output.shape}')
