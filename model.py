import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class AWD(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False):
        super(AWD, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            rnn.module.flatten_parameters()  # not working
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]


class LSTMModel(nn.Module):
    def __init__(self, num_tokens, hidden_size, embed_size, output_size, dropout=0.5, n_layers=1, wdrop=0, dropouth=0.5,
                 dropouti=0.5, dropoute=0.1, tie_weights=False):
        super(LSTMModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.tie_weights = tie_weights
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.dropout = dropout
        self.encoder = nn.Embedding(num_tokens, embed_size)

        # init LSTM layers
        self.lstms = []

        for l in range(n_layers):
            layer_input_size = embed_size if l == 0 else hidden_size
            layer_output_size = hidden_size if l != n_layers - 1 else (embed_size if tie_weights else hidden_size)
            self.lstms.append(nn.LSTM(layer_input_size, layer_output_size, num_layers=1, dropout=0))
        if wdrop:
            # Encapsulate lstms in DropConnect class to tap in on their forward() function and drop connections
            self.lstms = [WeightDrop(lstm, ['weight_hh_l0'], dropout=wdrop) for lstm in self.lstms]
        self.lstms = nn.ModuleList(self.lstms)

        self.decoder = nn.Linear(embed_size if tie_weights else hidden_size, output_size)

        if tie_weights:
            # Tie weights
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden):
        # Do embedding dropout
        emb = embedded_dropout(self.encoder, inp, dropout=self.dropoute if self.training else 0)
        # Do variational dropout
        emb = self.lockdrop(emb, self.dropouti)

        new_hidden = []
        outputs = []
        output = emb
        for i, lstm in enumerate(self.lstms):
            output, new_hid = lstm(output, hidden[i])

            new_hidden.append(new_hid)
            if i != self.n_layers - 1:
                # Do variational dropout
                output = self.lockdrop(output, self.dropouth)

        hidden = new_hidden
        # Do variational dropout
        output = self.lockdrop(output, self.dropout)

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return [(weight.new(1, bsz, self.hidden_size if l != self.n_layers - 1 else (
            self.embed_size if self.tie_weights else self.hidden_size)).zero_(),
                 weight.new(1, bsz, self.hidden_size if l != self.n_layers - 1 else (
                     self.embed_size if self.tie_weights else self.hidden_size)).zero_())
                for l in range(self.n_layers)]
