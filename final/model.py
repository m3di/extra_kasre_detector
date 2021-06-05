import torch

class MyModel(torch.nn.Module):

    def __init__(self, alphabet_size, embedding_size, lstm_size, fc_sizes):
        super(MyModel, self).__init__()

        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size

        self.drop = torch.nn.Dropout(p=0.1)
        self.emb = torch.nn.Embedding(alphabet_size, embedding_size)

        self.rnn = torch.nn.LSTM(embedding_size, lstm_size, bidirectional=True, batch_first=True)

        fc_layers = []
        for a, b in zip([lstm_size*2] + fc_sizes[:-1], fc_sizes):
            fc_layers.append(torch.nn.Linear(a, b))
            fc_layers.append(torch.nn.BatchNorm1d(b))
            fc_layers.append(torch.nn.ReLU())
            fc_layers.append(torch.nn.Dropout(0.3))

        self.fc = torch.nn.Sequential(*fc_layers)
        self.out = torch.nn.Linear(fc_sizes[-1], 1)

    def forward(self, lengths, x):
        s = x.shape[1]
        x = self.emb(x)
        x = self.drop(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]
        x2 = (torch.zeros((x.shape[0], s, 1)) - float('inf')).to(x.device)
        for i, l in enumerate(lengths):
            o = self.out(self.fc(x[i,0:l,:]))
            x2[i,0:l] = o
        return x2