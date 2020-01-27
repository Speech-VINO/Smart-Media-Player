from torch import nn

class SpeakerDiarization(nn.Module):
    """
    N (n_batch), I (n_input), S (n_seq), L (n_layer), H (n_hidden), C (n_output)
    """
    def __init__(self, n_input, n_output, n_hidden=128, n_layer=4, dropout=0.2):
        super().__init__()
        dropout = 0 if n_layer == 0 else dropout
        
        self.gru = nn.GRU(n_input, n_hidden, n_layer, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_output)
        
    def forward(self, x, h):
        x = x.transpose(1, 2) # (N, I, S) -> (N, S, I)
        x, h = self.gru(x, h) # (N, S, I), (L, N, H) -> (N, S, H), (L, N, H)
        x = self.fc(x) # (N, S, H) -> (N, S, C)
        x = x.transpose(1, 2) # (N, S, C) -> (N, C, S)
        return x, h