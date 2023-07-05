import torch
import torch.nn as nn 
from pathlib import Path

class DEG_LSTM(nn.Module):
    def __init__(self, input_size=768, deg_lstm_hidden_size=32, fc1_output_size=8, output_size=1):
        super().__init__()
        self.deg_lstm = nn.LSTM(input_size, deg_lstm_hidden_size, 2, bidirectional=True, batch_first = True)
        self.deg_fc1 = nn.Linear(deg_lstm_hidden_size*2, fc1_output_size)
        self.deg_fc2 = nn.Linear(fc1_output_size, output_size)

        
    def forward(self, input):
        output, (h, c) = self.deg_lstm(input)
        output = self.deg_fc1(output)
        output = torch.sigmoid(self.deg_fc2(output))
        return output

def get_degpred_model(fold, device='cpu'):
    assert fold in [1, 2, 3, 4, 5]
    
    ckpt_path = Path(__file__).parent / 'five_model' / f'degpred_model{fold}.pkl'
    model = torch.load(ckpt_path, map_location=device)#.to(device=device)
    return model

if __name__ == '__main__':
    for fold in [1, 2, 3, 4, 5]:
        model = get_degpred_model(fold)
        pt_ckpt_path = Path(__file__).parent / 'five_model' / f'degpred_model{fold}.pt'
        torch.save(model.state_dict(), pt_ckpt_path)
    