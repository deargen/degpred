import torch 
import torch.nn as nn
from pathlib import Path
from tape import ProteinBertModel, TAPETokenizer


class DegpredEmbedder:
    def __init__(self, device='cpu'):
        self.device = device
        self.tokenizer = TAPETokenizer(vocab='iupac')
        self.model = ProteinBertModel.from_pretrained('bert-base').to(self.device)

    def __call__(self, seq):
        seq_tensor = torch.tensor([self.tokenizer.encode(seq)]).to(self.device)
        seq_bert = self.model(seq_tensor)[0][0]
        return seq_bert
        
# define the architecture
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
        return output[:, 1:-1, :]
    
def get_degpred_model(fold, device='cpu'):
    assert fold in [1, 2, 3, 4, 5]
    ckpt_path = Path(__file__).parent / 'five_model' / f'degpred_model{fold}.pt'
    assert ckpt_path.exists(), ckpt_path
    
    model = DEG_LSTM().to(device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    return model

if __name__ == '__main__':
    device = 'cuda:0'
    
    model = get_degpred_model(1, device=device)
    embed = DegpredEmbedder(device=device)
    
    seq = 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'
    embedding = embed(seq)
    out = model(embedding.unsqueeze(0))[0, :, 0]
    print(torch.where(out>0.3))
    
