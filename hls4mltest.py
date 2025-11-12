import torch
import torch.nn as nn
import torch.nn.functional as F
import hls4ml

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        #self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)  # <-- advanced
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        # reshape for attention: (batch, seq_len, embed_dim)
        x = F.elu(self.fc1(x))
        #x = x.unsqueeze(1)  # add seq_len=1 dimension
        #x, _ = self.attn(x, x, x)  # self-attention
        #x = x.squeeze(1)           # remove seq_len
        x = F.elu(self.fc2(x))
        return x

# Instantiate the model
model = SimpleModel()

# (Here you would normally train the model...)

# Create a dummy input for tracing the PyTorch model
example_input = torch.randn(1, 16)  # batch_size=1, input_dim=16

# Generate hls4ml config from PyTorch model
config = hls4ml.utils.config_from_pytorch_model(model, (16,))

print(config)

# Convert to HLS project
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    example_input,
    hls_config=config,
    backend='Vitis'
)

hls_model.build(
    project_name='my_hls_project',  # folder name
    build_dir='hls_output',         # path where files are created
    csim=False,                     # run C simulation? False if you just want files
    synth=True,                     # run synthesis? True will call Vitis HLS
    vsynth=False                    # optional, run post-synthesis if True
)