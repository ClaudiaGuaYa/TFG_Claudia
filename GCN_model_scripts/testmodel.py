import os
import torch
import torchaudio


import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int, use_bn: bool = True):
        super().__init__()
        self.num_features = num_features
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x, cond):
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)
        if self.use_bn:
            x = self.bn(x)
        x = (x * g) + b
        return x

class Conv1dCausal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride,
            padding=0, dilation=dilation, bias=bias
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x

class GatedAF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_tanh, x_sigmoid = x.chunk(2, dim=1)
        x_tanh = torch.tanh(x_tanh)
        x_sigmoid = torch.sigmoid(x_sigmoid)
        return x_tanh * x_sigmoid

class GCN1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, stride=1, cond_dim=0, use_bias_in_conv=False, use_bn=False):
        super().__init__()
        self.conv = Conv1dCausal(in_ch, out_ch * 2, kernel_size, stride, dilation, use_bias_in_conv)
        self.film = FiLM(cond_dim, out_ch * 2) if cond_dim > 0 else None
        self.gated_activation = GatedAF()
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x, cond):
        x_in = x
        x = self.conv(x)
        if cond is not None and self.film is not None:
            x = self.film(x, cond)
        x = self.gated_activation(x)
        x_res = self.res(x_in)
        x = x + x_res
        return x

class GCN1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_blocks=4, n_channels=32, dil_growth=16, kernel_size=13, cond_dim=3, use_act=True, use_bias_in_conv=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dil_growth = dil_growth
        self.n_blocks = n_blocks
        self.cond_dim = cond_dim
        self.use_act = use_act
        self.use_bias_in_conv = use_bias_in_conv

        self.channels = [n_channels] * n_blocks
        self.dilations = [dil_growth**idx for idx in range(n_blocks)]
        self.strides = [1] * n_blocks

        self.blocks = nn.ModuleList()
        block_out_ch = None
        for idx, (curr_out_ch, dil, stride) in enumerate(zip(self.channels, self.dilations, self.strides)):
            block_in_ch = in_ch if idx == 0 else block_out_ch
            block_out_ch = curr_out_ch
            self.blocks.append(
                GCN1DBlock(
                    block_in_ch, block_out_ch, self.kernel_size, dilation=dil, stride=stride,
                    cond_dim=self.cond_dim, use_bias_in_conv=self.use_bias_in_conv
                )
            )
        self.out_net = nn.Conv1d(self.channels[-1], out_ch, kernel_size=1, stride=1, bias=False)
        self.act = nn.Tanh()

    def forward(self, x, cond):
        assert x.ndim == 3
        if cond is not None:
            assert cond.ndim == 2
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_net(x)
        if self.act is not None:
            x = self.act(x)
        return x


modelo_path = "audios/chorus_model.pt"  
entrada_dir = "audios/inputs"
salida_dir = "audios/outputs"
os.makedirs(salida_dir, exist_ok=True)


model = GCN1D(
    n_blocks=4,
    n_channels=32,
    dil_growth=16,
    kernel_size=13,
    cond_dim=3,
)
checkpoint = torch.load(modelo_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

cond = torch.zeros((1, 3))  

for archivo in os.listdir(entrada_dir):
    if not archivo.endswith(".wav"):
        continue

    ruta_entrada = os.path.join(entrada_dir, archivo)
    ruta_salida = os.path.join(salida_dir, f"procesado_{archivo}")

    print(f"Procesando: {ruta_entrada}")

    waveform, sample_rate = torchaudio.load(ruta_entrada)  

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform / waveform.abs().max()

    with torch.no_grad():
        entrada = waveform.unsqueeze(0) 
        salida = model(entrada, cond)
        if isinstance(salida, tuple):
            salida = salida[0]
        salida = salida.squeeze(0)  

    torchaudio.save(ruta_salida, salida, sample_rate)

print("Todos los archivos han sido procesados.")
