"""
MiniFASNet: rede leve de anti-spoofing facial.
Arquitetura reproduzida de minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0).
Modelo: 2.7_80x80_MiniFASNetV2.pth  — entrada 80x80, 3 classes, ~0.43M params.
"""
import os
import cv2
import torch
import numpy as np
import urllib.request
from collections import OrderedDict
from torch.nn import (
    Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid,
    AdaptiveAvgPool2d, Sequential, Module,
)

# =============================================
# BLOCOS DA ARQUITETURA
# =============================================

class _Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class _Conv(Module):
    def __init__(self, ic, oc, k=(1,1), s=(1,1), p=(0,0), g=1):
        super().__init__()
        self.conv  = Conv2d(ic, oc, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn    = BatchNorm2d(oc)
        self.prelu = PReLU(oc)
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class _Linear(Module):
    def __init__(self, ic, oc, k=(1,1), s=(1,1), p=(0,0), g=1):
        super().__init__()
        self.conv = Conv2d(ic, oc, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn   = BatchNorm2d(oc)
    def forward(self, x):
        return self.bn(self.conv(x))


class _DepthWise(Module):
    def __init__(self, c1, c2, c3, residual=False, k=(3,3), s=(2,2), p=(1,1)):
        super().__init__()
        self.conv    = _Conv(c1[0], c1[1], k=(1,1))
        self.conv_dw = _Conv(c2[0], c2[1], k=k, s=s, p=p, g=c2[0])
        self.project = _Linear(c3[0], c3[1], k=(1,1))
        self.residual = residual
    def forward(self, x):
        sc = x if self.residual else None
        x  = self.conv(x)
        x  = self.conv_dw(x)
        x  = self.project(x)
        return (sc + x) if self.residual else x


class _SE(Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = AdaptiveAvgPool2d(1)
        self.fc1  = Conv2d(channels, channels // r, 1, bias=False)
        self.bn1  = BatchNorm2d(channels // r)
        self.relu = ReLU(inplace=True)
        self.fc2  = Conv2d(channels // r, channels, 1, bias=False)
        self.bn2  = BatchNorm2d(channels)
        self.sig  = Sigmoid()
    def forward(self, x):
        w = self.sig(self.bn2(self.fc2(self.relu(self.bn1(self.fc1(self.pool(x)))))))
        return x * w


class _Residual(Module):
    def __init__(self, c1, c2, c3, n, k=(3,3), s=(1,1), p=(1,1)):
        super().__init__()
        self.model = Sequential(*[
            _DepthWise(c1[i], c2[i], c3[i], residual=True, k=k, s=s, p=p)
            for i in range(n)
        ])
    def forward(self, x):
        return self.model(x)


# =============================================
# MODELO MiniFASNetV2 (keep_dict '1.8M_')
# =============================================
_KEEP = [32,32,103,103,64,13,13,64,13,13,64,13,13,64,13,13,64,
         231,231,128,231,231,128,52,52,128,26,26,128,77,77,128,
         26,26,128,26,26,128,308,308,128,26,26,128,26,26,128,512,512]


class _MiniFASNetV2(Module):
    def __init__(self, conv6_kernel=(5,5), embedding_size=128, num_classes=3):
        super().__init__()
        k = _KEEP
        self.conv1    = _Conv(3, k[0], k=(3,3), s=(2,2), p=(1,1))
        self.conv2_dw = _Conv(k[0], k[1], k=(3,3), s=(1,1), p=(1,1), g=k[1])
        self.conv_23  = _DepthWise((k[1],k[2]),(k[2],k[3]),(k[3],k[4]), k=(3,3), s=(2,2), p=(1,1))

        c1=[(k[4],k[5]),(k[7],k[8]),(k[10],k[11]),(k[13],k[14])]
        c2=[(k[5],k[6]),(k[8],k[9]),(k[11],k[12]),(k[14],k[15])]
        c3=[(k[6],k[7]),(k[9],k[10]),(k[12],k[13]),(k[15],k[16])]
        self.conv_3   = _Residual(c1,c2,c3,4)

        self.conv_34  = _DepthWise((k[16],k[17]),(k[17],k[18]),(k[18],k[19]), k=(3,3), s=(2,2), p=(1,1))

        c1=[(k[19],k[20]),(k[22],k[23]),(k[25],k[26]),(k[28],k[29]),(k[31],k[32]),(k[34],k[35])]
        c2=[(k[20],k[21]),(k[23],k[24]),(k[26],k[27]),(k[29],k[30]),(k[32],k[33]),(k[35],k[36])]
        c3=[(k[21],k[22]),(k[24],k[25]),(k[27],k[28]),(k[30],k[31]),(k[33],k[34]),(k[36],k[37])]
        self.conv_4   = _Residual(c1,c2,c3,6)

        self.conv_45  = _DepthWise((k[37],k[38]),(k[38],k[39]),(k[39],k[40]), k=(3,3), s=(2,2), p=(1,1))

        c1=[(k[40],k[41]),(k[43],k[44])]
        c2=[(k[41],k[42]),(k[44],k[45])]
        c3=[(k[42],k[43]),(k[45],k[46])]
        self.conv_5     = _Residual(c1,c2,c3,2)
        self.conv_6_sep = _Conv(k[46], k[47], k=(1,1))
        self.conv_6_dw  = _Linear(k[47], k[48], k=conv6_kernel, g=k[48])
        self.flatten    = _Flatten()
        self.linear     = Linear(512, embedding_size, bias=False)
        self.bn         = BatchNorm1d(embedding_size)
        self.drop       = torch.nn.Dropout(p=0.2)
        self.prob       = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x);    x = self.conv2_dw(x); x = self.conv_23(x)
        x = self.conv_3(x);   x = self.conv_34(x);  x = self.conv_4(x)
        x = self.conv_45(x);  x = self.conv_5(x)
        x = self.conv_6_sep(x); x = self.conv_6_dw(x); x = self.flatten(x)
        x = self.linear(x);   x = self.bn(x); x = self.drop(x)
        return self.prob(x)


# =============================================
# CARREGAMENTO E INFERÊNCIA
# =============================================
MODELO_PATH  = 'anti_spoof_model.pth'
_INPUT_SIZE  = (80, 80)
_CONV6_KER   = (5, 5)   # get_kernel(80, 80) = ((80+15)//16, ...) = (5, 5)
CLASSE_REAL  = 1         # índice da classe "rosto real" (0=spoof, 1=real, 2=spoof)
THRESHOLD    = 0.55      # probabilidade mínima para confirmar rosto real

_MODELO_URL = (
    "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
    "/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
)

_model_cache = None


def _baixar_modelo():
    print(f"⬇️  Baixando pesos do MiniFASNet (~1.7 MB)...")
    tmp = MODELO_PATH + ".tmp"
    try:
        def _progresso(n_blocos, tam_bloco, tam_total):
            if tam_total > 0:
                pct = min(100, n_blocos * tam_bloco * 100 // tam_total)
                print(f"\r   {pct}%", end="", flush=True)
        urllib.request.urlretrieve(_MODELO_URL, tmp, reporthook=_progresso)
        print()  # quebra linha após a barra
        os.replace(tmp, MODELO_PATH)
        print(f"✅ Modelo salvo em '{MODELO_PATH}'.")
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Falha ao baixar o modelo: {exc}\n"
            f"Baixe manualmente com:\n"
            f"  curl -L -o {MODELO_PATH} '{_MODELO_URL}'"
        ) from exc


def carregar_modelo():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODELO_PATH):
        _baixar_modelo()

    model = _MiniFASNetV2(conv6_kernel=_CONV6_KER)
    state_dict = torch.load(MODELO_PATH, map_location='cpu', weights_only=True)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    model.eval()
    _model_cache = model
    print("✅ Modelo MiniFASNet carregado.")
    return _model_cache


def prever_liveness(frame_bgr, box_clean):
    """
    Retorna probabilidade [0,1] de ser rosto real.
    >= THRESHOLD → real  |  < THRESHOLD → spoof.
    """
    x_min, y_min, x_max, y_max = box_clean
    margem = int(max(x_max - x_min, y_max - y_min) * 0.15)
    ih, iw = frame_bgr.shape[:2]
    x1 = max(0, x_min - margem)
    y1 = max(0, y_min - margem)
    x2 = min(iw, x_max + margem)
    y2 = min(ih, y_max + margem)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    crop_rgb = cv2.cvtColor(cv2.resize(crop, _INPUT_SIZE), cv2.COLOR_BGR2RGB)
    tensor   = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    model = carregar_modelo()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return float(probs[CLASSE_REAL])
