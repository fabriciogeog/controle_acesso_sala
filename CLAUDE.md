# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12 and a working webcam. GPU (CUDA) is used automatically if available, otherwise falls back to CPU.

O arquivo de pesos `anti_spoof_model.pth` não está no repositório (binário ~1.7 MB). Ele é **baixado automaticamente** na primeira execução que ativar o anti-spoofing. Para baixar manualmente:

```bash
curl -L -o anti_spoof_model.pth \
  "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
```

## Running

```bash
# Ponto de entrada principal
python main.py

# Scripts legados (referência histórica, não usar em produção)
python controle_acesso_v2.py
python controle_acesso_v1.py

# Inspecionar embeddings no banco
python teste_embedding.py
```

## Architecture

O projeto está separado em três camadas:

| Arquivo | Responsabilidade |
|---------|-----------------|
| `db.py` | Todas as operações SQLite |
| `face.py` | Modelos ML, câmera, visão computacional |
| `main.py` | Menu, configuração, ponto de entrada |

**Pipeline de reconhecimento por frame:**
1. `MTCNN` detecta bounding boxes de rostos a partir de uma imagem PIL
2. `extrair_face_manualmente()` recorta, redimensiona para 160×160 e normaliza para `[-1, 1]`
3. `InceptionResnetV1` (pré-treinado em VGGFace2) gera embedding de 512 floats
4. Distância L2 mínima entre o embedding capturado e **todos os embeddings** da pessoa — threshold `< 0.7` = reconhecido

**Database schema (`alunos.db`):**
- `alunos` — CPF (PK), nome, curso, ultimo_acesso, embedding (BLOB legado, mantido para compatibilidade)
- `embeddings` — id, CPF (FK), embedding (BLOB, 512×float32 = 2048 bytes) — múltiplas entradas por aluno
- `registros` — id, CPF (FK), nome, curso, timestamp, foto_path

**Funções públicas de `db.py`:**
- `buscar_aluno(cpf)` → `(nome, curso, ultimo_acesso)` ou `None`
- `salvar_aluno(cpf, nome, curso, embeddings)` — insere/substitui aluno e seus embeddings
- `carregar_alunos()` → `[(cpf, nome, curso, [embeddings])]`
- `ja_registrado_recentemente(cpf, horas)` → `bool`
- `salvar_registro(cpf, nome, curso, timestamp_str, foto_path)`

**Constantes configuráveis (`face.py`):**
- `THRESHOLD_SIMILARIDADE = 0.7` — distância L2 máxima para reconhecimento
- `N_EMBEDDINGS_POR_ALUNO = 5` — capturas por aluno no cadastro
- `TEMPO_ENTRE_REGISTROS_SEGUNDOS = 30` — cooldown em memória entre reconhecimentos

**Constante configurável em runtime (`main.py`):**
- `TEMPO_MINIMO_ENTRE_REGISTROS_HORAS = 24` — horas mínimas entre registros persistidos no banco
