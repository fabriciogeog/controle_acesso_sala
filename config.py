import json
import os

_FILE = 'config.json'

_DEFAULTS = {
    'threshold_similaridade': 0.7,
    'n_embeddings_por_aluno': 5,
    'tempo_entre_registros_segundos': 30,
    'tempo_minimo_entre_registros_horas': 24,
    'anti_spoofing_ativo': True,
}

_data = None


def _ensure_loaded():
    global _data
    if _data is not None:
        return
    if os.path.exists(_FILE):
        with open(_FILE, encoding='utf-8') as f:
            _data = {**_DEFAULTS, **json.load(f)}
    else:
        _data = dict(_DEFAULTS)
        _salvar()


def _salvar():
    with open(_FILE, 'w', encoding='utf-8') as f:
        json.dump(_data, f, indent=2, ensure_ascii=False)


def get(chave):
    _ensure_loaded()
    return _data.get(chave, _DEFAULTS.get(chave))


def set(chave, valor):
    _ensure_loaded()
    _data[chave] = valor
    _salvar()


def tudo():
    _ensure_loaded()
    return dict(_data)
