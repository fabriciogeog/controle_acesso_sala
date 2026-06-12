"""Validação e formatação de CPF (sem dependências externas)."""

import re


def _digitos(cpf):
    return re.sub(r'\D', '', cpf or '')


def validar_cpf(cpf):
    """Retorna True se o CPF for válido (11 dígitos + dígitos verificadores corretos)."""
    nums = _digitos(cpf)
    if len(nums) != 11:
        return False
    if nums == nums[0] * 11:  # rejeita sequências como 000... ou 111...
        return False

    for i in (9, 10):
        soma = sum(int(nums[j]) * (i + 1 - j) for j in range(i))
        dv = (soma * 10) % 11
        if dv == 10:
            dv = 0
        if dv != int(nums[i]):
            return False
    return True


def formatar_cpf(cpf):
    """Formata como 000.000.000-00 quando houver 11 dígitos; senão devolve o original."""
    nums = _digitos(cpf)
    if len(nums) != 11:
        return cpf
    return f"{nums[:3]}.{nums[3:6]}.{nums[6:9]}-{nums[9:]}"
