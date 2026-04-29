import csv
from datetime import datetime

import config
import db
import face


# =============================================
# GESTÃO DE ALUNOS
# =============================================
def menu_gestao_alunos():
    while True:
        alunos = db.listar_alunos()
        print("\n" + "="*60)
        print("           👥 GESTÃO DE ALUNOS")
        print("="*60)
        print(f"  Total: {len(alunos)} aluno(s) cadastrado(s)")
        print("1️⃣  Listar alunos")
        print("2️⃣  Remover aluno")
        print("3️⃣  Voltar")
        print("="*60)
        opcao = input("👉 Escolha: ").strip()

        if opcao == '1':
            _listar_alunos_tela(alunos)
        elif opcao == '2':
            _remover_aluno_menu()
        elif opcao == '3':
            break
        else:
            print("❌ Opção inválida.")


def _listar_alunos_tela(alunos):
    if not alunos:
        print("\n⚠️  Nenhum aluno cadastrado.")
        return
    print(f"\n{'CPF':<15} {'Nome':<28} {'Curso':<20} {'Último Acesso':<21} Embs")
    print("-"*93)
    for cpf, nome, curso, ultimo, n_embs in alunos:
        ultimo = ultimo or 'Nunca'
        print(f"{cpf:<15} {nome:<28} {curso:<20} {ultimo:<21} {n_embs}")


def _remover_aluno_menu():
    cpf = input("\n➡️  CPF do aluno a remover: ").strip()
    row = db.buscar_aluno(cpf)
    if not row:
        print("❌ Aluno não encontrado.")
        return
    nome, curso, ultimo = row
    print(f"\n⚠️  Aluno encontrado: {nome} | {curso}")
    confirmacao = input(f"🗑️  Confirmar remoção de '{nome}'? (s/n): ").strip().lower()
    if confirmacao == 's':
        db.remover_aluno(cpf)
        print(f"✅ {nome} removido com sucesso.")
    else:
        print("❌ Remoção cancelada.")


# =============================================
# RELATÓRIOS
# =============================================
def menu_relatorios():
    while True:
        print("\n" + "="*60)
        print("           📊 RELATÓRIOS DE ACESSO")
        print("="*60)
        print("1️⃣  Registros de acesso")
        print("2️⃣  Frequência por aluno")
        print("3️⃣  Tentativas não reconhecidas")
        print("4️⃣  Exportar CSV")
        print("5️⃣  Voltar")
        print("="*60)
        opcao = input("👉 Escolha: ").strip()

        if opcao == '1':
            _menu_registros()
        elif opcao == '2':
            _mostrar_frequencia()
        elif opcao == '3':
            _mostrar_tentativas()
        elif opcao == '4':
            _exportar_csv_menu()
        elif opcao == '5':
            break
        else:
            print("❌ Opção inválida.")


def _menu_registros():
    print("\n  Filtros (Enter para ignorar):")
    cpf = input("  CPF: ").strip() or None
    data_inicio = input("  Data início (AAAA-MM-DD): ").strip() or None
    data_fim = input("  Data fim   (AAAA-MM-DD): ").strip() or None
    if data_fim:
        data_fim += ' 23:59:59'

    registros = db.listar_registros(cpf, data_inicio, data_fim)
    if not registros:
        print("\n⚠️  Nenhum registro encontrado.")
        return

    print(f"\n  {len(registros)} registro(s):\n")
    print(f"{'CPF':<15} {'Nome':<25} {'Curso':<20} {'Timestamp'}")
    print("-"*82)
    for r_cpf, nome, curso, ts, _ in registros:
        print(f"{r_cpf:<15} {nome:<25} {curso:<20} {ts}")


def _mostrar_frequencia():
    dados = db.relatorio_frequencia()
    if not dados:
        print("\n⚠️  Nenhum dado encontrado.")
        return
    print(f"\n{'Nome':<28} {'Curso':<20} {'Acessos':>8}  {'Primeiro':<20} {'Último'}")
    print("-"*95)
    for _, nome, curso, total, primeiro, ultimo in dados:
        primeiro = primeiro or '-'
        ultimo = ultimo or '-'
        print(f"{nome:<28} {curso:<20} {total:>8}  {primeiro:<20} {ultimo}")


def _mostrar_tentativas():
    print("\n  Filtros (Enter para ignorar):")
    data_inicio = input("  Data início (AAAA-MM-DD): ").strip() or None
    data_fim = input("  Data fim   (AAAA-MM-DD): ").strip() or None
    if data_fim:
        data_fim += ' 23:59:59'

    tentativas = db.listar_tentativas(data_inicio, data_fim)
    if not tentativas:
        print("\n⚠️  Nenhuma tentativa registrada.")
        return
    print(f"\n  {len(tentativas)} tentativa(s):\n")
    print(f"{'ID':>5}  {'Timestamp':<22} Foto")
    print("-"*70)
    for t_id, ts, foto in tentativas:
        print(f"{t_id:>5}  {ts:<22} {foto}")


def _exportar_csv_menu():
    print("\n  O que exportar?")
    print("  1 - Registros de acesso")
    print("  2 - Frequência por aluno")
    print("  3 - Tentativas não reconhecidas")
    opcao = input("  Escolha: ").strip()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if opcao == '1':
        dados = db.listar_registros()
        campos = ['cpf', 'nome', 'curso', 'timestamp', 'foto_path']
        arquivo = f'registros_{ts}.csv'
    elif opcao == '2':
        dados = db.relatorio_frequencia()
        campos = ['cpf', 'nome', 'curso', 'total_acessos', 'primeiro_acesso', 'ultimo_acesso']
        arquivo = f'frequencia_{ts}.csv'
    elif opcao == '3':
        dados = db.listar_tentativas()
        campos = ['id', 'timestamp', 'foto_path']
        arquivo = f'tentativas_{ts}.csv'
    else:
        print("❌ Opção inválida.")
        return

    if not dados:
        print("⚠️  Nenhum dado para exportar.")
        return

    with open(arquivo, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(campos)
        writer.writerows(dados)

    print(f"✅ Exportado: {arquivo} ({len(dados)} linha(s))")


# =============================================
# CONFIGURAÇÕES
# =============================================
def menu_configuracoes():
    while True:
        cfg = config.tudo()
        print("\n" + "="*60)
        print("           ⚙️  CONFIGURAÇÕES")
        print("="*60)
        status_as = "✅ ON" if cfg['anti_spoofing_ativo'] else "❌ OFF"
        print(f"1️⃣  Tempo entre registros no banco:  {cfg['tempo_minimo_entre_registros_horas']}h")
        print(f"2️⃣  Threshold de similaridade:       {cfg['threshold_similaridade']}")
        print(f"3️⃣  Capturas por cadastro:            {cfg['n_embeddings_por_aluno']}")
        print(f"4️⃣  Cooldown em tela (segundos):      {cfg['tempo_entre_registros_segundos']}")
        print(f"5️⃣  Anti-spoofing (blink detection):  {status_as}")
        print("6️⃣  Voltar")
        print("="*60)
        opcao = input("👉 Escolha: ").strip()

        if opcao == '1':
            _editar_config('tempo_minimo_entre_registros_horas', 'Horas entre registros', int, lambda v: v >= 0)
        elif opcao == '2':
            _editar_config('threshold_similaridade', 'Threshold (0.1–1.5)', float, lambda v: 0.1 <= v <= 1.5)
        elif opcao == '3':
            _editar_config('n_embeddings_por_aluno', 'Capturas por cadastro (1–20)', int, lambda v: 1 <= v <= 20)
        elif opcao == '4':
            _editar_config('tempo_entre_registros_segundos', 'Cooldown em tela (segundos)', int, lambda v: v >= 0)
        elif opcao == '5':
            novo = not cfg['anti_spoofing_ativo']
            config.set('anti_spoofing_ativo', novo)
            print(f"✅ Anti-spoofing {'ativado' if novo else 'desativado'}.")
        elif opcao == '6':
            break
        else:
            print("❌ Opção inválida.")


def _editar_config(chave, descricao, tipo, validar):
    print(f"\n  Valor atual: {config.get(chave)}")
    try:
        novo = tipo(input(f"  Novo valor para '{descricao}': ").strip())
        if validar(novo):
            config.set(chave, novo)
            print(f"✅ Atualizado para {novo}.")
        else:
            print("❌ Valor fora do intervalo permitido.")
    except ValueError:
        print("❌ Formato inválido.")


# =============================================
# MENU PRINCIPAL
# =============================================
def menu():
    db.init_db()
    db.atualizar_banco()
    while True:
        print("\n" + "="*60)
        print("    🔹 SISTEMA DE ACESSO COM RECONHECIMENTO FACIAL")
        print("="*60)
        print("1️⃣  Cadastrar novo aluno")
        print("2️⃣  Iniciar reconhecimento")
        print("3️⃣  Gerenciar alunos")
        print("4️⃣  Relatórios de acesso")
        print("5️⃣  Configurações")
        print("6️⃣  Sair")
        print("="*60)
        opcao = input("👉 Escolha: ").strip()

        if opcao == '1':
            face.cadastrar_aluno()
        elif opcao == '2':
            horas = config.get('tempo_minimo_entre_registros_horas')
            face.reconhecer_e_registrar(horas)
        elif opcao == '3':
            menu_gestao_alunos()
        elif opcao == '4':
            menu_relatorios()
        elif opcao == '5':
            menu_configuracoes()
        elif opcao == '6':
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida.")


if __name__ == "__main__":
    menu()
