import db
import face

TEMPO_MINIMO_ENTRE_REGISTROS_HORAS = 24  # Horas mínimas entre registros no banco


def configurar_tempo():
    global TEMPO_MINIMO_ENTRE_REGISTROS_HORAS
    print(f"\n⏱️  Tempo atual: {TEMPO_MINIMO_ENTRE_REGISTROS_HORAS} hora(s)")
    try:
        novo = int(input("Novo tempo mínimo entre registros (horas): "))
        if novo >= 0:
            TEMPO_MINIMO_ENTRE_REGISTROS_HORAS = novo
            print(f"✅ Atualizado para {novo} hora(s).")
        else:
            print("❌ Use um valor >= 0.")
    except ValueError:
        print("❌ Digite um número válido.")


def menu():
    db.init_db()
    db.atualizar_banco()
    while True:
        print("\n" + "="*60)
        print("           🔹 SISTEMA DE ACESSO COM RECONHECIMENTO FACIAL")
        print("="*60)
        print(f"🕒 Tempo entre registros: {TEMPO_MINIMO_ENTRE_REGISTROS_HORAS}h")
        print("1️⃣  Cadastrar novo aluno")
        print("2️⃣  Iniciar reconhecimento")
        print("3️⃣  Configurar tempo entre registros")
        print("4️⃣  Sair")
        print("="*60)
        opcao = input("👉 Escolha: ").strip()

        if opcao == '1':
            face.cadastrar_aluno()
        elif opcao == '2':
            face.reconhecer_e_registrar(TEMPO_MINIMO_ENTRE_REGISTROS_HORAS)
        elif opcao == '3':
            configurar_tempo()
        elif opcao == '4':
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida.")


if __name__ == "__main__":
    menu()
