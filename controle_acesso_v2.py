import cv2
import torch
import numpy as np
import sqlite3
import os
import time
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# =============================================
# CONFIGURAÇÕES GLOBAIS
# =============================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATABASE = 'alunos.db'
FOTOS_DIR = 'fotos_registros'
TEMPO_ENTRE_REGISTROS_SEGUNDOS = 30  # Evita múltiplos registros em segundos
TEMPO_MINIMO_ENTRE_REGISTROS_HORAS = 24  # Horas mínimas entre registros no banco
THRESHOLD_SIMILARIDADE = 0.7         # Distância L2 máxima para reconhecer um rosto
N_EMBEDDINGS_POR_ALUNO = 5           # Capturas por aluno no cadastro

os.makedirs(FOTOS_DIR, exist_ok=True)

# =============================================
# INICIALIZAÇÃO DOS MODELOS
# =============================================
mtcnn = MTCNN(keep_all=False, device=DEVICE, selection_method='largest')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# =============================================
# INICIALIZAR BANCO DE DADOS
# =============================================
def init_db():
    """Cria as tabelas se não existirem."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alunos (
                cpf TEXT PRIMARY KEY,
                nome TEXT NOT NULL,
                curso TEXT NOT NULL,
                ultimo_acesso TEXT,
                embedding BLOB NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpf TEXT,
                nome TEXT,
                curso TEXT,
                timestamp TEXT,
                foto_path TEXT,
                FOREIGN KEY (cpf) REFERENCES alunos (cpf)
            )
        ''')

# =============================================
# ATUALIZAR BANCO: Adicionar colunas novas
# =============================================
def atualizar_banco():
    """Cria estruturas novas e migra dados existentes se necessário."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Adiciona 'ultimo_acesso' se não existir
        try:
            cursor.execute('ALTER TABLE alunos ADD COLUMN ultimo_acesso TEXT')
            print("✅ Coluna 'ultimo_acesso' adicionada ao banco de dados.")
        except sqlite3.OperationalError:
            pass  # Já existe

        # Cria tabela de múltiplos embeddings por aluno
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpf TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (cpf) REFERENCES alunos (cpf)
            )
        ''')

        # Migra embeddings únicos existentes em 'alunos' que ainda não foram migrados
        cursor.execute('SELECT cpf, embedding FROM alunos WHERE embedding IS NOT NULL')
        alunos_existentes = cursor.fetchall()
        migrados = 0
        for cpf, emb_blob in alunos_existentes:
            cursor.execute('SELECT COUNT(*) FROM embeddings WHERE cpf = ?', (cpf,))
            if cursor.fetchone()[0] == 0:
                cursor.execute('INSERT INTO embeddings (cpf, embedding) VALUES (?, ?)', (cpf, emb_blob))
                migrados += 1
        if migrados > 0:
            print(f"✅ {migrados} embedding(s) migrado(s) para a nova tabela.")

# =============================================
# EXTRAÇÃO MANUAL DE FACE
# =============================================
def extrair_face_manualmente(img_pil, box):
    img_rgb = np.array(img_pil)
    x_min, y_min, x_max, y_max = map(int, box)

    h, w = img_rgb.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None

    face_crop = img_rgb[y_min:y_max, x_min:x_max]
    face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float().to(DEVICE)
    face_tensor = (face_tensor - 127.5) / 128.0

    return face_tensor

# =============================================
# VERIFICAR SE JÁ REGISTROU RECENTEMENTE
# =============================================
def ja_registrado_recentemente(cpf, horas_minimas):
    """Retorna True se já registrou nas últimas `horas_minimas` horas."""
    intervalo = f'-{int(horas_minimas)} hours'
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM registros WHERE cpf = ? AND timestamp > datetime('now', ?)",
            (cpf, intervalo)
        )
        return cursor.fetchone() is not None

# =============================================
# ATUALIZAR ÚLTIMO ACESSO
# =============================================
def atualizar_ultimo_acesso(cpf, timestamp_str):
    """Atualiza o campo 'ultimo_acesso' na tabela 'alunos'."""
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            'UPDATE alunos SET ultimo_acesso = ? WHERE cpf = ?',
            (timestamp_str, cpf)
        )

# =============================================
# OBTER ÚLTIMO ACESSO
# =============================================
def obter_ultimo_acesso(cpf):
    """Retorna o último acesso ou 'Nunca'."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT ultimo_acesso FROM alunos WHERE cpf = ?', (cpf,))
        row = cursor.fetchone()
    return row[0] if row and row[0] else 'Nunca'

# =============================================
# CADASTRAR ALUNO
# =============================================
def cadastrar_aluno():
    cpf = input("\n➡️ CPF (apenas números): ").strip()
    nome = input("➡️ Nome completo: ").strip()
    curso = input("➡️ Curso: ").strip()

    if not all([cpf, nome, curso]):
        print("❌ Todos os campos são obrigatórios!")
        return

    # Verifica se já existe
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT nome, curso, ultimo_acesso FROM alunos WHERE cpf = ?', (cpf,))
        row = cursor.fetchone()

    if row:
        ultimo = row[2] if row[2] else 'Nunca'
        print(f"\n⚠️  CPF já cadastrado:")
        print(f"   Nome: {row[0]}")
        print(f"   Curso: {row[1]}")
        print(f"   Último acesso: {ultimo}")
        if input("\n🔁 Atualizar dados? (s/n): ").strip().lower() != 's':
            return

    print(f"\n📸 Posicione o rosto de '{nome}' frente à câmera.")
    print(f"   Serão feitas {N_EMBEDDINGS_POR_ALUNO} capturas. Varie levemente o ângulo entre elas.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Câmera não disponível.")
        return

    cv2.destroyAllWindows()
    time.sleep(0.2)
    cv2.namedWindow('Cadastro', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cadastro', 640, 480)

    embeddings_coletados = []
    ultimo_frame_capturado = None

    while len(embeddings_coletados) < N_EMBEDDINGS_POR_ALUNO:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("❌ Cadastro cancelado.")
            cap.release()
            cv2.destroyAllWindows()
            return

        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)

        boxes, probs = mtcnn.detect(img_pil)

        progresso = f"Captura {len(embeddings_coletados)}/{N_EMBEDDINGS_POR_ALUNO}"
        cv2.putText(frame, progresso, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if boxes is not None and len(boxes) > 0 and probs[0] > 0.9:
            try:
                box_clean = np.array(boxes[0][:4]).astype(int)
                x_min, y_min, x_max, y_max = box_clean
            except Exception:
                cv2.imshow('Cadastro', frame)
                continue

            face_tensor = extrair_face_manualmente(img_pil, box_clean)
            if face_tensor is None:
                cv2.imshow('Cadastro', frame)
                continue

            try:
                with torch.no_grad():
                    emb = resnet(face_tensor.unsqueeze(0)).cpu()
                embedding = emb.numpy().flatten().astype(np.float32)
                if embedding.shape != (512,):
                    cv2.imshow('Cadastro', frame)
                    continue
            except Exception as e:
                print(f"❌ Erro ao gerar embedding: {e}")
                cv2.imshow('Cadastro', frame)
                continue

            embeddings_coletados.append(embedding)
            ultimo_frame_capturado = frame.copy()
            print(f"   📷 Captura {len(embeddings_coletados)}/{N_EMBEDDINGS_POR_ALUNO} registrada.")

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'✅ {len(embeddings_coletados)}/{N_EMBEDDINGS_POR_ALUNO}',
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Cadastro', frame)
            time.sleep(0.8)  # Pausa para variar o ângulo entre capturas
            continue

        cv2.imshow('Cadastro', frame)

    cap.release()
    cv2.destroyAllWindows()

    # Salvar aluno e todos os embeddings
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO alunos (cpf, nome, curso, ultimo_acesso, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', (cpf, nome, curso, None, embeddings_coletados[0].tobytes()))

        # Remove embeddings antigos e insere os novos
        cursor.execute('DELETE FROM embeddings WHERE cpf = ?', (cpf,))
        for emb in embeddings_coletados:
            cursor.execute('INSERT INTO embeddings (cpf, embedding) VALUES (?, ?)',
                           (cpf, emb.tobytes()))

    print(f"✅ {nome} cadastrado com {N_EMBEDDINGS_POR_ALUNO} embeddings!")

# =============================================
# CARREGAR ALUNOS
# =============================================
def carregar_alunos():
    """Retorna lista de (cpf, nome, curso, [embeddings]) carregada da tabela embeddings."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.cpf, a.nome, a.curso, e.embedding
            FROM alunos a
            JOIN embeddings e ON a.cpf = e.cpf
            ORDER BY a.cpf
        ''')
        rows = cursor.fetchall()

    # Agrupa embeddings por aluno
    alunos = {}
    for cpf, nome, curso, emb_blob in rows:
        try:
            emb_array = np.frombuffer(emb_blob, dtype=np.float32)
            if emb_array.shape != (512,):
                continue
            if cpf not in alunos:
                alunos[cpf] = {'nome': nome, 'curso': curso, 'embeddings': []}
            alunos[cpf]['embeddings'].append(emb_array)
        except Exception as e:
            print(f"❌ Falha ao carregar embedding de {cpf}: {e}")

    known_people = [
        (cpf, dados['nome'], dados['curso'], dados['embeddings'])
        for cpf, dados in alunos.items()
    ]

    total_embs = sum(len(p[3]) for p in known_people)
    print(f"✅ {len(known_people)} aluno(s) carregado(s) | {total_embs} embedding(s) no total.")
    return known_people

# =============================================
# REGISTRAR ENTRADA
# =============================================
def registrar_entrada(cpf, nome, curso, frame):
    """Registra entrada apenas se não foi registrada nas últimas X horas."""
    global TEMPO_MINIMO_ENTRE_REGISTROS_HORAS

    if ja_registrado_recentemente(cpf, TEMPO_MINIMO_ENTRE_REGISTROS_HORAS):
        print(f"ℹ️  {nome} já registrado nas últimas {TEMPO_MINIMO_ENTRE_REGISTROS_HORAS}h. Ignorado.")
        return False

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    foto_path = os.path.join(FOTOS_DIR, f"{cpf}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")

    frame_copy = frame.copy()
    cv2.putText(frame_copy, timestamp_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(foto_path, frame_copy)

    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            'INSERT INTO registros (cpf, nome, curso, timestamp, foto_path) VALUES (?, ?, ?, ?, ?)',
            (cpf, nome, curso, timestamp_str, foto_path)
        )

    atualizar_ultimo_acesso(cpf, timestamp_str)
    print(f"✅ {nome} entrou às {timestamp_str}")
    return True

# =============================================
# RECONHECIMENTO EM TEMPO REAL
# =============================================
def reconhecer_e_registrar():
    known_people = carregar_alunos()
    if not known_people:
        print("⚠️ Nenhum aluno cadastrado.")
        return

    cv2.destroyAllWindows()
    time.sleep(0.2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Câmera não disponível.")
        return

    cv2.namedWindow('🔐 Acesso à Sala', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('🔐 Acesso à Sala', 800, 600)

    last_recognized = {}
    print(f"\n🎥 Registros a cada {TEMPO_MINIMO_ENTRE_REGISTROS_HORAS}h. Pressione 'Q' para sair.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ret, frame = cap.read()
        if not ret:
            print("❌ Falha ao ler frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)

        boxes, probs = mtcnn.detect(img_pil)

        current_time = datetime.now()

        if boxes is not None:
            for i, box in enumerate(boxes):
                if probs[i] < 0.9:
                    continue

                try:
                    box_clean = np.array(box[:4]).astype(int)
                    x_min, y_min, x_max, y_max = box_clean
                except Exception:
                    continue

                face_tensor = extrair_face_manualmente(img_pil, box_clean)
                if face_tensor is None:
                    continue

                try:
                    with torch.no_grad():
                        emb = resnet(face_tensor.unsqueeze(0)).cpu()
                    embedding = emb.numpy().flatten().astype(np.float32)
                except Exception:
                    continue

                # Para cada pessoa, calcula a menor distância entre todos os seus embeddings
                melhor_distancia = float('inf')
                melhor_pessoa = None
                for cpf, nome, curso, person_embeddings in known_people:
                    dist = min(np.linalg.norm(embedding - e) for e in person_embeddings)
                    if dist < melhor_distancia:
                        melhor_distancia = dist
                        melhor_pessoa = (cpf, nome, curso)

                if melhor_distancia < THRESHOLD_SIMILARIDADE and melhor_pessoa:
                    cpf, nome, curso = melhor_pessoa

                    if cpf not in last_recognized or (current_time - last_recognized[cpf]).seconds > TEMPO_ENTRE_REGISTROS_SEGUNDOS:
                        registrar_entrada(cpf, nome, curso, frame)
                        last_recognized[cpf] = current_time

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, nome, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(frame, "Desconhecido", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('🔐 Acesso à Sala', frame)

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.2)
    print("✅ Reconhecimento encerrado.")

# =============================================
# CONFIGURAR TEMPO ENTRE REGISTROS
# =============================================
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

# =============================================
# MENU PRINCIPAL
# =============================================
def menu():
    init_db()
    atualizar_banco()  # ✅ Garante que colunas novas existam
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
            cadastrar_aluno()
        elif opcao == '2':
            reconhecer_e_registrar()
        elif opcao == '3':
            configurar_tempo()
        elif opcao == '4':
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida.")

# =============================================
# EXECUÇÃO
# =============================================
if __name__ == "__main__":
    menu()