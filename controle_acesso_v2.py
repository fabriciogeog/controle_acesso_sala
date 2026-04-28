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

os.makedirs(FOTOS_DIR, exist_ok=True)

# =============================================
# INICIALIZAÇÃO DOS MODELOS
# =============================================
mtcnn = MTCNN(keep_all=False, device=DEVICE, selection_method='largest')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# =============================================
# VARIÁVEL GLOBAL: Tempo mínimo entre registros (em horas)
# =============================================
TEMPO_MINIMO_ENTRE_REGISTROS_HORAS = 24  # Padrão: 24 horas

# =============================================
# INICIALIZAR BANCO DE DADOS
# =============================================
def init_db():
    """Cria as tabelas se não existirem."""
    conn = sqlite3.connect(DATABASE)
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
    conn.commit()
    conn.close()

# =============================================
# ATUALIZAR BANCO: Adicionar colunas novas
# =============================================
def atualizar_banco():
    """Adiciona colunas novas (como 'ultimo_acesso') se não existirem."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Adiciona 'ultimo_acesso' se não existir
    try:
        cursor.execute('ALTER TABLE alunos ADD COLUMN ultimo_acesso TEXT')
        print("✅ Coluna 'ultimo_acesso' adicionada ao banco de dados.")
    except sqlite3.OperationalError:
        pass  # Já existe

    conn.commit()
    conn.close()

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
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 1 FROM registros 
        WHERE cpf = ? AND timestamp > datetime('now', '-{} hours')
    '''.format(horas_minimas), (cpf,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# =============================================
# ATUALIZAR ÚLTIMO ACESSO
# =============================================
def atualizar_ultimo_acesso(cpf, timestamp_str):
    """Atualiza o campo 'ultimo_acesso' na tabela 'alunos'."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE alunos SET ultimo_acesso = ? WHERE cpf = ?
    ''', (timestamp_str, cpf))
    conn.commit()
    conn.close()

# =============================================
# OBTER ÚLTIMO ACESSO
# =============================================
def obter_ultimo_acesso(cpf):
    """Retorna o último acesso ou 'Nunca'."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT ultimo_acesso FROM alunos WHERE cpf = ?', (cpf,))
    row = cursor.fetchone()
    conn.close()
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
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT nome, curso, ultimo_acesso FROM alunos WHERE cpf = ?', (cpf,))
    row = cursor.fetchone()
    conn.close()

    if row:
        ultimo = row[2] if row[2] else 'Nunca'
        print(f"\n⚠️  CPF já cadastrado:")
        print(f"   Nome: {row[0]}")
        print(f"   Curso: {row[1]}")
        print(f"   Último acesso: {ultimo}")
        if input("\n🔁 Atualizar dados? (s/n): ").strip().lower() != 's':
            return

    print(f"\n📸 Posicione o rosto de '{nome}' frente à câmera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Câmera não disponível.")
        return

    cv2.destroyAllWindows()
    time.sleep(0.2)
    cv2.namedWindow('Cadastro', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cadastro', 640, 480)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("❌ Cadastro cancelado.")
            break

        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)

        boxes, probs = mtcnn.detect(img_pil)

        if boxes is not None and len(boxes) > 0:
            if probs[0] > 0.9:
                try:
                    box_clean = np.array(boxes[0][:4]).astype(int)
                    x_min, y_min, x_max, y_max = box_clean
                except:
                    continue

                face_tensor = extrair_face_manualmente(img_pil, box_clean)
                if face_tensor is None:
                    continue

                try:
                    with torch.no_grad():
                        emb = resnet(face_tensor.unsqueeze(0)).cpu()
                    embedding = emb.numpy().flatten().astype(np.float32)
                    if embedding.shape != (512,):
                        continue
                    embedding_bytes = embedding.tobytes()
                except Exception as e:
                    print(f"❌ Erro ao gerar embedding: {e}")
                    continue

                # Salvar/atualizar
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alunos (cpf, nome, curso, ultimo_acesso, embedding)
                    VALUES (?, ?, ?, ?, ?)
                ''', (cpf, nome, curso, None, embedding_bytes))
                conn.commit()
                conn.close()

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, '✅ CADASTRADO', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Cadastro', frame)
                print(f"✅ {nome} cadastrado!")
                time.sleep(1.5)
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow('Cadastro', frame)

    cap.release()
    cv2.destroyAllWindows()

# =============================================
# CARREGAR ALUNOS
# =============================================
def carregar_alunos():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT cpf, nome, curso, embedding FROM alunos")
    rows = cursor.fetchall()
    conn.close()

    known_embeddings = []
    known_data = []

    for row in rows:
        try:
            emb_array = np.frombuffer(row[3], dtype=np.float32)
            if emb_array.shape == (512,):
                known_embeddings.append(emb_array)
                known_data.append(row[:3])
        except Exception as e:
            print(f"❌ Falha ao carregar embedding de {row[0]}: {e}")

    print(f"✅ {len(known_data)} aluno(s) carregado(s).")
    return known_embeddings, known_data

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

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO registros (cpf, nome, curso, timestamp, foto_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (cpf, nome, curso, timestamp_str, foto_path))
    conn.commit()
    conn.close()

    atualizar_ultimo_acesso(cpf, timestamp_str)
    print(f"✅ {nome} entrou às {timestamp_str}")
    return True

# =============================================
# RECONHECIMENTO EM TEMPO REAL
# =============================================
def reconhecer_e_registrar():
    known_embeddings, known_data = carregar_alunos()
    if len(known_embeddings) == 0:
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
                except:
                    continue

                face_tensor = extrair_face_manualmente(img_pil, box_clean)
                if face_tensor is None:
                    continue

                try:
                    with torch.no_grad():
                        emb = resnet(face_tensor.unsqueeze(0)).cpu()
                    embedding = emb.numpy().flatten().astype(np.float32)
                except:
                    continue

                distances = [np.linalg.norm(embedding - e) for e in known_embeddings]
                if not distances:
                    continue

                min_distance = min(distances)
                if min_distance < 0.7:
                    idx = np.argmin(distances)
                    cpf, nome, curso = known_data[idx]

                    if cpf not in last_recognized or (current_time - last_recognized[cpf]).seconds > TEMPO_ENTRE_REGISTROS_SEGUNDOS:
                        registrar_entrada(cpf, nome, curso, frame)
                        last_recognized[cpf] = current_time

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{nome}", (x_min, y_min - 10),
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