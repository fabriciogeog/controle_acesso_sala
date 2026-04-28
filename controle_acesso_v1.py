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
# CONFIGURAÇÕES
# =============================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATABASE = 'alunos.db'
FOTOS_DIR = 'fotos_registros'
THRESHOLD_SIMILARIDADE = 0.7
TEMPO_ENTRE_REGISTROS = 30

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
    """Cria tabelas se não existirem."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alunos (
            cpf TEXT PRIMARY KEY,
            nome TEXT NOT NULL,
            curso TEXT NOT NULL,
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
    print("✅ Banco de dados inicializado.")

# =============================================
# EXTRAÇÃO MANUAL DE FACE (ROBUSTA)
# =============================================
def extrair_face_manualmente(img_pil, box):
    """Extrai e preprocessa a face para o InceptionResnet."""
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
    face_tensor = (face_tensor - 127.5) / 128.0  # Normalização [-1, 1]

    return face_tensor

# =============================================
# CADASTRAR ALUNO
# =============================================
def cadastrar_aluno():
    """Captura rosto e cadastra aluno com embedding."""
    cpf = input("\n➡️ CPF (apenas números): ").strip()
    nome = input("➡️ Nome completo: ").strip()
    curso = input("➡️ Curso: ").strip()

    if not all([cpf, nome, curso]):
        print("❌ Todos os campos são obrigatórios!")
        return

    print(f"\n📸 Posicione o rosto de '{nome}' frente à câmera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Não foi possível acessar a câmera.")
        return

    # ✅ Garantir janela limpa
    cv2.destroyAllWindows()
    time.sleep(0.2)
    cv2.namedWindow('Cadastro', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cadastro', 640, 480)

    while True:
        # ✅ Verifica 'q' ANTES de processar novo frame
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
            i = 0  # Usa a primeira (ou maior) face
            if probs[i] > 0.9:
                try:
                    box_clean = np.array(boxes[i][:4]).astype(int)
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

                # Salvar no banco
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alunos (cpf, nome, curso, embedding)
                    VALUES (?, ?, ?, ?)
                ''', (cpf, nome, curso, embedding_bytes))
                conn.commit()
                conn.close()

                # Feedback visual
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, '✅ CADASTRADO', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"✅ {nome} cadastrado! (Embedding: 2048 bytes)")

                cv2.imshow('Cadastro', frame)
                time.sleep(1.5)
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow('Cadastro', frame)

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.2)

# =============================================
# CARREGAR ALUNOS CADASTRADOS
# =============================================
def carregar_alunos():
    """Carrega embeddings e dados dos alunos."""
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

    print(f"✅ {len(known_data)} aluno(s) carregado(s) para reconhecimento.")
    return known_embeddings, known_data

# =============================================
# REGISTRAR ENTRADA COM FOTO E TIMESTAMP
# =============================================
def registrar_entrada(cpf, nome, curso, frame):
    """Salva foto com timestamp e registro no banco."""
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

    print(f"✅ {nome} entrou às {timestamp_str}")

# =============================================
# RECONHECIMENTO EM TEMPO REAL
# =============================================
def reconhecer_e_registrar():
    """Inicia reconhecimento facial e registro de acesso."""
    known_embeddings, known_data = carregar_alunos()
    if len(known_embeddings) == 0:
        print("⚠️ Nenhum aluno cadastrado.")
        return

    # ✅ Garantir ambiente limpo
    cv2.destroyAllWindows()
    time.sleep(0.2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Câmera não disponível.")
        return

    # ✅ Janela única e controlada
    cv2.namedWindow('🔐 Acesso à Sala', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('🔐 Acesso à Sala', 800, 600)

    last_recognized = {}
    print("\n🎥 Reconhecimento iniciado. Pressione 'Q' para sair.")

    while True:
        # ✅ Verifica tecla ANTES de processar
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
                if min_distance < THRESHOLD_SIMILARIDADE:
                    idx = np.argmin(distances)
                    cpf, nome, curso = known_data[idx]

                    if cpf not in last_recognized or (current_time - last_recognized[cpf]).seconds > TEMPO_ENTRE_REGISTROS:
                        registrar_entrada(cpf, nome, curso, frame)
                        last_recognized[cpf] = current_time

                    # Feedback: reconhecido
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{nome}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Desconhecido
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(frame, "Desconhecido", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ✅ ÚNICA chamada para exibir a imagem
        cv2.imshow('🔐 Acesso à Sala', frame)

    # ✅ Libera recursos
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.2)
    print("✅ Reconhecimento encerrado.")

# =============================================
# MENU PRINCIPAL
# =============================================
def menu():
    init_db()
    while True:
        print("\n" + "="*50)
        print("     🔹 SISTEMA DE ACESSO COM RECONHECIMENTO FACIAL")
        print("="*50)
        print("1️⃣  Cadastrar novo aluno")
        print("2️⃣  Iniciar reconhecimento e registro de entrada")
        print("3️⃣  Sair")
        print("="*50)
        opcao = input("👉 Escolha uma opção: ").strip()

        if opcao == '1':
            cadastrar_aluno()
        elif opcao == '2':
            reconhecer_e_registrar()
        elif opcao == '3':
            print("👋 Encerrando o sistema. Até logo!")
            break
        else:
            print("❌ Opção inválida. Escolha 1, 2 ou 3.")

# =============================================
# EXECUÇÃO
# =============================================
if __name__ == "__main__":
    menu()
