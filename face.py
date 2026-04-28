import cv2
import torch
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import db

# =============================================
# CONFIGURAÇÕES
# =============================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FOTOS_DIR = 'fotos_registros'
THRESHOLD_SIMILARIDADE = 0.7
N_EMBEDDINGS_POR_ALUNO = 5
TEMPO_ENTRE_REGISTROS_SEGUNDOS = 30

os.makedirs(FOTOS_DIR, exist_ok=True)

# =============================================
# MODELOS
# =============================================
mtcnn = MTCNN(keep_all=False, device=DEVICE, selection_method='largest')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


# =============================================
# VISÃO
# =============================================
def extrair_face_manualmente(img_pil, box):
    """Recorta, redimensiona e normaliza a face para o InceptionResnetV1."""
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
# CADASTRO
# =============================================
def cadastrar_aluno():
    """Coleta dados do aluno, captura N embeddings e salva no banco."""
    cpf = input("\n➡️ CPF (apenas números): ").strip()
    nome = input("➡️ Nome completo: ").strip()
    curso = input("➡️ Curso: ").strip()

    if not all([cpf, nome, curso]):
        print("❌ Todos os campos são obrigatórios!")
        return

    row = db.buscar_aluno(cpf)
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
        cv2.putText(frame, progresso, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

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
            print(f"   📷 Captura {len(embeddings_coletados)}/{N_EMBEDDINGS_POR_ALUNO} registrada.")

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'✅ {len(embeddings_coletados)}/{N_EMBEDDINGS_POR_ALUNO}',
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Cadastro', frame)
            time.sleep(0.8)
            continue

        cv2.imshow('Cadastro', frame)

    cap.release()
    cv2.destroyAllWindows()

    db.salvar_aluno(cpf, nome, curso, embeddings_coletados)
    print(f"✅ {nome} cadastrado com {N_EMBEDDINGS_POR_ALUNO} embeddings!")


# =============================================
# REGISTRO DE ENTRADA
# =============================================
def registrar_entrada(cpf, nome, curso, frame, horas_minimas):
    """Salva foto e grava registro no banco, respeitando o cooldown por horas."""
    if db.ja_registrado_recentemente(cpf, horas_minimas):
        print(f"ℹ️  {nome} já registrado nas últimas {horas_minimas}h. Ignorado.")
        return False

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    foto_path = os.path.join(FOTOS_DIR, f"{cpf}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")

    frame_copy = frame.copy()
    cv2.putText(frame_copy, timestamp_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(foto_path, frame_copy)

    db.salvar_registro(cpf, nome, curso, timestamp_str, foto_path)
    print(f"✅ {nome} entrou às {timestamp_str}")
    return True


# =============================================
# RECONHECIMENTO EM TEMPO REAL
# =============================================
def reconhecer_e_registrar(horas_minimas):
    """Inicia reconhecimento facial e registra entradas respeitando o cooldown."""
    known_people = db.carregar_alunos()
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
    print(f"\n🎥 Registros a cada {horas_minimas}h. Pressione 'Q' para sair.")

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

                # Menor distância entre todos os embeddings de cada pessoa
                melhor_distancia = float('inf')
                melhor_pessoa = None
                for cpf, nome, curso, person_embeddings in known_people:
                    dist = min(np.linalg.norm(embedding - e) for e in person_embeddings)
                    if dist < melhor_distancia:
                        melhor_distancia = dist
                        melhor_pessoa = (cpf, nome, curso)

                if melhor_distancia < THRESHOLD_SIMILARIDADE and melhor_pessoa:
                    cpf, nome, curso = melhor_pessoa
                    if cpf not in last_recognized or \
                            (current_time - last_recognized[cpf]).seconds > TEMPO_ENTRE_REGISTROS_SEGUNDOS:
                        registrar_entrada(cpf, nome, curso, frame, horas_minimas)
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
