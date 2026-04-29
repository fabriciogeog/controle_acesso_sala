import cv2
import torch
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import db
import config
import minifasnet

# =============================================
# CONFIGURAÇÕES
# =============================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FOTOS_DIR = 'fotos_registros'
FOTOS_DESCONHECIDOS_DIR = 'fotos_desconhecidos'
COOLDOWN_DESCONHECIDO_SEGUNDOS = 60

os.makedirs(FOTOS_DIR, exist_ok=True)
os.makedirs(FOTOS_DESCONHECIDOS_DIR, exist_ok=True)

# =============================================
# MODELOS
# =============================================
mtcnn = MTCNN(keep_all=False, device=DEVICE, selection_method='largest')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Anti-spoofing passivo: MiniFASNetV2 avalia cada frame (~0.43M params, 80x80)
_N_FRAMES_REAL       = 3   # frames consecutivos acima do threshold para confirmar
_TIMEOUT_LIVENESS_SEG = 10  # descarta pending após N segundos sem confirmação


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

    n_capturas = config.get('n_embeddings_por_aluno')
    print(f"\n📸 Posicione o rosto de '{nome}' frente à câmera.")
    print(f"   Serão feitas {n_capturas} capturas. Varie levemente o ângulo entre elas.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Câmera não disponível.")
        return

    cv2.destroyAllWindows()
    time.sleep(0.2)
    cv2.namedWindow('Cadastro', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cadastro', 640, 480)

    embeddings_coletados = []

    while len(embeddings_coletados) < n_capturas:
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

        progresso = f"Captura {len(embeddings_coletados)}/{n_capturas}"
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
            print(f"   📷 Captura {len(embeddings_coletados)}/{n_capturas} registrada.")

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'✅ {len(embeddings_coletados)}/{n_capturas}',
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Cadastro', frame)
            time.sleep(0.8)
            continue

        cv2.imshow('Cadastro', frame)

    cap.release()
    cv2.destroyAllWindows()

    db.salvar_aluno(cpf, nome, curso, embeddings_coletados)
    print(f"✅ {nome} cadastrado com {n_capturas} embeddings!")


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

    if config.get('anti_spoofing_ativo'):
        minifasnet.carregar_modelo()  # aquece o modelo antes de abrir a câmera

    cv2.destroyAllWindows()
    time.sleep(0.2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Câmera não disponível.")
        return

    cv2.namedWindow('🔐 Acesso à Sala', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('🔐 Acesso à Sala', 800, 600)

    threshold     = config.get('threshold_similaridade')
    cooldown_tela = config.get('tempo_entre_registros_segundos')
    anti_spoofing = config.get('anti_spoofing_ativo')

    last_recognized    = {}
    last_unknown_saved = None
    # pending: cpf -> {'since': datetime, 'frames_reais': int}
    pending = {}

    modo = "Anti-spoofing: ON 👁️" if anti_spoofing else "Anti-spoofing: OFF ⚠️"
    print(f"\n🎥 {modo} | Registros a cada {horas_minimas}h. Pressione 'Q' para sair.")

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

        # HUD: modo de segurança no canto superior esquerdo
        cor_hud = (0, 200, 0) if anti_spoofing else (0, 165, 255)
        cv2.putText(frame, modo, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_hud, 2)

        if boxes is not None:
            # Fase 1: coleta de faces válidas do frame
            rostos = []
            for i, box in enumerate(boxes):
                if probs[i] < 0.9:
                    continue
                try:
                    box_clean = np.array(box[:4]).astype(int)
                except Exception:
                    continue
                face_tensor = extrair_face_manualmente(img_pil, box_clean)
                if face_tensor is not None:
                    rostos.append((box_clean, face_tensor))

            # Fase 2: único forward pass para todos os rostos do frame
            rostos_com_emb = []
            if rostos:
                try:
                    batch = torch.stack([ft for _, ft in rostos])
                    with torch.no_grad():
                        embeddings_batch = resnet(batch).cpu().numpy()  # (N, 512)
                    rostos_com_emb = [
                        (box_clean, embeddings_batch[idx].astype(np.float32))
                        for idx, (box_clean, _) in enumerate(rostos)
                    ]
                except Exception as e:
                    print(f"❌ Erro na inferência em batch: {e}")

            # Fase 3: matching e controle de acesso por rosto
            for box_clean, embedding in rostos_com_emb:
                x_min, y_min, x_max, y_max = box_clean

                # Menor distância entre todos os embeddings de cada pessoa
                melhor_distancia = float('inf')
                melhor_pessoa = None
                for cpf, nome, curso, person_embeddings in known_people:
                    dist = min(np.linalg.norm(embedding - e) for e in person_embeddings)
                    if dist < melhor_distancia:
                        melhor_distancia = dist
                        melhor_pessoa = (cpf, nome, curso)

                if melhor_distancia < threshold and melhor_pessoa:
                    cpf, nome, curso = melhor_pessoa
                    confirmado = False

                    if anti_spoofing:
                        if cpf not in pending:
                            pending[cpf] = {'since': current_time, 'frames_reais': 0}

                        estado  = pending[cpf]
                        elapsed = (current_time - estado['since']).total_seconds()
                        score   = minifasnet.prever_liveness(frame, box_clean)

                        if score >= minifasnet.THRESHOLD:
                            estado['frames_reais'] += 1
                            if estado['frames_reais'] >= _N_FRAMES_REAL:
                                confirmado = True
                                del pending[cpf]
                        else:
                            estado['frames_reais'] = 0  # reinicia contagem ao detectar spoof

                        if not confirmado:
                            if elapsed > _TIMEOUT_LIVENESS_SEG:
                                del pending[cpf]
                            elif score < minifasnet.THRESHOLD:
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 200), 2)
                                cv2.putText(frame, f"{nome} - Spoof!",
                                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
                            else:
                                pct = int(score * 100)
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                                cv2.putText(frame, f"Verificando... {pct}%",
                                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    else:
                        confirmado = True

                    if confirmado:
                        if cpf not in last_recognized or \
                                (current_time - last_recognized[cpf]).total_seconds() > cooldown_tela:
                            registrar_entrada(cpf, nome, curso, frame, horas_minimas)
                            last_recognized[cpf] = current_time
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, nome, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(frame, "Desconhecido", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if last_unknown_saved is None or \
                            (current_time - last_unknown_saved).total_seconds() > COOLDOWN_DESCONHECIDO_SEGUNDOS:
                        ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        foto_path = os.path.join(
                            FOTOS_DESCONHECIDOS_DIR,
                            f"desconhecido_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        )
                        cv2.imwrite(foto_path, frame)
                        db.salvar_tentativa_desconhecida(foto_path, ts_str)
                        last_unknown_saved = current_time
                        print(f"⚠️  Rosto desconhecido registrado às {current_time.strftime('%H:%M:%S')}")

        cv2.imshow('🔐 Acesso à Sala', frame)

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.2)
    print("✅ Reconhecimento encerrado.")
