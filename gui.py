import csv
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

import config
import db
import face
import minifasnet


class _CamState(Enum):
    OFF     = "off"
    STANDBY = "standby"
    ON      = "on"


_POLL_ON_MS      = 30
_POLL_STANDBY_MS = 2000
_CAM_W, _CAM_H   = 640, 480


def _center_window(win, w, h):
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")


# =========================================================
# DIÁLOGO — PRIMEIRO ACESSO
# =========================================================
class _PrimeiroAcessoDialog(tk.Toplevel):
    """Cria o primeiro usuário administrador (exibido apenas na primeira execução)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.result = False
        self.title("Configuração Inicial")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancelar)
        _center_window(self, 400, 330)
        self._build()

    def _build(self):
        bg, fg, ef = "#1a1a2e", "#e0e0f0", "#2c3e50"

        tk.Label(self, text="Configuração Inicial", bg=bg, fg="#7f8fff",
                 font=("Helvetica", 14, "bold")).pack(pady=(20, 4))
        tk.Label(self, text="Primeira execução detectada.\nCrie a conta de administrador do sistema.",
                 bg=bg, fg="#aaaacc", font=("Helvetica", 9),
                 justify="center").pack(pady=(0, 14))

        form = tk.Frame(self, bg=bg)
        form.pack(padx=40, fill="x")

        self._vars = {}
        for lbl, key, secret in [("Usuário:", "user", False),
                                   ("Senha:", "pass", True),
                                   ("Confirmar senha:", "pass2", True)]:
            tk.Label(form, text=lbl, bg=bg, fg=fg,
                     font=("Helvetica", 10), anchor="w").pack(fill="x", pady=(6, 0))
            var = tk.StringVar()
            self._vars[key] = var
            tk.Entry(form, textvariable=var, bg=ef, fg=fg, insertbackground=fg,
                     font=("Helvetica", 11), relief="flat",
                     show="•" if secret else "").pack(fill="x", ipady=4)

        self._vars["user"].set("admin")

        self._msg_var = tk.StringVar()
        tk.Label(self, textvariable=self._msg_var, bg=bg, fg="#e74c3c",
                 font=("Helvetica", 9)).pack(pady=(8, 0))

        tk.Button(self, text="Criar conta", command=self._confirmar,
                  bg="#3498db", fg="white", font=("Helvetica", 11, "bold"),
                  relief="flat", cursor="hand2",
                  pady=6, padx=20).pack(pady=12)

    def _confirmar(self):
        user  = self._vars["user"].get().strip()
        senha = self._vars["pass"].get()
        conf  = self._vars["pass2"].get()

        if not user or " " in user:
            self._msg_var.set("Usuário inválido (sem espaços).")
            return
        if len(senha) < 4:
            self._msg_var.set("Senha deve ter ao menos 4 caracteres.")
            return
        if senha != conf:
            self._msg_var.set("As senhas não coincidem.")
            return

        db.criar_usuario(user, senha)
        self.result = True
        self.destroy()

    def _cancelar(self):
        self.destroy()  # result = False → App encerra


# =========================================================
# DIÁLOGO — LOGIN
# =========================================================
class _LoginDialog(tk.Toplevel):
    """Solicita credenciais antes de abrir a interface principal."""

    _MAX_TENTATIVAS = 3

    def __init__(self, parent):
        super().__init__(parent)
        self.result     = False
        self._tentativas = 0
        self.title("Acesso ao Sistema")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancelar)
        _center_window(self, 360, 270)
        self._build()

    def _build(self):
        bg, fg, ef = "#1a1a2e", "#e0e0f0", "#2c3e50"

        tk.Label(self, text="🔐  Acesso ao Sistema", bg=bg, fg="#7f8fff",
                 font=("Helvetica", 14, "bold")).pack(pady=(24, 18))

        form = tk.Frame(self, bg=bg)
        form.pack(padx=40, fill="x")

        for lbl, attr, secret in [("Usuário:", "_e_user", False),
                                    ("Senha:", "_e_pass", True)]:
            tk.Label(form, text=lbl, bg=bg, fg=fg,
                     font=("Helvetica", 10), anchor="w").pack(fill="x", pady=(6, 0))
            e = tk.Entry(form, bg=ef, fg=fg, insertbackground=fg,
                         font=("Helvetica", 11), relief="flat",
                         show="•" if secret else "")
            e.pack(fill="x", ipady=4)
            setattr(self, attr, e)

        self._e_pass.bind("<Return>", lambda _: self._confirmar())

        self._msg_var = tk.StringVar()
        tk.Label(self, textvariable=self._msg_var, bg=bg, fg="#e74c3c",
                 font=("Helvetica", 9)).pack(pady=(10, 0))

        tk.Button(self, text="Entrar", command=self._confirmar,
                  bg="#2ecc71", fg="white", font=("Helvetica", 11, "bold"),
                  relief="flat", cursor="hand2",
                  pady=6, padx=24).pack(pady=10)

        self._e_user.focus_set()

    def _confirmar(self):
        user  = self._e_user.get().strip()
        senha = self._e_pass.get()

        if db.verificar_usuario(user, senha):
            self.result = True
            self.destroy()
            return

        self._tentativas += 1
        self._e_pass.delete(0, "end")

        if self._tentativas >= self._MAX_TENTATIVAS:
            self._msg_var.set("Acesso bloqueado. Encerrando.")
            self.after(2000, self._cancelar)
        else:
            restantes = self._MAX_TENTATIVAS - self._tentativas
            self._msg_var.set(f"Credenciais inválidas. {restantes} tentativa(s) restante(s).")

    def _cancelar(self):
        self.destroy()  # result = False → App encerra


# =========================================================
# APLICAÇÃO PRINCIPAL
# =========================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()  # oculto até o login ser concluído

        db.init_db()
        db.atualizar_banco()
        db.init_usuarios()

        if db.count_usuarios() == 0:
            if not self._dialogo_primeiro_acesso():
                self.destroy()
                return

        if not self._dialogo_login():
            self.destroy()
            return

        # Login bem-sucedido — configura e exibe a janela principal
        self.title("Controle de Acesso — Reconhecimento Facial")
        self.resizable(False, False)

        self._cap       = None
        self._cam_state = _CamState.OFF
        self._after_id  = None
        self._photo     = None
        self._placeholder_photo = None

        self._known_people       = []
        self._last_recognized    = {}
        self._last_unknown_saved = None
        self._pending            = {}

        self._cadastro_mode = False
        self._cadastro_data = {}

        self._rel_cache = None

        self._build_ui()
        self._switch_panel("cadastro")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.deiconify()

    # =========================================================
    # LOGIN
    # =========================================================
    def _dialogo_primeiro_acesso(self):
        dlg = _PrimeiroAcessoDialog(self)
        self.wait_window(dlg)
        return dlg.result

    def _dialogo_login(self):
        dlg = _LoginDialog(self)
        self.wait_window(dlg)
        return dlg.result

    # =========================================================
    # UI BUILD
    # =========================================================
    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    def _build_left(self):
        left = tk.Frame(self, bg="#1a1a2e")
        left.grid(row=0, column=0, sticky="nsew")

        blank = Image.new("RGB", (_CAM_W, _CAM_H), (13, 13, 26))
        self._placeholder_photo = ImageTk.PhotoImage(blank)

        self._cam_label = tk.Label(left, image=self._placeholder_photo, bg="#0d0d1a")
        self._cam_label.pack(padx=8, pady=(8, 4))

        self._status_var = tk.StringVar(value="Câmera desligada")
        tk.Label(left, textvariable=self._status_var, bg="#1a1a2e", fg="#aaaacc",
                 font=("Helvetica", 10)).pack()

        ctrl = tk.Frame(left, bg="#1a1a2e")
        ctrl.pack(pady=6)
        _b = dict(width=9, font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2")
        tk.Button(ctrl, text="▶ ON",       bg="#2ecc71", fg="white",
                  command=self._cam_on,      **_b).grid(row=0, column=0, padx=4)
        tk.Button(ctrl, text="⏸ Stand-by", bg="#f39c12", fg="white",
                  command=self._cam_standby, **_b).grid(row=0, column=1, padx=4)
        tk.Button(ctrl, text="■ OFF",      bg="#e74c3c", fg="white",
                  command=self._cam_off,     **_b).grid(row=0, column=2, padx=4)

        tk.Label(left, text="Últimos registros", bg="#1a1a2e", fg="#7777aa",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 0))
        self._log_text = tk.Text(left, height=5, width=52, bg="#0d0d1a", fg="#ccccee",
                                  font=("Courier", 9), state="disabled", relief="flat")
        self._log_text.pack(padx=8, pady=(0, 8))

    def _build_right(self):
        right = tk.Frame(self, bg="#f0f0f5")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        nav = tk.Frame(right, bg="#2c3e50")
        nav.grid(row=0, column=0, sticky="ew")
        for label, key in [("Cadastro", "cadastro"), ("Alunos", "alunos"),
                             ("Relatórios", "relatorios"), ("Configurações", "config"),
                             ("Usuários", "usuarios")]:
            tk.Button(nav, text=label, command=lambda k=key: self._switch_panel(k),
                      bg="#2c3e50", fg="white", font=("Helvetica", 10, "bold"),
                      relief="flat", cursor="hand2",
                      padx=12, pady=8).pack(side="left")
        tk.Button(nav, text="Sair", command=self._on_close,
                  bg="#c0392b", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", cursor="hand2",
                  padx=12, pady=8).pack(side="right")

        container = tk.Frame(right, bg="#f0f0f5")
        container.grid(row=1, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self._panels = {}
        for pid in ("cadastro", "alunos", "relatorios", "config", "usuarios"):
            f = tk.Frame(container, bg="#f0f0f5")
            f.grid(row=0, column=0, sticky="nsew")
            self._panels[pid] = f

        self._build_panel_cadastro(self._panels["cadastro"])
        self._build_panel_alunos(self._panels["alunos"])
        self._build_panel_relatorios(self._panels["relatorios"])
        self._build_panel_config(self._panels["config"])
        self._build_panel_usuarios(self._panels["usuarios"])

    # =========================================================
    # PANEL SWITCHER
    # =========================================================
    def _switch_panel(self, key):
        self._panels[key].tkraise()
        if key == "alunos":
            self._refresh_alunos()
        elif key == "relatorios":
            self._refresh_relatorios()
        elif key == "usuarios":
            self._refresh_usuarios()

    # =========================================================
    # CAMERA CONTROL
    # =========================================================
    def _cam_on(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                messagebox.showerror("Câmera", "Não foi possível abrir a câmera.")
                self._cap = None
                return
            self._known_people = db.carregar_alunos()
            if config.get('anti_spoofing_ativo'):
                minifasnet.carregar_modelo()
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._cam_state = _CamState.ON
        self._status_var.set("Reconhecimento ativo")
        self._schedule_poll()

    def _cam_standby(self):
        if self._cap is None or not self._cap.isOpened():
            self._cam_on()
            if self._cap is None:
                return
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._cam_state = _CamState.STANDBY
        self._cadastro_mode = False
        self._status_var.set("Stand-by")
        self._schedule_poll()

    def _cam_off(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None
        self._cam_state = _CamState.OFF
        self._cadastro_mode = False
        self._status_var.set("Câmera desligada")
        self._cam_label.config(image=self._placeholder_photo)
        self._photo = None

    def _schedule_poll(self):
        ms = _POLL_ON_MS if self._cam_state == _CamState.ON else _POLL_STANDBY_MS
        self._after_id = self.after(ms, self._poll)

    def _poll(self):
        if self._cam_state == _CamState.OFF or self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            self._schedule_poll()
            return

        if self._cam_state == _CamState.STANDBY:
            display = frame.copy()
            h, w    = display.shape[:2]
            cv2.putText(display, "STAND-BY", (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (80, 80, 180), 4)
        elif self._cadastro_mode:
            display = self._process_cadastro(frame)
        else:
            display = self._process_recognition(frame)

        self._show_frame(display)
        self._schedule_poll()

    def _show_frame(self, frame_bgr):
        rgb         = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img         = Image.fromarray(rgb).resize((_CAM_W, _CAM_H))
        self._photo = ImageTk.PhotoImage(img)
        self._cam_label.config(image=self._photo)

    # =========================================================
    # RECOGNITION (per-frame, called from _poll)
    # =========================================================
    def _process_recognition(self, frame):
        display      = frame.copy()
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil      = Image.fromarray(rgb)
        boxes, probs = face.mtcnn.detect(img_pil)
        now          = datetime.now()

        threshold     = config.get('threshold_similaridade')
        cooldown_tela = config.get('tempo_entre_registros_segundos')
        horas_min     = config.get('tempo_minimo_entre_registros_horas')
        anti_spoofing = config.get('anti_spoofing_ativo')

        modo    = "Anti-spoofing: ON" if anti_spoofing else "Anti-spoofing: OFF"
        cor_hud = (0, 200, 0) if anti_spoofing else (0, 165, 255)
        cv2.putText(display, modo, (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_hud, 1)

        if boxes is None or not self._known_people:
            return display

        rostos = []
        for i, box in enumerate(boxes):
            if probs[i] < 0.9:
                continue
            try:
                bc = np.array(box[:4]).astype(int)
            except Exception:
                continue
            ft = face.extrair_face_manualmente(img_pil, bc)
            if ft is not None:
                rostos.append((bc, ft))

        if not rostos:
            return display

        try:
            batch = torch.stack([ft for _, ft in rostos])
            with torch.no_grad():
                embs = face.resnet(batch).cpu().numpy()
            rostos_emb = [(rostos[i][0], embs[i].astype(np.float32))
                          for i in range(len(rostos))]
        except Exception:
            return display

        for bc, emb in rostos_emb:
            x1, y1, x2, y2 = bc

            best_d, best_p = float('inf'), None
            for cpf, nome, curso, person_embs in self._known_people:
                d = min(np.linalg.norm(emb - e) for e in person_embs)
                if d < best_d:
                    best_d, best_p = d, (cpf, nome, curso)

            if best_d < threshold and best_p:
                cpf, nome, curso = best_p
                confirmado = False

                if anti_spoofing:
                    if cpf not in self._pending:
                        self._pending[cpf] = {'since': now, 'frames_reais': 0}
                    estado  = self._pending[cpf]
                    elapsed = (now - estado['since']).total_seconds()
                    score   = minifasnet.prever_liveness(frame, bc)

                    if score >= minifasnet.THRESHOLD:
                        estado['frames_reais'] += 1
                        if estado['frames_reais'] >= face._N_FRAMES_REAL:
                            confirmado = True
                            del self._pending[cpf]
                    else:
                        estado['frames_reais'] = max(0, estado['frames_reais'] - 1)

                    if not confirmado:
                        if elapsed > face._TIMEOUT_LIVENESS_SEG:
                            del self._pending[cpf]
                        elif score < minifasnet.THRESHOLD:
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 200), 2)
                            cv2.putText(display, f"{nome} - Spoof!",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 0, 200), 2)
                        else:
                            pct = int(score * 100)
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(display, f"Verificando... {pct}%",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 255, 255), 2)
                else:
                    confirmado = True

                if confirmado:
                    last = self._last_recognized.get(cpf)
                    if last is None or (now - last).total_seconds() > cooldown_tela:
                        registrado = face.registrar_entrada(cpf, nome, curso, frame, horas_min)
                        self._last_recognized[cpf] = now
                        self._log_acesso(nome, curso, novo=registrado)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, nome, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display, "Desconhecido", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if self._last_unknown_saved is None or \
                        (now - self._last_unknown_saved).total_seconds() \
                        > face.COOLDOWN_DESCONHECIDO_SEGUNDOS:
                    ts_str    = now.strftime("%Y-%m-%d %H:%M:%S")
                    foto_path = os.path.join(
                        face.FOTOS_DESCONHECIDOS_DIR,
                        f"desconhecido_{now.strftime('%Y%m%d_%H%M%S')}.jpg",
                    )
                    cv2.imwrite(foto_path, frame)
                    db.salvar_tentativa_desconhecida(foto_path, ts_str)
                    self._last_unknown_saved = now

        return display

    def _log_acesso(self, nome, curso, novo=True):
        ts   = datetime.now().strftime("%H:%M:%S")
        icon = "✅" if novo else "👤"
        line = f"{ts} {icon} {nome[:20]:<20} {curso[:14]}\n"
        self._log_text.config(state="normal")
        self._log_text.insert("1.0", line)
        self._log_text.config(state="disabled")

    # =========================================================
    # CADASTRO (capture driven by _poll — no extra thread)
    # =========================================================
    def _build_panel_cadastro(self, parent):
        parent.columnconfigure(0, weight=1)

        tk.Label(parent, text="Cadastrar Aluno", font=("Helvetica", 14, "bold"),
                 bg="#f0f0f5").grid(row=0, column=0, pady=(16, 8))

        form = tk.Frame(parent, bg="#f0f0f5")
        form.grid(row=1, column=0, padx=24, sticky="ew")
        form.columnconfigure(1, weight=1)

        self._cad_vars = {}
        for i, (lbl, key) in enumerate([("CPF:", "cpf"),
                                          ("Nome completo:", "nome"),
                                          ("Curso:", "curso")]):
            tk.Label(form, text=lbl, bg="#f0f0f5",
                     font=("Helvetica", 10)).grid(row=i, column=0, sticky="w", pady=6)
            var = tk.StringVar()
            self._cad_vars[key] = var
            tk.Entry(form, textvariable=var, font=("Helvetica", 10),
                     width=28).grid(row=i, column=1, sticky="ew", padx=(8, 0), pady=6)

        self._cad_progress = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._cad_progress, bg="#f0f0f5", fg="#27ae60",
                 font=("Helvetica", 10, "bold")).grid(row=2, column=0, pady=4)

        btn_row = tk.Frame(parent, bg="#f0f0f5")
        btn_row.grid(row=3, column=0, pady=8)
        self._btn_cad = tk.Button(btn_row, text="Iniciar Captura",
                                   command=self._iniciar_cadastro,
                                   bg="#3498db", fg="white",
                                   font=("Helvetica", 11, "bold"),
                                   relief="flat", cursor="hand2",
                                   pady=6, padx=16)
        self._btn_cad.pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancelar",
                  command=self._cancelar_cadastro,
                  bg="#95a5a6", fg="white",
                  font=("Helvetica", 10), relief="flat",
                  cursor="hand2", pady=6, padx=10).pack(side="left", padx=6)

    def _iniciar_cadastro(self):
        cpf   = self._cad_vars["cpf"].get().strip()
        nome  = self._cad_vars["nome"].get().strip()
        curso = self._cad_vars["curso"].get().strip()

        if not all([cpf, nome, curso]):
            messagebox.showwarning("Cadastro", "Preencha todos os campos.")
            return

        row = db.buscar_aluno(cpf)
        if row:
            if not messagebox.askyesno("Cadastro",
                    f"CPF já cadastrado: {row[0]}.\nAtualizar dados?"):
                return

        if self._cam_state == _CamState.OFF:
            self._cam_on()
            if self._cap is None:
                return
        elif self._cam_state == _CamState.STANDBY:
            self._cam_on()

        self._btn_cad.config(state="disabled")
        n_total = config.get('n_embeddings_por_aluno')
        self._cadastro_data = {
            'cpf': cpf, 'nome': nome, 'curso': curso,
            'embeddings': [], 'n_total': n_total,
            'last_capture': 0.0,
        }
        self._cadastro_mode = True
        self._cad_progress.set(f"Posicione o rosto. 0/{n_total}")

    def _cancelar_cadastro(self):
        self._cadastro_mode = False
        self._cadastro_data = {}
        self._btn_cad.config(state="normal")
        self._cad_progress.set("Captura cancelada.")

    def _process_cadastro(self, frame):
        display = frame.copy()
        cad     = self._cadastro_data
        n       = len(cad['embeddings'])
        total   = cad['n_total']

        cv2.putText(display, f"Cadastro: {n}/{total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # enforce 0.8 s gap between captures so the user can shift angle
        if time.time() - cad['last_capture'] < 0.8:
            return display

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        boxes, probs = face.mtcnn.detect(img_pil)

        if boxes is None or len(boxes) == 0 or probs[0] < 0.9:
            return display

        try:
            bc = np.array(boxes[0][:4]).astype(int)
        except Exception:
            return display

        ft = face.extrair_face_manualmente(img_pil, bc)
        if ft is None:
            return display

        try:
            with torch.no_grad():
                emb = face.resnet(ft.unsqueeze(0)).cpu().numpy().flatten().astype(np.float32)
            if emb.shape != (512,):
                return display
        except Exception:
            return display

        cad['embeddings'].append(emb)
        cad['last_capture'] = time.time()
        n = len(cad['embeddings'])
        self._cad_progress.set(f"Captura {n}/{total} registrada ✓")

        x1, y1, x2, y2 = bc
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"OK {n}/{total}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if n == total:
            db.salvar_aluno(cad['cpf'], cad['nome'], cad['curso'], cad['embeddings'])
            self._known_people = db.carregar_alunos()
            self._cadastro_mode = False
            self._btn_cad.config(state="normal")
            self._cad_progress.set(f"✅ {cad['nome']} cadastrado com {total} embeddings!")

        return display

    # =========================================================
    # ALUNOS PANEL
    # =========================================================
    def _build_panel_alunos(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        tk.Label(parent, text="Alunos Cadastrados", font=("Helvetica", 14, "bold"),
                 bg="#f0f0f5").grid(row=0, column=0, pady=(16, 4))

        cols = ("CPF", "Nome", "Curso", "Último Acesso", "Embs")
        tf   = tk.Frame(parent)
        tf.grid(row=1, column=0, sticky="nsew", padx=16, pady=4)
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        self._tree_alunos = ttk.Treeview(tf, columns=cols, show="headings", height=14)
        for col, w in zip(cols, [110, 180, 130, 140, 50]):
            self._tree_alunos.heading(col, text=col)
            self._tree_alunos.column(col, width=w, anchor="w")
        self._tree_alunos.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self._tree_alunos.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._tree_alunos.config(yscrollcommand=sb.set)

        btn_row = tk.Frame(parent, bg="#f0f0f5")
        btn_row.grid(row=2, column=0, pady=8)
        tk.Button(btn_row, text="Atualizar",
                  command=self._refresh_alunos,
                  bg="#3498db", fg="white", font=("Helvetica", 10),
                  relief="flat", cursor="hand2", padx=10).pack(side="left", padx=6)
        tk.Button(btn_row, text="Remover selecionado",
                  command=self._remover_aluno,
                  bg="#e74c3c", fg="white", font=("Helvetica", 10),
                  relief="flat", cursor="hand2", padx=10).pack(side="left", padx=6)

    def _refresh_alunos(self):
        for r in self._tree_alunos.get_children():
            self._tree_alunos.delete(r)
        for cpf, nome, curso, ultimo, n_embs in db.listar_alunos():
            self._tree_alunos.insert("", "end",
                values=(cpf, nome, curso, ultimo or "Nunca", n_embs))

    def _remover_aluno(self):
        sel = self._tree_alunos.selection()
        if not sel:
            messagebox.showinfo("Alunos", "Selecione um aluno para remover.")
            return
        vals = self._tree_alunos.item(sel[0])["values"]
        cpf, nome = str(vals[0]), str(vals[1])
        if messagebox.askyesno("Remover", f"Remover '{nome}' ({cpf})?"):
            db.remover_aluno(cpf)
            self._refresh_alunos()
            self._known_people = db.carregar_alunos()

    # =========================================================
    # RELATÓRIOS PANEL
    # =========================================================
    def _build_panel_relatorios(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        tk.Label(parent, text="Relatórios", font=("Helvetica", 14, "bold"),
                 bg="#f0f0f5").grid(row=0, column=0, pady=(16, 4))

        top = tk.Frame(parent, bg="#f0f0f5")
        top.grid(row=1, column=0, sticky="ew", padx=16, pady=4)

        self._rel_tipo = tk.StringVar(value="registros")
        for txt, val in [("Registros", "registros"),
                          ("Frequência", "frequencia"),
                          ("Desconhecidos", "tentativas")]:
            tk.Radiobutton(top, text=txt, variable=self._rel_tipo, value=val,
                           bg="#f0f0f5",
                           command=self._refresh_relatorios).pack(side="left", padx=8)

        tk.Label(top, text="De:", bg="#f0f0f5").pack(side="left", padx=(12, 2))
        self._rel_ini = tk.Entry(top, width=11, font=("Helvetica", 9))
        self._rel_ini.pack(side="left")
        tk.Label(top, text="Até:", bg="#f0f0f5").pack(side="left", padx=(8, 2))
        self._rel_fim = tk.Entry(top, width=11, font=("Helvetica", 9))
        self._rel_fim.pack(side="left")

        tk.Button(top, text="Filtrar", command=self._refresh_relatorios,
                  bg="#3498db", fg="white", relief="flat",
                  cursor="hand2", padx=6).pack(side="left", padx=6)
        tk.Button(top, text="Exportar CSV", command=self._exportar_csv,
                  bg="#27ae60", fg="white", relief="flat",
                  cursor="hand2", padx=6).pack(side="left")

        tf = tk.Frame(parent)
        tf.grid(row=2, column=0, sticky="nsew", padx=16, pady=4)
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        self._tree_rel = ttk.Treeview(tf, show="headings", height=14)
        self._tree_rel.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self._tree_rel.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._tree_rel.config(yscrollcommand=sb.set)

    def _refresh_relatorios(self):
        tipo        = self._rel_tipo.get()
        data_inicio = self._rel_ini.get().strip() or None
        data_fim    = self._rel_fim.get().strip() or None
        if data_fim:
            data_fim += " 23:59:59"

        if tipo == "registros":
            cols = ("CPF", "Nome", "Curso", "Timestamp")
            rows = [(r[0], r[1], r[2], r[3])
                    for r in db.listar_registros(data_inicio=data_inicio, data_fim=data_fim)]
        elif tipo == "frequencia":
            cols = ("CPF", "Nome", "Curso", "Acessos", "Primeiro", "Último")
            rows = [(r[0], r[1], r[2], r[3], r[4] or "-", r[5] or "-")
                    for r in db.relatorio_frequencia()]
        else:
            cols = ("ID", "Timestamp", "Foto")
            rows = list(db.listar_tentativas(data_inicio, data_fim))

        self._rel_cache = (tipo, cols, rows)
        self._tree_rel.config(columns=cols)
        for col in cols:
            self._tree_rel.heading(col, text=col)
            self._tree_rel.column(col, width=130, anchor="w")
        for r in self._tree_rel.get_children():
            self._tree_rel.delete(r)
        for row in rows:
            self._tree_rel.insert("", "end", values=row)

    def _exportar_csv(self):
        if not self._rel_cache:
            messagebox.showinfo("Exportar", "Carregue um relatório primeiro.")
            return
        tipo, cols, rows = self._rel_cache
        if not rows:
            messagebox.showinfo("Exportar", "Nenhum dado para exportar.")
            return
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        arquivo = f"{tipo}_{ts}.csv"
        with open(arquivo, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        messagebox.showinfo("Exportar", f"Exportado: {arquivo} ({len(rows)} linha(s))")

    # =========================================================
    # CONFIGURAÇÕES PANEL
    # =========================================================
    def _build_panel_config(self, parent):
        parent.columnconfigure(0, weight=1)

        tk.Label(parent, text="Configurações", font=("Helvetica", 14, "bold"),
                 bg="#f0f0f5").grid(row=0, column=0, pady=(16, 8))

        form = tk.Frame(parent, bg="#f0f0f5")
        form.grid(row=1, column=0, padx=32, sticky="ew")
        form.columnconfigure(1, weight=1)

        self._cfg_fields = [
            ("Horas entre registros",     "tempo_minimo_entre_registros_horas",  int),
            ("Threshold de similaridade", "threshold_similaridade",               float),
            ("Capturas por cadastro",     "n_embeddings_por_aluno",              int),
            ("Cooldown em tela (s)",      "tempo_entre_registros_segundos",       int),
        ]
        self._cfg_vars = {}
        for i, (lbl, key, _) in enumerate(self._cfg_fields):
            tk.Label(form, text=lbl + ":", bg="#f0f0f5",
                     font=("Helvetica", 10)).grid(row=i, column=0, sticky="w", pady=6)
            var = tk.StringVar(value=str(config.get(key)))
            self._cfg_vars[key] = var
            tk.Entry(form, textvariable=var, width=12,
                     font=("Helvetica", 10)).grid(row=i, column=1, sticky="w",
                                                   padx=(12, 0), pady=6)

        n = len(self._cfg_fields)
        tk.Label(form, text="Anti-spoofing:", bg="#f0f0f5",
                 font=("Helvetica", 10)).grid(row=n, column=0, sticky="w", pady=6)
        self._as_var = tk.BooleanVar(value=config.get('anti_spoofing_ativo'))
        tk.Checkbutton(form, variable=self._as_var,
                       bg="#f0f0f5").grid(row=n, column=1, sticky="w", padx=(12, 0))

        tk.Button(parent, text="Salvar", command=self._salvar_config,
                  bg="#27ae60", fg="white", font=("Helvetica", 11, "bold"),
                  relief="flat", cursor="hand2",
                  pady=6, padx=20).grid(row=2, column=0, pady=16)

        self._cfg_status = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._cfg_status, bg="#f0f0f5", fg="#27ae60",
                 font=("Helvetica", 10)).grid(row=3, column=0)

    def _salvar_config(self):
        limits = {
            'tempo_minimo_entre_registros_horas': lambda v: v >= 0,
            'threshold_similaridade':              lambda v: 0.1 <= v <= 1.5,
            'n_embeddings_por_aluno':              lambda v: 1 <= v <= 20,
            'tempo_entre_registros_segundos':      lambda v: v >= 0,
        }
        for lbl, key, tipo in self._cfg_fields:
            try:
                val = tipo(self._cfg_vars[key].get())
            except ValueError:
                messagebox.showerror("Configurações", f"Formato inválido: '{lbl}'.")
                return
            if not limits[key](val):
                messagebox.showerror("Configurações", f"Valor fora do intervalo: '{lbl}'.")
                return
            config.set(key, val)
        config.set('anti_spoofing_ativo', self._as_var.get())
        self._cfg_status.set("✅ Configurações salvas.")
        self.after(3000, lambda: self._cfg_status.set(""))

    # =========================================================
    # USUÁRIOS PANEL
    # =========================================================
    def _build_panel_usuarios(self, parent):
        parent.columnconfigure(0, weight=1)

        tk.Label(parent, text="Usuários do Sistema", font=("Helvetica", 14, "bold"),
                 bg="#f0f0f5").grid(row=0, column=0, pady=(16, 8))

        # lista de usuários
        lf = tk.Frame(parent)
        lf.grid(row=1, column=0, padx=32, sticky="ew")
        lf.columnconfigure(0, weight=1)

        self._lb_usuarios = tk.Listbox(lf, height=5, font=("Helvetica", 11),
                                        selectmode="single", relief="solid",
                                        borderwidth=1, activestyle="none")
        self._lb_usuarios.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self._lb_usuarios.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._lb_usuarios.config(yscrollcommand=sb.set)

        # separador visual
        tk.Frame(parent, bg="#cccccc", height=1).grid(
            row=2, column=0, sticky="ew", padx=32, pady=10)

        # adicionar usuário
        add_frame = tk.LabelFrame(parent, text=" Adicionar usuário ",
                                   bg="#f0f0f5", font=("Helvetica", 9))
        add_frame.grid(row=3, column=0, padx=32, sticky="ew", pady=(0, 8))
        add_frame.columnconfigure(1, weight=1)

        self._usr_add_vars = {}
        for i, (lbl, key, secret) in enumerate([("Usuário:", "user", False),
                                                  ("Senha:", "pass", True),
                                                  ("Confirmar:", "pass2", True)]):
            tk.Label(add_frame, text=lbl, bg="#f0f0f5",
                     font=("Helvetica", 10)).grid(row=i, column=0, sticky="w",
                                                   padx=8, pady=4)
            var = tk.StringVar()
            self._usr_add_vars[key] = var
            tk.Entry(add_frame, textvariable=var, width=20,
                     font=("Helvetica", 10),
                     show="•" if secret else "").grid(row=i, column=1, sticky="ew",
                                                       padx=(0, 8), pady=4)

        self._usr_add_msg = tk.StringVar()
        tk.Label(add_frame, textvariable=self._usr_add_msg,
                 bg="#f0f0f5", fg="#e74c3c",
                 font=("Helvetica", 9)).grid(row=3, column=0, columnspan=2)
        tk.Button(add_frame, text="Adicionar", command=self._adicionar_usuario,
                  bg="#3498db", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", cursor="hand2",
                  padx=10).grid(row=4, column=0, columnspan=2, pady=6)

        # alterar senha
        pw_frame = tk.LabelFrame(parent, text=" Alterar senha do usuário selecionado ",
                                  bg="#f0f0f5", font=("Helvetica", 9))
        pw_frame.grid(row=4, column=0, padx=32, sticky="ew", pady=(0, 8))
        pw_frame.columnconfigure(1, weight=1)

        self._usr_pw_vars = {}
        for i, (lbl, key) in enumerate([("Nova senha:", "pass"), ("Confirmar:", "pass2")]):
            tk.Label(pw_frame, text=lbl, bg="#f0f0f5",
                     font=("Helvetica", 10)).grid(row=i, column=0, sticky="w",
                                                   padx=8, pady=4)
            var = tk.StringVar()
            self._usr_pw_vars[key] = var
            tk.Entry(pw_frame, textvariable=var, width=20,
                     font=("Helvetica", 10), show="•").grid(row=i, column=1, sticky="ew",
                                                             padx=(0, 8), pady=4)

        self._usr_pw_msg = tk.StringVar()
        tk.Label(pw_frame, textvariable=self._usr_pw_msg,
                 bg="#f0f0f5", fg="#e74c3c",
                 font=("Helvetica", 9)).grid(row=2, column=0, columnspan=2)
        tk.Button(pw_frame, text="Alterar senha", command=self._alterar_senha,
                  bg="#f39c12", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", cursor="hand2",
                  padx=10).grid(row=3, column=0, columnspan=2, pady=6)

        # remover
        tk.Button(parent, text="Remover usuário selecionado",
                  command=self._remover_usuario,
                  bg="#e74c3c", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", cursor="hand2",
                  padx=12, pady=5).grid(row=5, column=0, pady=4)

    def _refresh_usuarios(self):
        self._lb_usuarios.delete(0, "end")
        for u in db.listar_usuarios():
            self._lb_usuarios.insert("end", u)

    def _adicionar_usuario(self):
        user  = self._usr_add_vars["user"].get().strip()
        senha = self._usr_add_vars["pass"].get()
        conf  = self._usr_add_vars["pass2"].get()

        if not user or " " in user:
            self._usr_add_msg.set("Usuário inválido (sem espaços).")
            return
        if user in db.listar_usuarios():
            self._usr_add_msg.set("Usuário já existe.")
            return
        if len(senha) < 4:
            self._usr_add_msg.set("Senha deve ter ao menos 4 caracteres.")
            return
        if senha != conf:
            self._usr_add_msg.set("As senhas não coincidem.")
            return

        db.criar_usuario(user, senha)
        for v in self._usr_add_vars.values():
            v.set("")
        self._usr_add_msg.set(f"✅ Usuário '{user}' criado.")
        self._refresh_usuarios()
        self.after(3000, lambda: self._usr_add_msg.set(""))

    def _alterar_senha(self):
        sel = self._lb_usuarios.curselection()
        if not sel:
            self._usr_pw_msg.set("Selecione um usuário na lista.")
            return
        user  = self._lb_usuarios.get(sel[0])
        senha = self._usr_pw_vars["pass"].get()
        conf  = self._usr_pw_vars["pass2"].get()

        if len(senha) < 4:
            self._usr_pw_msg.set("Senha deve ter ao menos 4 caracteres.")
            return
        if senha != conf:
            self._usr_pw_msg.set("As senhas não coincidem.")
            return

        db.alterar_senha(user, senha)
        for v in self._usr_pw_vars.values():
            v.set("")
        self._usr_pw_msg.set(f"✅ Senha de '{user}' alterada.")
        self.after(3000, lambda: self._usr_pw_msg.set(""))

    def _remover_usuario(self):
        sel = self._lb_usuarios.curselection()
        if not sel:
            messagebox.showinfo("Usuários", "Selecione um usuário para remover.")
            return
        user = self._lb_usuarios.get(sel[0])
        if db.count_usuarios() <= 1:
            messagebox.showerror("Usuários",
                "Não é possível remover o último usuário.\nCrie outro antes.")
            return
        if messagebox.askyesno("Remover", f"Remover o usuário '{user}'?"):
            db.remover_usuario(user)
            self._refresh_usuarios()

    # =========================================================
    # CLOSE
    # =========================================================
    def _on_close(self):
        self._cam_off()
        self.destroy()


def main():
    app = App()
    if app.winfo_exists():
        app.mainloop()


if __name__ == "__main__":
    main()
