import sqlite3
import numpy as np

DATABASE = 'alunos.db'


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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tentativas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                foto_path TEXT
            )
        ''')


def atualizar_banco():
    """Cria estruturas novas e migra dados existentes se necessário."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        try:
            cursor.execute('ALTER TABLE alunos ADD COLUMN ultimo_acesso TEXT')
            print("✅ Coluna 'ultimo_acesso' adicionada ao banco de dados.")
        except sqlite3.OperationalError:
            pass  # Já existe

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpf TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (cpf) REFERENCES alunos (cpf)
            )
        ''')

        cursor.execute('SELECT cpf, embedding FROM alunos WHERE embedding IS NOT NULL')
        alunos_existentes = cursor.fetchall()
        migrados = 0
        for cpf, emb_blob in alunos_existentes:
            cursor.execute('SELECT COUNT(*) FROM embeddings WHERE cpf = ?', (cpf,))
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    'INSERT INTO embeddings (cpf, embedding) VALUES (?, ?)',
                    (cpf, emb_blob)
                )
                migrados += 1
        if migrados > 0:
            print(f"✅ {migrados} embedding(s) migrado(s) para a nova tabela.")


def buscar_aluno(cpf):
    """Retorna (nome, curso, ultimo_acesso) ou None se não existir."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT nome, curso, ultimo_acesso FROM alunos WHERE cpf = ?', (cpf,))
        return cursor.fetchone()


def salvar_aluno(cpf, nome, curso, embeddings):
    """Insere ou substitui aluno e grava todos os seus embeddings."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO alunos (cpf, nome, curso, ultimo_acesso, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', (cpf, nome, curso, None, embeddings[0].tobytes()))
        cursor.execute('DELETE FROM embeddings WHERE cpf = ?', (cpf,))
        for emb in embeddings:
            cursor.execute(
                'INSERT INTO embeddings (cpf, embedding) VALUES (?, ?)',
                (cpf, emb.tobytes())
            )


def carregar_alunos():
    """Retorna lista de (cpf, nome, curso, [embeddings]) para reconhecimento."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.cpf, a.nome, a.curso, e.embedding
            FROM alunos a
            JOIN embeddings e ON a.cpf = e.cpf
            ORDER BY a.cpf
        ''')
        rows = cursor.fetchall()

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


def ja_registrado_recentemente(cpf, horas_minimas):
    """Retorna True se já houve registro nas últimas `horas_minimas` horas."""
    intervalo = f'-{int(horas_minimas)} hours'
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM registros WHERE cpf = ? AND timestamp > datetime('now', ?)",
            (cpf, intervalo)
        )
        return cursor.fetchone() is not None


def listar_alunos():
    """Retorna [(cpf, nome, curso, ultimo_acesso, n_embeddings)]."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.cpf, a.nome, a.curso, a.ultimo_acesso, COUNT(e.id)
            FROM alunos a
            LEFT JOIN embeddings e ON a.cpf = e.cpf
            GROUP BY a.cpf
            ORDER BY a.nome
        ''')
        return cursor.fetchall()


def remover_aluno(cpf):
    """Remove aluno e todos os seus dados. Retorna True se encontrado."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM embeddings WHERE cpf = ?', (cpf,))
        cursor.execute('DELETE FROM registros WHERE cpf = ?', (cpf,))
        cursor.execute('DELETE FROM alunos WHERE cpf = ?', (cpf,))
        return cursor.rowcount > 0


def listar_registros(cpf=None, data_inicio=None, data_fim=None):
    """Retorna registros com filtros opcionais, mais recentes primeiro."""
    query = 'SELECT cpf, nome, curso, timestamp, foto_path FROM registros WHERE 1=1'
    params = []
    if cpf:
        query += ' AND cpf = ?'
        params.append(cpf)
    if data_inicio:
        query += ' AND timestamp >= ?'
        params.append(data_inicio)
    if data_fim:
        query += ' AND timestamp <= ?'
        params.append(data_fim)
    query += ' ORDER BY timestamp DESC'
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()


def relatorio_frequencia():
    """Retorna [(cpf, nome, curso, total_acessos, primeiro_acesso, ultimo_acesso)]."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.cpf, a.nome, a.curso,
                   COUNT(r.id),
                   MIN(r.timestamp),
                   MAX(r.timestamp)
            FROM alunos a
            LEFT JOIN registros r ON a.cpf = r.cpf
            GROUP BY a.cpf
            ORDER BY COUNT(r.id) DESC, a.nome
        ''')
        return cursor.fetchall()


def salvar_tentativa_desconhecida(foto_path, timestamp_str):
    """Grava tentativa de acesso de rosto não reconhecido."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO tentativas (timestamp, foto_path) VALUES (?, ?)',
            (timestamp_str, foto_path)
        )


def listar_tentativas(data_inicio=None, data_fim=None):
    """Retorna tentativas de acesso não reconhecidas, mais recentes primeiro."""
    query = 'SELECT id, timestamp, foto_path FROM tentativas WHERE 1=1'
    params = []
    if data_inicio:
        query += ' AND timestamp >= ?'
        params.append(data_inicio)
    if data_fim:
        query += ' AND timestamp <= ?'
        params.append(data_fim)
    query += ' ORDER BY timestamp DESC'
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()


def salvar_registro(cpf, nome, curso, timestamp_str, foto_path):
    """Grava registro de entrada e atualiza ultimo_acesso do aluno."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO registros (cpf, nome, curso, timestamp, foto_path) VALUES (?, ?, ?, ?, ?)',
            (cpf, nome, curso, timestamp_str, foto_path)
        )
        cursor.execute(
            'UPDATE alunos SET ultimo_acesso = ? WHERE cpf = ?',
            (timestamp_str, cpf)
        )
