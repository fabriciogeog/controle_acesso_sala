import sqlite3
import numpy as np

conn = sqlite3.connect('alunos.db')
cursor = conn.cursor()
cursor.execute("SELECT cpf, LENGTH(embedding) FROM alunos")
rows = cursor.fetchall()
for r in rows:
    print(f"CPF: {r[0]} | Tamanho embedding: {r[1]} bytes")  # Deve ser 2048 (512 * 4)
conn.close()
