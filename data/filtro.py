import pandas as pd

# Caminho do arquivo original (ajuste para o local do seu CSV)
INPUT_FILE = "data/charts_albums_weekly.csv"

# Caminho do novo arquivo que será gerado apenas com os dados do Brasil
OUTPUT_FILE = "data/charts_br_albums_weekly.csv"

# Quantidade de linhas lidas por vez (ajuste conforme a memória da sua máquina)
CHUNK_SIZE = 100_000

total_linhas = 0
total_linhas_br = 0
primeiro_chunk = True

# Lê o CSV em pedaços (chunks) em vez de carregar tudo de uma vez na memória
for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    total_linhas += len(chunk)

    # Filtra apenas as linhas em que o país é "br" dentro desse pedaço
    chunk_br = chunk[chunk["country"] == "br"]
    total_linhas_br += len(chunk_br)

    # Escreve no arquivo de saída:
    # - no primeiro chunk, escreve com cabeçalho e cria o arquivo (modo "w")
    # - nos demais, apenas adiciona as linhas, sem repetir o cabeçalho (modo "a")
    if primeiro_chunk:
        chunk_br.to_csv(OUTPUT_FILE, index=False, mode="w", header=True)
        primeiro_chunk = False
    else:
        chunk_br.to_csv(OUTPUT_FILE, index=False, mode="a", header=False)

print(f"Linhas lidas no total: {total_linhas}")
print(f"Linhas filtradas (country == 'br'): {total_linhas_br}")
print(f"Novo arquivo salvo em: {OUTPUT_FILE}")