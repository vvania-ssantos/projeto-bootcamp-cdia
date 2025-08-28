import pandas as pd
import os

# Pegando o caminho do diretório onde este arquivo de código está
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

# Construindo o caminho completo para o arquivo de dados de forma confiável
caminho_arquivo = os.path.join(diretorio_atual, '..', 'data', 'bootcamp_train.csv')

# Carregando o arquivo para um DataFrame do pandas
df = pd.read_csv(caminho_arquivo)

# Imprimindo as primeiras 5 linhas do DataFrame
print(df.head())