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

print(df.info())

print(df['falha_maquina'].value_counts())

# Convertendo todos os valores da coluna 'falha_maquina' para minúsculas
df['falha_maquina'] = df['falha_maquina'].str.lower()

# Unificando os valores que indicam 'sem_falha'
df['falha_maquina'] = df['falha_maquina'].replace(['não', 'n', '0'], 'sem_falha')

# Unificando os valores que indicam 'com_falha' para um melhor entendimento
df['falha_maquina'] = df['falha_maquina'].replace(['sim', 'y', '1'], 'com_falha')

# Imprimindo a contagem de valores para verificar a mudança visual
print(df['falha_maquina'].value_counts())

# Imprime a contagem de valores para a coluna de 'Falha Desgaste Ferramenta'
print(df['FDF (Falha Desgaste Ferramenta)'].value_counts())

# Unificação dos valores que indicam 'sem falha' para 0
df['FDF (Falha Desgaste Ferramenta)'] = df['FDF (Falha Desgaste Ferramenta)'].replace(['False', 'N', '0', '-'], 0)

# Unificação dos valores que indicam 'com falha' para 1
df['FDF (Falha Desgaste Ferramenta)'] = df['FDF (Falha Desgaste Ferramenta)'].replace(['True', '1'], 1)

# Imprimindo a contagem de valores para verificar a mudança visual
print(df['FDF (Falha Desgaste Ferramenta)'].value_counts())

print(df['FDC (Falha Dissipacao Calor)'].value_counts())

# Unificando os valores que indicam 'sem falha' para 0
df['FDC (Falha Dissipacao Calor)'] = df['FDC (Falha Dissipacao Calor)'].replace(['False', 'nao', '0'], 0)

# Unificando os valores que indicam 'com falha' para 1
df['FDC (Falha Dissipacao Calor)'] = df['FDC (Falha Dissipacao Calor)'].replace(['True', 'y', '1'], 1)

# Imprimindo a contagem de valores para verificar a mudança
print(df['FDC (Falha Dissipacao Calor)'].value_counts())