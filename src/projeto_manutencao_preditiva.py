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

print(df['FP (Falha Potencia)'].value_counts())

# Unificando os valores que indicam 'sem falha' para 0
df['FP (Falha Potencia)'] = df['FP (Falha Potencia)'].replace(['False', 'N', '0'], 0)

# Unificando os valores que indicam 'com falha' para 1
df['FP (Falha Potencia)'] = df['FP (Falha Potencia)'].replace(['True', '1'], 1)

# Imprimindo a contagem de valores para verificar a mudança

print(df['FA (Falha Aleatoria)'].value_counts())

# Unificando os valores que indicam 'sem falha' para 0
df['FA (Falha Aleatoria)'] = df['FA (Falha Aleatoria)'].replace(['Não', 'não', '0', '-'], 0)

# Unificando os valores que indicam 'com falha' para 1
df['FA (Falha Aleatoria)'] = df['FA (Falha Aleatoria)'].replace(['Sim', 'sim', '1'], 1)

# Imprimindo a contagem de valores para verificar a mudança
print(df['FA (Falha Aleatoria)'].value_counts())

# Analisando as estatísticas das colunas numéricas
print(df.describe())

# Unificando os valores que indicam 'sem falha' para 0
df['FA (Falha Aleatoria)'] = df['FA (Falha Aleatoria)'].replace(['Não', 'não', '0', '-'], 0)

# Unificando os valores que indicam 'com falha' para 1
df['FA (Falha Aleatoria)'] = df['FA (Falha Aleatoria)'].replace(['Sim', 'sim', '1'], 1)

# Imprimindo a contagem de valores para verificar a mudança
print(df['FA (Falha Aleatoria)'].value_counts())

# Tratando dados faltantes nas colunas numéricas
colunas_numericas_com_nan = ['temperatura_ar', 'temperatura_processo', 'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']

for coluna in colunas_numericas_com_nan:
    mediana = df[coluna].median()
    df[coluna].fillna(mediana, inplace=True)

# Imprimindo o df.info() novamente para confirmar que não há mais valores nulos
print(df.info())

# Listando as colunas numéricas com valores faltantes
colunas_numericas_com_nan = ['temperatura_ar', 'temperatura_processo', 'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']

# Preenchendo os valores faltantes de cada coluna com a mediana dela
for coluna in colunas_numericas_com_nan:
    mediana = df[coluna].median()
    df[coluna] = df[coluna].fillna(mediana)

# Imprimindo o df.info() novamente para confirmar que não há mais valores nulos
print(df.info())

print(df['tipo'].value_counts())

# Analisando a coluna 'tipo'
tipo_dummies = pd.get_dummies(df['tipo'], prefix='tipo')

# Concatenando o novo DataFrame de 'dummies' ao DataFrame original
df = pd.concat([df, tipo_dummies], axis=1)

# Removendo a coluna 'tipo' original
df = df.drop('tipo', axis=1)

# Imprimindo o df.head() para ver as novas colunas
print(df.head())

print(df['id_produto'].value_counts())

# Remove a coluna 'id_produto'
df = df.drop('id_produto', axis=1)

# Imprimindo o df.info() para verificar se a coluna foi removida
print(df.info())

# Removendo as colunas que não serão usadas para o modelo
df = df.drop(['id'], axis=1)

# Separando as variáveis preditoras (X) e a variável-alvo (y)
X = df.drop('falha_maquina', axis=1)
y = df['falha_maquina']

# Imprimindo o formato (shape) dos novos DataFrames para verificação
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)

print(X.info())

print(df['FP (Falha Potencia)'].value_counts())

print(df['FP (Falha Potencia)'].value_counts())

# Unificando os valores que indicam 'sem falha' para 0
df['FP (Falha Potencia)'] = df['FP (Falha Potencia)'].replace(['Não', 'não', 'N', '0'], 0)

# Unificando os valores que indicam 'com falha' para 1
df['FP (Falha Potencia)'] = df['FP (Falha Potencia)'].replace(['Sim', 'sim', '1', 'y'], 1)

# Imprimindo a contagem de valores para verificar a mudança
print(df['FP (Falha Potencia)'].value_counts())

print(df['FP (Falha Potencia)'].value_counts())