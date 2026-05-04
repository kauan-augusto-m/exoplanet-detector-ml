import joblib
import pandas as pd
import os

# 1. LOCALIZAR O MODELO
# Pega o caminho da pasta onde este script está e aponta para a pasta 'models'
diretorio_atual = os.path.dirname(__file__)
caminho_modelo = os.path.join(diretorio_atual, '..', 'models', 'modelo_exoplanetas.pkl')

try:
    # Carrega o modelo
    modelo = joblib.load(caminho_modelo)
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo não foi encontrado em: {caminho_modelo}")
    print("Certifique-se de rodar o 'modelo.py' primeiro para gerar o arquivo .pkl")
    exit()

# 2. DEFINIR CARACTERÍSTICAS (Na mesma ordem do treino!)
caracteristicas = [
    'koi_period', 'koi_depth', 'koi_duration', 'koi_prad',
    'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad'
]

# Exemplo: características de uma estrela nova
nova_estrela = pd.DataFrame([{
    'koi_period': 30.0,
    'koi_depth': 500.0,
    'koi_duration': 3.5,
    'koi_prad': 2.1,
    'koi_model_snr': 25.0,
    'koi_steff': 5800.0,
    'koi_slogg': 4.4,
    'koi_srad': 1.0,
}], columns=caracteristicas) # Garante a ordem das colunas

# 3. REALIZAR PREVISÃO
previsao = modelo.predict(nova_estrela)
probabilidade = modelo.predict_proba(nova_estrela)

# O Scikit-Learn organiza as classes em ordem alfabética: 
# [0] costuma ser 'CONFIRMED' e [1] 'FALSE POSITIVE'
# Vamos verificar qual é qual para não exibir o dado errado
classes = modelo.classes_

print("-" * 30)
print(f"Resultado para a Estrela de Teste:")
print(f"Previsão: {previsao[0]}")

# Exibe a probabilidade de acordo com o que o modelo aprendeu
for i, classe in enumerate(classes):
    print(f"Confiança para {classe}: {probabilidade[0][i]:.2%}")