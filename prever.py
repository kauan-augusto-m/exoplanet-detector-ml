import joblib
import pandas as pd

# Carrega o modelo já treinado
modelo = joblib.load('modelo_exoplanetas.pkl')

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
}])

previsao = modelo.predict(nova_estrela)
probabilidade = modelo.predict_proba(nova_estrela)

print(f"Previsão: {previsao[0]}")
print(f"Probabilidade de ser planeta: {probabilidade[0][0]:.2%}")