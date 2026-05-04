import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. BAIXAR DADOS
url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv"
print("Baixando catálogo da NASA...")
df = pd.read_csv(url)
print(f"Total de objetos: {len(df)}")
print(df['koi_disposition'].value_counts())

# 2. SELECIONAR CARACTERÍSTICAS
caracteristicas = [
    'koi_period',
    'koi_depth',
    'koi_duration',
    'koi_prad',
    'koi_model_snr',
    'koi_steff',
    'koi_slogg',
    'koi_srad',
   
]
df_treino = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
X = df_treino[caracteristicas].dropna()
y = df_treino['koi_disposition'][X.index]
print(f"\nApós limpeza: {len(X)} objetos")
print(y.value_counts())

# 3. DIVIDIR TREINO E TESTE PRIMEIRO
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTreino: {len(X_treino)} | Teste: {len(X_teste)}")

# 4. BALANCEAR COM SMOTE APENAS NO TREINO
smote = SMOTE(random_state=42)
X_treino, y_treino = smote.fit_resample(X_treino, y_treino)
print(f"\nApós SMOTE no treino:")
print(pd.Series(y_treino).value_counts())
# 5. TREINAR MODELO
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

# 6. AVALIAR
y_previsto = modelo.predict(X_teste)
print(f"\nAcurácia: {accuracy_score(y_teste, y_previsto):.2%}")
print(classification_report(y_teste, y_previsto))

# 7. IMPORTÂNCIAS
importancias = pd.Series(
    modelo.feature_importances_, index=caracteristicas
).sort_values(ascending=False)
print("\nCaracterísticas mais importantes:")
print(importancias)
importancias.plot(kind='bar')
plt.title("O que o modelo usa pra detectar planetas")
plt.ylabel("Importância")
plt.tight_layout()
plt.show()

# 8. PREVER CANDIDATOS
df_candidatos = df[df['koi_disposition'] == 'CANDIDATE'].dropna(subset=caracteristicas)
previsoes = modelo.predict(df_candidatos[caracteristicas])
probabilidades = modelo.predict_proba(df_candidatos[caracteristicas])

resultado = df_candidatos.copy()
resultado['previsao'] = previsoes
resultado['probabilidade_planeta'] = probabilidades[:, 0]

planetas_previstos = resultado[resultado['previsao'] == 'CONFIRMED']
planetas_previstos = planetas_previstos.sort_values('probabilidade_planeta', ascending=False)

print(f"\nCandidatos previstos como planeta: {len(planetas_previstos)}")
print("\nTop 10 mais prováveis:")
print(planetas_previstos[['kepoi_name', 'koi_period', 'koi_prad', 'probabilidade_planeta']].head(10))


# Salva o modelo em disco


caminho_models = os.path.join(os.path.dirname(__file__), '..', 'models')
caminho_modelo = os.path.join(caminho_models, 'modelo_exoplanetas.pkl')

joblib.dump(modelo, caminho_modelo)

print(f"Modelo salvo em {caminho_modelo}")