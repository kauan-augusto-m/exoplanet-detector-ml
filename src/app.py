import joblib
import pandas as pd
import streamlit as st
import os

diretorio_atual = os.path.dirname(__file__)
caminho_modelo = os.path.join(diretorio_atual, '..', 'models', 'modelo_exoplanetas.pkl')

try:
    modelo = joblib.load(caminho_modelo)
except FileNotFoundError:
    st.error("Modelo não encontrado. Rode o modelo.py primeiro!")
    st.stop()

st.title("Detector de Exoplanetas")
st.write("Digite as características da estrela para saber se possui um planeta.")

periodo = st.number_input("Período orbital (dias)", value=30.0)
raio = st.number_input("Raio do planeta (raios terrestres)", value=2.0)
profundidade_queda = st.number_input("Profundidade de queda", value=500.0)
duracao_transito = st.number_input("Duracao do transito", value= 3.5)
qualidade_sinal = st.number_input("qualidade do sinal", value= 25.0)
temp_estrela = st.number_input("temperatura da estrela", value= 5800.0)
gravidade_estrela = st.number_input("Gravidade da estrela", value=4.4)
raio_estrela = st.number_input("raio da estrela", value= 1.0)

if st.button("Analisar estrela"):
    nova_estrela = pd.DataFrame([{
        'koi_period': periodo,
        'koi_depth': profundidade_queda,
        'koi_duration': duracao_transito,
        'koi_prad': raio,
        'koi_model_snr': qualidade_sinal,
        'koi_steff': temp_estrela,
        'koi_slogg': gravidade_estrela,
        'koi_srad': raio_estrela,
    }])

    previsao = modelo.predict(nova_estrela)
    probabilidade = modelo.predict_proba(nova_estrela)
    classes = modelo.classes_

    if previsao[0] == 'CONFIRMED':
        st.success(f"PLANETA DETECTADO!")
    else:
        st.error(f"FALSO POSITIVO")

    for i, classe in enumerate(classes):
        st.write(f"Confiança para {classe}: {probabilidade[0][i]:.2%}")