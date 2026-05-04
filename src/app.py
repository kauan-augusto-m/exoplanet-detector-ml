import joblib
import pandas as pd
import streamlit as st
import os

diretorio_atual = os.path.dirname(__file__)
caminho_modelo = os.path.join(diretorio_atual, '..', 'models', 'modelo_exoplanetas.pkl')

@st.cache_resource
def carregar_modelo():
    if not os.path.exists(caminho_modelo):
        st.warning("Modelo não encontrado. Treinando agora, aguarde...")
        import subprocess
        subprocess.run(["python", os.path.join(diretorio_atual, "modelo.py")])
    return joblib.load(caminho_modelo)

modelo = carregar_modelo()
st.title("Detector de Exoplanetas")
st.write("Digite as características da estrela para saber se possui um planeta.")

periodo = st.number_input("Período orbital (dias)", value=30.0,help="Tempo que o planeta leva para dar uma volta completa ao redor da estrela")
raio = st.number_input(
    "Raio do Planeta (raios terrestres)",
    value=2.0,
    help="Tamanho do planeta em comparação com a Terra. A Terra = 1.0"
)
profundidade_queda = st.number_input("Profundidade do Trânsito (ppm)", value=500.0)
duracao_transito = st.number_input("Duração do Trânsito (horas)", value= 3.5)
qualidade_sinal = st.number_input("Qualidade do Sinal (SNR)", value= 25.0)
temp_estrela = st.number_input(
    "Temperatura da Estrela (K)",
    value=5800.0,
    help="Temperatura superficial da estrela em Kelvin. O Sol tem ~5778 K"
)
gravidade_estrela = st.number_input(
    "Gravidade da Estrela (log g)",
    value=4.4,
    help="Gravidade superficial da estrela em escala logarítmica. O Sol tem log g ≈ 4.44"
)
raio_estrela = st.number_input("Raio da Estrela (R☉)", value= 1.0)

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