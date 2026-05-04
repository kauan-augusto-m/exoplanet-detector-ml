import joblib
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go

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

periodo = st.number_input(
    "Período Orbital (dias)",
    min_value=0.2,
    max_value=500.0,
    value=30.0,
    help="Tempo que o planeta leva para dar uma volta completa ao redor da estrela"
)
raio = st.number_input(
    "Raio do Planeta (raios terrestres)",
    min_value=0.5,
    max_value=100.0,
    value=2.0,
    help="Tamanho do planeta em comparação com a Terra. A Terra = 1.0"
)
profundidade_queda = st.number_input(
    "Profundidade do Trânsito (ppm)",
    min_value=0.0,
    max_value=100000.0,
    value=500.0,
    help="Queda de brilho da estrela durante o trânsito em partes por milhão"
)
duracao_transito = st.number_input(
    "Duração do Trânsito (horas)",
    min_value=0.05,
    max_value=50.0,
    value=3.5,
    help="Tempo que o planeta leva para atravessar o disco da estrela"
)
qualidade_sinal = st.number_input(
    "Qualidade do Sinal (SNR)",
    min_value=0.0,
    max_value=500.0,
    value=25.0,
    help="Relação sinal-ruído. Quanto maior, mais confiável é a detecção"
)
temp_estrela = st.number_input(
    "Temperatura da Estrela (K)",
    min_value=2500.0,
    max_value=16000.0,
    value=5800.0,
    help="Temperatura superficial da estrela em Kelvin. O Sol tem ~5778 K"
)
gravidade_estrela = st.number_input(
    "Gravidade da Estrela (log g)",
    min_value=0.0,
    max_value=5.5,
    value=4.4,
    help="Gravidade superficial da estrela em escala logarítmica. O Sol tem log g ≈ 4.44"
)
raio_estrela = st.number_input(
    "Raio da Estrela (R☉)",
    min_value=0.1,
    max_value=20.0,
    value=1.0,
    help="Tamanho da estrela em comparação com o Sol. O Sol = 1.0"
)
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

    # Gráfico de probabilidade
    cores = ['#2ecc71' if c == 'CONFIRMED' else '#e74c3c' for c in classes]
    
    fig = go.Figure(go.Bar(
        x=[probabilidade[0][i] * 100 for i in range(len(classes))],
        y=[c.replace('_', ' ') for c in classes],
        orientation='h',
        marker_color=cores,
        text=[f"{probabilidade[0][i]:.1%}" for i in range(len(classes))],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Confiança do Modelo",
        xaxis_title="Probabilidade (%)",
        xaxis=dict(range=[0, 100]),
        height=250
    )
    
    st.plotly_chart(fig, use_container_width=True)