# 🪐 Exoplanet Detector with Machine Learning

Projeto de Machine Learning que utiliza dados reais da NASA para identificar exoplanetas com base em características observadas pelo telescópio Kepler.

---

## 📌 Visão Geral

A detecção de exoplanetas pode ser feita analisando pequenas variações no brilho de estrelas. Este projeto utiliza dados do catálogo Kepler Objects of Interest (KOI) para treinar um modelo capaz de classificar objetos como:

* ✅ **CONFIRMED** — Planetas confirmados
* ❌ **FALSE POSITIVE** — Falsos positivos

Além disso, o modelo é aplicado em novos candidatos para prever quais possuem maior probabilidade de serem exoplanetas reais.

---

## 🛰️ Fonte dos Dados

Dados públicos fornecidos pela NASA
Missão: Kepler Space Telescope

Método científico utilizado:

* Detecção por trânsito (queda de brilho estelar)

---

## ⚙️ Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Imbalanced-learn (SMOTE)

---

## 🧠 Pipeline do Projeto

1. Coleta de dados da API da NASA
2. Limpeza e seleção de features
3. Balanceamento com SMOTE
4. Treinamento com Random Forest
5. Avaliação do modelo
6. Predição de novos candidatos

---

## 📊 Features Utilizadas

O modelo utiliza apenas características físicas observáveis:

* Período orbital (`koi_period`)
* Profundidade do trânsito (`koi_depth`)
* Duração (`koi_duration`)
* Raio do planeta (`koi_prad`)
* Relação sinal/ruído (`koi_model_snr`)
* Temperatura da estrela (`koi_steff`)
* Gravidade da estrela (`koi_slogg`)
* Raio da estrela (`koi_srad`)

---

## 🧠 Observação Importante (Data Leakage)

Durante o desenvolvimento, foi identificado que algumas variáveis do dataset funcionavam como um **atalho para o modelo**, pois estavam diretamente relacionadas ao resultado final.

Isso é conhecido como *data leakage*.

Para evitar esse problema, o modelo final foi treinado **sem essas variáveis**, utilizando apenas dados físicos reais.

### Resultado:

* Modelo com leakage → ~99% (irrealista)
* Modelo final (honesto) → ~92%

Apesar da acurácia menor, o modelo final é **muito mais confiável e aplicável no mundo real**, pois consegue generalizar para estrelas nunca analisadas.

---

## 📈 Exemplo de Uso

```text
Previsão: CONFIRMED
Probabilidade de ser planeta: 97%
```

---

## 🚀 Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/kauan-augusto-m/exoplanet-detector-ml.git
cd exoplanet-detector-ml
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
python modelo.py
```

### 4. Fazer previsões

```bash
python prever.py
```
### 5. Interface web

```bash
streamlit run src/app.py
```

---

## 📊 Resultados

O modelo Random Forest conseguiu:

* Classificar corretamente planetas confirmados e falsos positivos
* Identificar candidatos promissores
* Demonstrar quais variáveis são mais relevantes

---

## ⚠️ Limitações

* Utiliza dados já processados (não usa curva de luz bruta)
* Dependente da qualidade do dataset
* Possível overfitting sem validação adequada
* SMOTE pode introduzir viés

---

## 🔮 Melhorias Futuras

* Uso de curvas de luz diretamente
* Aplicação de Deep Learning
* Detecção automática de trânsitos
* Publicar interface online (Streamlit Cloud)
* Cross-validation mais robusta

---

## 📚 Aprendizados

Este projeto envolve:

* Machine Learning aplicado a dados reais
* Engenharia de features
* Tratamento de dados desbalanceados
* Detecção de data leakage
* Boas práticas com Git

---

## 👨‍💻 Autor

Desenvolvido por Kauan Augusto como projeto de estudo em Machine Learning aplicado à astronomia.
