# AMMCI: Aprendizagem de Máquina sob Mudança Temporal no Spotify Charts (2020-2026)

Repositório do trabalho prático da disciplina de **Aprendizagem de Máquina e Modelagem de Conhecimento Incerto** da Universidade Estadual de Maringá (UEM), sob orientação do Prof. Dr. Wagner Igarashi.

## Equipe
* **Nuno Miguel Mendonça Abilio**
* **Eduardo Angelo Rozada Minholi**
* **Janaina Maria Cera da Silva**

---

## 1. Visão Geral do Projeto

O objetivo deste projeto é analisar a ocorrência de **Data Drift** e **Concept Drift** ao longo do tempo no consumo musical da plataforma Spotify, comparando diferentes estratégias de atualização de modelos de Aprendizagem de Máquina:
* Treinamento exclusivamente histórico ($M_0$);
* Atualização por Fine-Tuning incremental ($M_{FT}$);
* Retreinamento completo contínuo ($M_{RT}$);
* Treinamento apenas com dados recentes ($M_{REC}$).

### Dataset e Escopo
* **Fonte:** [Spotify Charts Daily & Weekly Updated (Kaggle)](https://www.kaggle.com/datasets/gonzalopezgil/spotify-charts-daily-updated)
* **Mercados Filtrados:** Brasil (`br`), Grã-Bretanha (`gb`), Japão (`jp`), Estados Unidos (`us`) e África do Sul (`za`).
* **Período Temporal:** Semanalmente, de 22/10/2020 a 28/05/2026.
* **Problema:** Classificação binária.
* **Variável-Alvo (`target_trend_up`):** $1$ se o álbum subiu de posição (atingiu um ranking numericamente menor) na semana seguinte; $0$ caso tenha caído ou se mantido.
* **Partições Temporais:**
  * **$D_0$ (Histórico - ~60%):** Outubro/2020 a Fevereiro/2024.
  * **$D_1$ (Recente - ~20%):** Março/2024 a Abril/2025.
  * **$D_2$ (Futuro - ~20%):** Maio/2025 a Maio/2026 (*conjunto cego de teste final*).

---

## 2. Estrutura do Repositório

```text
spotify-charts-ml/
├── data/
│   ├── raw/                  # Contém o arquivo CSV bruto original (ignorado pelo git)
│   └── processed/            # Onde os scripts geram d0.csv, d1.csv e d2.csv (ignorado pelo git)
├── notebooks/
│   ├── 01_data_preparation_and_split.ipynb   # Engenharia de atributos, target e divisão D0/D1/D2
│   ├── 02_exploratory_data_analysis.ipynb    # Auditoria formal e AED com foco em D0
│   ├── 03_drift_and_distribution.ipynb       # Análise quantitativa de drift (KS/PSI)
│   └── 04_experiments_evaluation.ipynb       # Treinamento e avaliação M0, MFT, MRT e MREC
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Funções de engenharia de atributos e classes anti-leakage
│   └── custom_models.py      # Algoritmo implementado do zero com NumPy
├── docs/                     # Relatório em formato SBC (LaTeX) e slides da apresentação
├── .gitignore
├── requirements.txt
└── README.md

```

---

## 3. Guia de Instalação e Configuração do Ambiente

Siga as instruções abaixo no terminal para configurar o ambiente virtual e garantir a paridade de versões.

### 3.1. Clonar o Repositório

```bash
git clone [https://github.com/](https://github.com/)[seu-usuario]/spotify-charts-ml.git
cd spotify-charts-ml

```

### 3.2. Criar e Ativar o Ambiente Virtual (`.venv`)

**No Windows (PowerShell):**

```powershell
# 1. Liberar execução de scripts caso ainda não tenha liberado
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Criar o ambiente virtual
python -m venv .venv

# 3. Ativar o ambiente
.\.venv\Scripts\Activate.ps1

```

*(Quando ativado, o prefixo `(.venv)` aparecerá no início da linha de comando).*

**No Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### 3.3. Instalar as Dependências

Com o ambiente ativado, execute:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install ipykernel

```

---

## 4. Como Obter e Preparar a Base de Dados

> **Importante:** Arquivos `.csv` não são versionados no Git para não poluir o repositório.

1. Crie a pasta `data/raw/` na raiz do seu projeto (caso ela não exista).
2. Obtenha o arquivo consolidado de dados brutos com a equipe e coloque-o dentro de `data/raw/` (ex: `spotify_albums_all_countries.csv`).
3. Abra o VS Code (ou Jupyter Lab).
4. Abra o notebook `notebooks/01_data_preparation_and_split.ipynb`.
5. No canto superior direito do editor, clique em **Select Kernel** -> **Python Environments...** -> selecione o interpretador com a tag **`('.venv': venv)`**.
6. Clique em **Run All** (Executar Tudo).

Ao concluir a execução do notebook `01`, a pasta `data/processed/` conterá automaticamente:

* `d0.csv`: Conjunto histórico para modelagem e ajuste de hiperparâmetros.
* `d1.csv`: Conjunto recente para fine-tuning e testes intermediários.
* `d2.csv`: Conjunto futuro preservado para a avaliação comparativa cega.

---

## 5. Como Executar a Auditoria e a Análise Exploratória (AED)

1. Abra o notebook `notebooks/02_exploratory_data_analysis.ipynb`.
2. Certifique-se de que o Kernel selecionado é o do `.venv`.
3. Clique em **Run All**.
4. O notebook executará:
* A checagem de tipos e dados ausentes;
* O relatório resumido de auditoria (comparativo de instâncias, nulos tratados e dimensões);
* O cálculo de balanceamento de classes da variável-alvo;
* Os gráficos de distribuição, relação idade versus tendência e a matriz de correlação linear das variáveis explicativas sobre $D_0$.

```