# AMMCI: Aprendizagem de Máquina sob Mudança Temporal no Spotify Charts (2020-2026)

Repositório do trabalho prático da disciplina de **Aprendizagem de Máquina e Modelagem de
Conhecimento Incerto** da Universidade Estadual de Maringá (UEM), sob orientação do
Prof. Dr. Wagner Igarashi.

## Equipe
* **Nuno Miguel Mendonça Abilio**
* **Eduardo Angelo Rozada Minholi**
* **Janaina Maria Cera da Silva**

---

## 1. Visão Geral do Projeto

O objetivo deste projeto é analisar a ocorrência de **Data Drift** e **Concept Drift** ao
longo do tempo no consumo musical da plataforma Spotify, comparando diferentes estratégias
de atualização de modelos de Aprendizagem de Máquina:
* Treinamento exclusivamente histórico ($M_0$);
* Atualização por Fine-Tuning incremental ($M_{FT}$);
* Retreinamento completo contínuo ($M_{RT}$);
* Treinamento apenas com dados recentes ($M_{REC}$).

**Hipótese inicial:** espera-se que $M_0$ perca desempenho ao ser avaliado em dados futuros
($D_2$) devido a mudanças no comportamento de consumo musical entre 2020-2024 e 2025-2026, e
que o retreinamento completo ($M_{RT}$) apresente desempenho igual ou superior ao
fine-tuning ($M_{FT}$), por incorporar tanto o padrão histórico quanto o recente sem risco de
esquecimento catastrófico.

### Dataset e Escopo
* **Fonte:** [Spotify Charts Daily & Weekly Updated (Kaggle)](https://www.kaggle.com/datasets/gonzalopezgil/spotify-charts-daily-updated)
* **Mercados Filtrados:** Brasil (`br`), Grã-Bretanha (`gb`), Japão (`jp`), Estados Unidos
  (`us`) e África do Sul (`za`).
* **Período Temporal:** Semanalmente, de 22/10/2020 a 28/05/2026.
* **Problema:** Classificação binária.
* **Variável-Alvo (`target_trend_up`):** $1$ se o álbum subiu de posição (atingiu um ranking
  numericamente menor) na semana seguinte; $0$ caso tenha caído ou se mantido.
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
│   ├── 03_drift_and_distribution.ipynb       # Análise quantitativa de drift (KS/PSI/Qui²)
│   └── 04_experiments_evaluation.ipynb       # Tuning, M0, MFT, MRT, MREC, SHAP e Tabela 17
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Engenharia de atributos, split temporal e funções de drift
│   └── custom_models.py      # Algoritmo implementado do zero com NumPy
├── docs/
│   ├── diario_experimentos.md  # Diário com no mínimo 8 execuções relevantes documentadas
│   ├── short_paper/            # Artigo em formato SBC (a ser adicionado)
│   └── slides/                 # Apresentação em PDF (a ser adicionada)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. Guia de Instalação e Configuração do Ambiente

Siga as instruções abaixo no terminal para configurar o ambiente virtual e garantir a
paridade de versões entre os membros da equipe.

### 3.1. Clonar o Repositório

```bash
git clone https://github.com/[seu-usuario]/spotify-charts-ml.git
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

*(Quando ativado, o prefixo `(.venv)` aparecerá no início da linha de comando.)*

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

**Dependências principais (ver `requirements.txt` para versões mínimas):**
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `shap`.

---

## 4. Como Obter e Preparar a Base de Dados

> **Importante:** Arquivos `.csv` não são versionados no Git para não poluir o repositório
> (RN01.3).

1. Crie a pasta `data/raw/` na raiz do seu projeto (caso ela não exista).
2. Obtenha o arquivo consolidado de dados brutos com a equipe e coloque-o dentro de
   `data/raw/` (ex: `spotify_albums_all_countries.csv`), filtrado para as 5 nacionalidades
   do estudo e apenas para o tipo `album`.
3. Abra o VS Code (ou Jupyter Lab).
4. Abra o notebook `notebooks/01_data_preparation_and_split.ipynb`.
5. No canto superior direito do editor, clique em **Select Kernel** → **Python
   Environments...** → selecione o interpretador com a tag **`('.venv': venv)`**.
6. Clique em **Run All** (Executar Tudo).

Ao concluir a execução do notebook `01`, a pasta `data/processed/` conterá automaticamente:

* `d0.csv`: Conjunto histórico para modelagem e ajuste de hiperparâmetros.
* `d1.csv`: Conjunto recente para fine-tuning e testes intermediários.
* `d2.csv`: Conjunto futuro preservado para a avaliação comparativa cega.

---

## 5. Ordem de Execução Completa do Pipeline

Os notebooks devem ser executados **estritamente na ordem abaixo**, pois cada etapa depende
dos arquivos gerados pela anterior:

| Ordem | Notebook | O que faz | Depende de |
|---|---|---|---|
| 1 | `01_data_preparation_and_split.ipynb` | Engenharia de atributos, target e split D0/D1/D2 | `data/raw/*.csv` |
| 2 | `02_exploratory_data_analysis.ipynb` | Auditoria e AED focada em D0 | `data/processed/d0.csv` |
| 3 | `03_drift_and_distribution.ipynb` | Testes KS, PSI e Qui-Quadrado entre D0/D1/D2 | `data/processed/*.csv` |
| 4 | `04_experiments_evaluation.ipynb` | Tuning, M0/MFT/MRT/MREC, SHAP, Tabela 17 | `data/processed/*.csv` |

Cada notebook pode ser executado via **Run All**, com o kernel do `.venv` selecionado.

---

## 6. Reprodutibilidade e Seeds

Conforme RN10.1, todas as sementes aleatórias utilizadas nos experimentos estão explícitas
no código-fonte e documentadas aqui:

* **Seeds utilizadas nos experimentos com 3 inicializações (M0, MFT, MRT, MREC):**
  `SEEDS = [42, 123, 999]` (definidas em `notebooks/04_experiments_evaluation.ipynb`).
* **Seed do algoritmo do zero (`LogisticRegressionCustom`):** `42` (padrão do construtor,
  em `src/custom_models.py`).
* **Seed dos baselines de biblioteca (Regressão Logística e Random Forest):** `42`.
* **`TimeSeriesSplit`:** 3 folds, sem embaralhamento (ordem cronológica preservada).

O diário de experimentos completo, com hipóteses, resultados e interpretações de cada
decisão relevante, está em [`docs/diario_experimentos.md`](docs/diario_experimentos.md).

---

## 7. Uso de Ferramentas de IA Generativa

*A IA generativa (Claude, Anthropic) foi usada para auxiliar na compreensão do trabalho e de certos termos técnicos, para a criação dos épicos e user stories que guiaram a criação do projeto pela equipe, bem como para a solução de dúvidas e geração de texto para o relatório.*

---

## 8. Status do Checklist de Entrega

| Item | Status |
|---|---|
| Dataset do Kaggle com dados temporais recentes e proposta registrada | Feito |
| D0, D1 e D2 definidos cronologicamente, sem embaralhamento indevido | Feito |
| D2 não utilizado para ajustes, seleção ou pré-processamento | Feito |
| Algoritmo implementado do zero e comparado a uma biblioteca | Feito |
| Número mínimo de modelos atendido (n=3) | Feito |
| Modelo principal iniciado sem pesos pré-treinados | Feito |
| Experimentos M0, MFT, MRT e MREC executados | Feito |
| Pelo menos duas configurações de fine-tuning testadas | Feito |
| Pelo menos uma análise quantitativa de drift realizada | Feito (KS, PSI e Qui-Quadrado) |
| Métricas adequadas ao problema apresentadas | Feito |
| Três seeds/inicializações utilizadas quando aplicável | Feito |
| Diário com pelo menos 8 experimentos relevantes | Feito |
| Explicabilidade e análise de erros realizadas | Feito |
| Short paper em formato SBC concluído | Feito |
| Slides concluídos e salvos em formato PDF | Feito |
| README e dependências documentados | Feito |
| Uso de IA generativa declarado | Feito|
| Todos os integrantes preparados para a defesa individual | Feito |