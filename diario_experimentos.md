# Diário de Experimentos — AMMCI (Spotify Charts)

Este diário documenta as principais decisões técnicas tomadas ao longo do desenvolvimento,
conforme exigido no item 13 do edital (mínimo de 8 execuções relevantes). Os valores de
resultado devem ser preenchidos com as saídas reais obtidas ao executar os notebooks
`03_drift_and_distribution.ipynb` e `04_experiments_evaluation.ipynb`.

**Seeds utilizadas em todos os experimentos estocásticos:** 42, 123, 999 (ver `SEEDS` no
Notebook 04). Ambiente documentado em `requirements.txt`.

---

## E01 — Ajuste de Hiperparâmetros do Modelo Principal (TimeSeriesSplit)

- **Alteração realizada:** Busca em grade sobre 4 configurações de arquitetura/otimização do
  MLP (`hidden_layer_sizes`, `learning_rate_init`, `alpha`), validadas com `TimeSeriesSplit`
  (3 folds) exclusivamente em D0.
- **Hipótese:** Uma arquitetura mais profunda (`(64,32)` ou `(128,64)`) capturaria melhor as
  interações entre variáveis de dinâmica de ranking do que uma camada única `(32,)`, mas com
  risco de overfitting em uma tarefa tabular relativamente simples.
- **Resultado:** `[preencher com a linha vencedora da tabela df_tuning — F1_mean ± F1_std]`
- **Interpretação:** `[preencher: a configuração vencedora foi a mais simples ou a mais
  complexa? Isso sugere que o problema tem baixa/alta complexidade não-linear?]`

---

## E02 — Regressão Logística do Zero (NumPy) vs. Scikit-learn

- **Alteração realizada:** Comparação direta entre a implementação própria
  (`LogisticRegressionCustom`) e `sklearn.linear_model.LogisticRegression`, usando exatamente
  as mesmas matrizes de treino (D0) e validação (D1).
- **Hipótese:** As duas implementações deveriam convergir para resultados de F1-Score
  semelhantes (diferença < 0.02), validando a corretude do algoritmo do zero; a versão de
  biblioteca deveria ser sensivelmente mais rápida por usar otimização vetorizada (LBFGS).
- **Resultado:** `[preencher com F1 e tempo de execução de ambas as versões]`
- **Interpretação:** `[preencher: a proximidade dos F1-Scores confirma que o gradiente
  descendente implementado está correto? A diferença de tempo foi a esperada?]`

---

## E03 — Baseline de Árvores (Random Forest) em D0

- **Alteração realizada:** Treinamento de um `RandomForestClassifier` (50 árvores,
  profundidade máxima 10) em D0, validado em D1, como terceira família de modelos
  (atendendo à exigência de n=3 famílias, RN06.2).
- **Hipótese:** Modelos baseados em árvores deveriam superar a regressão logística por
  capturarem relações não-lineares e interações entre `dist_from_peak`, `rank_jump_prev` e
  `persistence_ratio` sem exigir engenharia explícita de termos de interação.
- **Resultado:** `[preencher com F1 do Random Forest em D1, comparado ao F1 da LR]`
- **Interpretação:** `[preencher: o ganho (se houve) justifica a maior complexidade do
  modelo? Isso é um indício de que a fronteira de decisão não é linear?]`

---

## E04 — Fine-Tuning: Configuração A (Conservadora)

- **Alteração realizada:** A partir dos pesos de M0, continuação do treinamento com D1
  usando `learning_rate_init=0.0005` e apenas 10 épocas extras.
- **Hipótese:** Um ajuste sutil deveria adaptar o modelo ao período recente sem apagar o
  conhecimento acumulado em D0 (baixo risco de esquecimento catastrófico), mas também com
  ganho limitado se o drift entre D0 e D1 for grande.
- **Resultado:** `[preencher com F1 (D2) da Config A — ver comparativo_ft no Notebook 04]`
- **Interpretação:** `[preencher: o ganho/perda em relação a M0 foi pequeno, como esperado?]`

---

## E05 — Fine-Tuning: Configuração B (Agressiva)

- **Alteração realizada:** A partir dos mesmos pesos de M0, continuação do treinamento com
  D1 usando `learning_rate_init=0.01`, `alpha=0.001` (mais regularização) e 40 épocas extras.
- **Hipótese:** Uma adaptação mais forte deveria melhorar mais o desempenho em D2 (mais
  próximo temporalmente de D1), mas com maior risco de overfitting ao padrão recente e de
  esquecimento catastrófico do conhecimento histórico.
- **Resultado:** `[preencher com F1 (D2) da Config B, e indicar qual configuração venceu]`
- **Interpretação:** `[preencher: houve sinal de overfitting (queda de desempenho apesar de
  mais treino)? A configuração mais agressiva valeu a pena?]`

---

## E06 — Retreinamento Completo (MRT) com D0 + D1

- **Alteração realizada:** Reinicialização completa dos pesos do MLP (mesma arquitetura de
  M0) e treinamento do zero sobre a base combinada D0 ∪ D1.
- **Hipótese:** O retreinamento completo deveria superar o fine-tuning por ter acesso
  simultâneo ao padrão histórico e ao recente, sem o viés de partir de um ótimo local
  específico de D0.
- **Resultado:** `[preencher com F1 (D2) do MRT, comparado ao melhor MFT]`
- **Interpretação:** `[preencher: MRT superou o MFT? O ganho compensa o maior custo
  computacional/tempo de treinamento (registrar tempo de execução)?]`

---

## E07 — Treinamento Apenas com Dados Recentes (MREC)

- **Alteração realizada:** Reinicialização completa dos pesos e treinamento do zero usando
  exclusivamente D1 (sem nenhuma linha de D0).
- **Hipótese:** Se houver forte concept drift entre o período histórico e o recente, o MREC
  poderia superar M0 e até o MRT; se o padrão for majoritariamente estável, o MREC deveria
  performar pior que MRT por ter muito menos dados de treino.
- **Resultado:** `[preencher com F1 (D2) do MREC, comparado a M0, MFT e MRT]`
- **Interpretação:** `[preencher: dados antigos ajudaram ou atrapalharam a previsão do
  período futuro? Isso favorece qual hipótese de drift?]`

---

## E08 — Quantificação de Drift (KS + PSI) entre D0, D1 e D2

- **Alteração realizada:** Aplicação de teste Kolmogorov-Smirnov e cálculo de PSI para as
  variáveis numéricas, comparando D0 vs. D1 e D0 vs. D2 (Notebook 03).
- **Hipótese:** Variáveis relacionadas à idade do álbum (`album_age_days`) e à distância do
  pico (`dist_from_peak`) apresentariam maior drift ao longo do tempo, refletindo mudanças no
  perfil de lançamentos e consumo entre 2020–2026.
- **Resultado:** `[preencher com os atributos de maior PSI/KS — ver ranking_drift]`
- **Interpretação:** `[preencher: quais atributos tiveram drift severo (PSI > 0.25)? Isso é
  compatível com a queda de desempenho observada em M0 entre D1 e D2?]`

---

## E09 — Drift Categórico (Qui-Quadrado) em `entry_status`

- **Alteração realizada:** Teste Qui-Quadrado comparando a distribuição de `entry_status`
  entre D0, D1 e D2.
- **Hipótese:** A proporção de `NEW_ENTRY` vs. `RE_ENTRY` vs. `MOVED_UP`/`MOVED_DOWN` deveria
  se manter relativamente estável, já que essas categorias refletem mecânica do próprio
  ranking, não comportamento externo — uma mudança significativa aqui indicaria alteração
  estrutural no próprio catálogo/algoritmo do Spotify.
- **Resultado:** `[preencher com p-valor e conclusão de drift/não-drift]`
- **Interpretação:** `[preencher: o resultado confirmou ou refutou a hipótese?]`

---

## E10 — Explicabilidade Comparativa (SHAP): M0 vs. Modelo Atualizado

- **Alteração realizada:** Cálculo de SHAP (`KernelExplainer`) para M0 e para o melhor modelo
  atualizado (MFT, MRT ou MREC, conforme F1 em D2), sobre a mesma amostra de D2.
- **Hipótese:** As variáveis de dinâmica de curto prazo (`rank`, `dist_from_peak`,
  `rank_jump_prev`) deveriam permanecer as mais importantes em ambos os modelos, indicando
  que a *relação* entre atributos e alvo é relativamente estável (baixo concept drift), mesmo
  que a *distribuição* dos atributos tenha mudado (data drift).
- **Resultado:** `[preencher com o ranking de importância de ambos os modelos —
  comparativo_shap]`
- **Interpretação:** `[preencher: a ordem de importância mudou? Isso é evidência de concept
  drift ou apenas de data drift?]`

---

## E11 — Análise de Erros: Falsos Positivos e Falsos Negativos em D2

- **Alteração realizada:** Isolamento dos casos de erro mais severos (maior confiança
  incorreta) do modelo atualizado em D2, com exemplos reais de álbuns/artistas.
- **Hipótese:** Falsos negativos (o modelo previu queda, mas o álbum subiu) deveriam se
  concentrar em álbuns antigos que tiveram picos inesperados (viralização, eventos externos);
  falsos positivos deveriam se concentrar em lançamentos recentes que já haviam atingido seu
  pico de popularidade.
- **Resultado:** `[preencher com os exemplos reais obtidos — nomes de álbuns/artistas]`
- **Interpretação:** `[preencher: os padrões observados confirmam a hipótese? Que tipo de
  atributo adicional poderia reduzir esses erros (ex: sinal de redes sociais, sazonalidade)?]`

---

> **Observação sobre reprodutibilidade:** todos os experimentos acima são reprodutíveis a
> partir da execução sequencial dos notebooks 01 → 02 → 03 → 04, com as seeds fixas
> documentadas no início deste diário. Nenhum experimento envolveu o uso de D2 para ajuste
> de hiperparâmetros ou seleção de modelo — D2 foi utilizado exclusivamente para a avaliação
> final (Tabela 17), conforme RN12.1.