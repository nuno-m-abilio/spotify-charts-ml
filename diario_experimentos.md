# Diário de Experimentos - AMMCI (Spotify Charts)

Este diário documenta as principais decisões técnicas tomadas ao longo do desenvolvimento,
conforme exigido no item 13 do edital (mínimo de 8 execuções relevantes).

**Seeds utilizadas em todos os experimentos estocásticos:** 42, 123, 999 (ver `SEEDS` no
Notebook 04). Ambiente documentado em `requirements.txt`.

**Volumetria das partições:** D0 = 163.932 registros | D1 = 57.497 registros |
D2 = 51.350 registros (após pré-processamento, 19 features finais).

---

## E01 - Ajuste de Hiperparâmetros do Modelo Principal (TimeSeriesSplit)

- **Alteração realizada:** Busca em grade sobre 4 configurações de arquitetura/otimização do
  MLP (`hidden_layer_sizes`, `learning_rate_init`, `alpha`), validadas com `TimeSeriesSplit`
  (3 folds) exclusivamente em D0.
- **Hipótese:** Uma arquitetura mais profunda (`(64,32)` ou `(128,64)`) capturaria melhor as
  interações entre variáveis de dinâmica de ranking do que uma camada única `(32,)`, mas com
  risco de overfitting em uma tarefa tabular relativamente simples.
- **Resultado:**

  | Arquitetura | learning_rate_init | alpha | F1 médio (TimeSeriesSplit) |
  |---|---|---|---|
  | **(128, 64)** | 0.001 | 0.0001 | **0.2633 ± 0.0157** (vencedora) |
  | (32,) | 0.001 | 0.0001 | 0.2563 ± 0.0052 |
  | (64, 32) | 0.001 | 0.0001 | 0.2530 ± 0.0315 |
  | (64, 32) | 0.010 | 0.0010 | 0.2434 ± 0.0191 |

- **Interpretação:** A arquitetura mais profunda venceu, mas por uma margem pequena (~0,01
  de F1 sobre a mais simples), e com desvio-padrão comparável entre as configurações. Isso
  sugere que a tarefa não possui uma estrutura não-linear muito complexa a ser explorada -
  o ganho de capacidade do modelo tem retorno marginal decrescente. A configuração `(64,32)`
  com `lr=0.01` teve o pior desempenho médio e o maior desvio-padrão entre folds, indicando
  instabilidade de convergência com taxa de aprendizado mais alta.

---

## E02 - Regressão Logística do Zero (NumPy) vs. Scikit-learn

- **Alteração realizada:** Comparação direta entre a implementação própria
  (`LogisticRegressionCustom`, `lr=0.1`, 500 iterações) e `sklearn.linear_model.LogisticRegression`
  (`max_iter=500`), usando exatamente as mesmas matrizes de treino (D0) e validação (D1).
- **Hipótese:** As duas implementações deveriam convergir para resultados de F1-Score
  semelhantes, validando a corretude do algoritmo do zero; a versão de biblioteca deveria ser
  sensivelmente mais rápida por usar otimização vetorizada (LBFGS).
- **Resultado:**

  | Implementação | F1 (D1) | Tempo de execução |
  |---|---|---|
  | Regressão Logística Customizada (NumPy) | 0.3131 | 4.84s |
  | Regressão Logística (Scikit-learn) | 0.2997 | 0.33s |

- **Interpretação:** Os F1-Scores ficaram próximos (diferença de ~0,013), o que confirma que
  o gradiente descendente implementado do zero está funcionalmente correto - a pequena
  diferença é esperada, já que o scikit-learn usa por padrão o otimizador L-BFGS (uma
  quasi-Newton, que converge com trajetória diferente do gradiente descendente puro) e
  regularização L2 ativada por padrão, enquanto a implementação própria não usa regularização.
  Como esperado, a versão de biblioteca foi cerca de **15x mais rápida**, evidenciando o
  ganho de eficiência das rotinas otimizadas em C/Fortran do scikit-learn frente a um laço de
  gradiente descendente em NumPy puro.

---

## E03 - Baseline de Árvores (Random Forest) em D0

- **Alteração realizada:** Treinamento de um `RandomForestClassifier` (50 árvores,
  profundidade máxima 10) em D0, validado em D1, como terceira família de modelos
  (atendendo à exigência de n=3 famílias, RN06.2).
- **Hipótese:** Modelos baseados em árvores deveriam superar a regressão logística por
  capturarem relações não-lineares e interações entre `dist_from_peak`, `rank_jump_prev` e
  `persistence_ratio` sem exigir engenharia explícita de termos de interação.
- **Resultado:** F1 (D1) = **0.2194** - pior que ambas as versões de Regressão Logística
  (0.3131 e 0.2997).
- **Interpretação:** A hipótese foi **refutada**. O Random Forest, mesmo sendo um modelo
  não-linear, teve desempenho inferior à Regressão Logística na validação em D1. Uma
  explicação plausível é que, com `max_depth=10` e apenas 50 árvores, o modelo pode não ter
  capacidade suficiente para o volume de dados (163 mil linhas), ou que a relação entre os
  atributos numéricos (muitos já são "rankings" ou diferenças de rankings) e o alvo seja
  predominantemente monotônica/linear, favorecendo um classificador linear bem calibrado.
  Vale registrar como limitação: não foi feito tuning de hiperparâmetros para o Random Forest
  (apenas para o MLP), o que pode ter penalizado sua comparação.

---

## E04 - Fine-Tuning: Configuração A (Conservadora)

- **Alteração realizada:** A partir dos pesos de M0, continuação do treinamento com D1
  usando `learning_rate_init=0.0005` e apenas 10 épocas extras.
- **Hipótese:** Um ajuste sutil deveria adaptar o modelo ao período recente sem apagar o
  conhecimento acumulado em D0, mas com ganho limitado se o drift entre D0 e D1 for grande.
- **Resultado:** F1 (D2) = **0.2781 ± 0.0325** - praticamente empatado com M0 puro
  (0.2783 ± 0.0107), variação de **-0.06%**.
- **Interpretação:** Confirma a hipótese de ganho limitado: o ajuste conservador não alterou
  o desempenho médio de forma prática, embora tenha praticamente dobrado o desvio-padrão
  entre seeds (0.0107 → 0.0325), indicando maior instabilidade/sensibilidade à inicialização
  quando poucas épocas extras são aplicadas sobre pesos já convergidos.

---

## E05 - Fine-Tuning: Configuração B (Agressiva)

- **Alteração realizada:** A partir dos mesmos pesos de M0, continuação do treinamento com
  D1 usando `learning_rate_init=0.01`, `alpha=0.001` e 40 épocas extras.
- **Hipótese:** Uma adaptação mais forte deveria melhorar mais o desempenho em D2, mas com
  maior risco de overfitting ao padrão recente e de esquecimento catastrófico do
  conhecimento histórico.
- **Resultado:** F1 (D2) = **0.2383 ± 0.0459** - pior que M0 (0.2783) e pior que a Config A.
  **A Config A foi adotada como MFT oficial na Tabela 17.**
- **Interpretação:** A hipótese de risco foi confirmada: a configuração agressiva **piorou**
  o desempenho em relação ao próprio M0, com o maior desvio-padrão entre todas as
  configurações testadas (0.0459). Isso é um sinal claro de **esquecimento catastrófico** -
  o learning rate mais alto e as 40 épocas extras fizeram o modelo se afastar demais da
  solução histórica, sem que os dados de D1 (menores em volume que D0) fossem suficientes
  para compensar essa perda.

---

## E06 - Retreinamento Completo (MRT) com D0 + D1

- **Alteração realizada:** Reinicialização completa dos pesos do MLP (mesma arquitetura de
  M0) e treinamento do zero sobre a base combinada D0 ∪ D1.
- **Hipótese:** O retreinamento completo deveria superar o fine-tuning por ter acesso
  simultâneo ao padrão histórico e ao recente, sem o viés de partir de um ótimo local
  específico de D0.
- **Resultado:** F1 (D2) = **0.2962 ± 0.0123** - superior a M0 (+6.45%) e superior a ambas
  as configurações de MFT.
- **Interpretação:** A hipótese foi **confirmada**. O MRT superou o melhor MFT (Config A:
  0.2781) e o próprio M0, com desvio-padrão baixo e estável (0.0123, o menor entre as
  quatro estratégias - ver E13). Isso sugere que treinar do zero com todo o histórico
  acumulado é mais robusto do que continuar o treinamento a partir de um modelo já
  convergido em D0.

---

## E07 - Treinamento Apenas com Dados Recentes (MREC)

- **Alteração realizada:** Reinicialização completa dos pesos e treinamento do zero usando
  exclusivamente D1 (sem nenhuma linha de D0).
- **Hipótese:** Se houver forte concept drift entre o período histórico e o recente, o MREC
  poderia superar M0 e até o MRT; se o padrão for majoritariamente estável, o MREC deveria
  performar pior que MRT por ter muito menos dados de treino.
- **Resultado:** F1 (D2) = **0.3260 ± 0.0148** - o **melhor resultado entre as quatro
  estratégias**, superando M0 em **+17.15%** e superando também o MRT (+10% relativo).
- **Interpretação:** A hipótese de "dados antigos atrapalham" foi a que melhor se confirmou.
  Mesmo com menos da metade do volume de dados de treino do MRT, o MREC generalizou melhor
  para D2. Isso é evidência de que o padrão de consumo mudou o suficiente entre a era
  histórica e a recente para que os dados de D0 introduzam **ruído** em vez de sinal útil -
  um indício relevante de **concept drift**, e não apenas data drift, já que mais dados
  (MRT) não superaram menos dados mais recentes (MREC). Essa é a informação mais importante
  para a seção de Discussão do relatório.

---

## E08 - Quantificação de Drift Numérico (KS + PSI) entre D0, D1 e D2

- **Alteração realizada:** Teste Kolmogorov-Smirnov e cálculo de PSI para as variáveis
  numéricas, comparando D0 vs. D1 e D0 vs. D2 (Notebook 03).
- **Hipótese:** Variáveis relacionadas à idade do álbum (`album_age_days`) e à distância do
  pico (`dist_from_peak`) apresentariam maior drift ao longo do tempo, refletindo mudanças no
  perfil de lançamentos e consumo entre 2020–2026.
- **Resultado (PSI, D0 vs D2, ordenado por magnitude):**

  | Atributo | KS (D0 vs D2) | PSI (D0 vs D2) | Classificação |
  |---|---|---|---|
  | `weeks_on_chart` | 0.3474 | **0.6380** | Drift severo |
  | `consecutive_weeks` | 0.2092 | **0.2586** | Drift severo |
  | `peak_rank` | 0.1618 | 0.1348 | Drift moderado |
  | `album_age_days` | 0.1413 | 0.1172 | Drift moderado |
  | `dist_from_peak` | 0.1336 | 0.0938 | Drift leve/moderado |
  | `rank_jump_prev` | 0.0331 | 0.0153 | Sem drift relevante |
  | `rank` | 0.0047 | 0.0002 | Sem drift |

- **Interpretação:** A hipótese foi **parcialmente refutada**: os maiores drifts não vieram
  de `album_age_days`/`dist_from_peak` (que tiveram drift moderado/leve), e sim de
  `weeks_on_chart` e `consecutive_weeks` - variáveis de **permanência** dos álbuns no
  ranking. Isso sugere uma mudança estrutural em como os álbuns se comportam ao longo do
  tempo no gráfico (ex: álbuns permanecendo mais tempo no ranking em anos recentes, possível
  reflexo de mudanças no algoritmo de recomendação do Spotify ou em hábitos de consumo em
  streaming). Como esperado, `rank` não mostrou drift (é a variável usada para construir os
  cortes de Top 50/dinâmicas, mantendo distribuição por desenho do próprio ranking, que
  sempre vai de 1 a 200).

---

## E09 - Drift Categórico (Qui-Quadrado) em `country` e `entry_status`

- **Alteração realizada:** Teste Qui-Quadrado comparando a distribuição de `country` e
  `entry_status` entre D0, D1 e D2.
- **Hipótese:** A proporção de `country` deveria se manter estável (desenho amostral fixo);
  já `entry_status` poderia mudar caso o comportamento de entrada/permanência dos álbuns no
  ranking tenha se alterado estruturalmente.
- **Resultado:**

  | Atributo | Chi² (D0 vs D2) | p-valor (D2) | Drift? |
  |---|---|---|---|
  | `country` | 1.7599 | 0.7798 | **Não** |
  | `entry_status` | 340.7908 | < 0.0001 | **Sim** |

- **Interpretação:** A hipótese foi **confirmada** nos dois atributos. `country` permaneceu
  estatisticamente estável entre D0 e D2 (p=0.78), validando que o desenho amostral (5
  países fixos) não introduziu viés temporal artificial. Já `entry_status` mudou de forma
  altamente significativa, o que é consistente com o drift observado em E08 nas variáveis de
  permanência (`weeks_on_chart`, `consecutive_weeks`) - reforça a hipótese de que a *mecânica*
  de entrada/saída/permanência dos álbuns no ranking mudou estruturalmente ao longo do
  período estudado.

---

## E10 - Explicabilidade Comparativa (SHAP): M0 vs. MREC

- **Alteração realizada:** Cálculo de SHAP (`KernelExplainer`) para M0 e para o modelo
  atualizado com melhor F1 em D2 (selecionado automaticamente: **MREC**), sobre a mesma
  amostra de 50 instâncias de D2.
- **Hipótese:** As variáveis de dinâmica de curto prazo (`rank`, `dist_from_peak`,
  `rank_jump_prev`) deveriam permanecer as mais importantes em ambos os modelos, indicando
  relação estável entre atributos e alvo (baixo concept drift), mesmo com mudança na
  distribuição dos atributos (data drift).
- **Resultado (importância média |SHAP|, top variáveis):**

  | Atributo | Importância M0 | Importância MREC | Variação |
  |---|---|---|---|
  | `rank` | 0.0611 (1º) | 0.0408 (2º) | **-33.3%** |
  | `peak_rank` | 0.0559 (2º) | 0.0673 (1º) | **+20.4%** |
  | `dist_from_peak` | 0.0353 (3º) | 0.0437 (3º) | +23.8% |
  | `album_age_days` | 0.0220 | 0.0114 | **-48.1%** |
  | `consecutive_weeks` | 0.0138 | 0.0185 | +34.3% |
  | `entry_status_MOVED_DOWN` | 0.0149 | 0.0199 | +33.5% |

- **Interpretação:** A hipótese foi **parcialmente refutada**: houve uma **inversão** no
  topo do ranking - `rank` (posição atual) era a variável mais importante em M0 e caiu para
  a 2ª posição no MREC, cedendo lugar a `peak_rank` (posição de pico histórico). Isso é um
  indício de **concept drift**, não apenas data drift: o modelo atualizado passou a se apoiar
  mais no "quão alto o álbum já chegou" do que em "onde ele está agora" para prever se vai
  subir - uma mudança na própria relação P(target | X), e não apenas na distribuição de X.
  A queda de importância de `album_age_days` (-48%) também é coerente com E07: no período
  recente, a idade do álbum importa menos para prever ascensão, possivelmente porque
  lançamentos recentes e catálogo antigo têm se comportado de forma mais parecida no
  ranking do que no período histórico.

---

## E11 - Análise de Erros: Falsos Positivos e Falsos Negativos em D2 (modelo MREC)

- **Alteração realizada:** Isolamento dos casos de erro mais severos (maior confiança
  incorreta) do MREC em D2, com exemplos reais de álbuns/artistas.
- **Hipótese:** Falsos negativos (previu queda, mas o álbum subiu) deveriam se concentrar em
  álbuns antigos/reentradas com picos inesperados; falsos positivos deveriam se concentrar em
  lançamentos recentes que já haviam atingido seu pico de popularidade.
- **Resultado:** Total de **3.614 Falsos Positivos** e **15.588 Falsos Negativos** em D2
  (o modelo erra muito mais por excesso de cautela - prevendo queda quando o álbum sobe -
  do que por excesso de otimismo, o que é coerente com o Recall baixo observado na Tabela 17,
  0.18–0.23).
  - **Falsos Positivos mais severos:** *Heathen Chemistry* (Oasis, RE_ENTRY, prob=0.99),
    *Reckless 30th Anniversary* (Bryan Adams, NEW_ENTRY, prob=0.96), *KinKi Single Selection*
    (KinKi Kids, MOVED_DOWN, prob=0.96), álbuns do Exaltasamba e Creep Hyp - **em sua maioria
    reentradas ou lançamentos de catálogo/aniversário** que o modelo esperava continuar
    subindo, mas que na prática já haviam atingido seu pico de interesse.
  - **Falsos Negativos mais severos:** *NEVER SAY NEVER* (ZEROBASEONE), *DAiLY&LOVE*
    (Number_i), *Wuthering Heights* (Charli xcx, NEW_ENTRY, prob=0.02), a trilha sonora de
    *KPop Demon Hunters* (NEW_ENTRY, prob=0.02), *West End Girl* (Lily Allen, NEW_ENTRY,
    prob=0.02) - **predominantemente lançamentos novos ou grupos de K-pop/J-pop** que
    subiram de forma abrupta, um padrão de viralização que o modelo não captura.
- **Interpretação:** A hipótese foi **confirmada de forma clara**. O padrão de falso positivo
  bate com "catálogo/reentrada que não sustenta alta" e o de falso negativo bate com
  "lançamento novo com salto viral" - especialmente evidente no forte viés para K-pop/J-pop
  e para a trilha sonora de *KPop Demon Hunters* (fenômeno de audiência muito acima do normal
  em 2025). Isso indica que o modelo carece de um sinal de "momento cultural"/viralização
  (ex: menções em redes sociais, tendências do TikTok) que não está presente nas variáveis
  puramente derivadas do próprio histórico de ranking.

---

## E12 - Degradação de M0 ao Longo do Tempo: D0 → D1 → D2

- **Alteração realizada:** Os mesmos modelos M0 (treinados uma única vez em D0, sem
  retreinar nada) foram avaliados também em D1, além da avaliação já feita em D2 - permitindo
  observar a trajetória completa de degradação temporal.
- **Hipótese:** O desempenho de M0 deveria cair de forma consistente e monotônica à medida
  que se distancia do período de treino (D0 → D1 → D2).
- **Resultado:**

  | Conjunto | F1 | Variação vs. D0 (validação do tuning) |
  |---|---|---|
  | D0 (validação do tuning, TimeSeriesSplit) | 0.2633 | - |
  | D1 (recente) | 0.2962 ± 0.0058 | **+12.48%** |
  | D2 (futuro) | 0.2783 ± 0.0107 | +5.69% |

  Queda especificamente entre D1 e D2: **-6.04%**.

- **Interpretação:** A hipótese de queda **monotônica** foi **refutada**: M0 na verdade
  performou *melhor* em D1 do que na sua própria validação em D0, e só caiu ao chegar em D2.
  Uma explicação plausível é que a validação em D0 foi feita via `TimeSeriesSplit` (folds
  parciais, com menos dados de treino em cada fold), enquanto a avaliação em D1 usa o modelo
  final treinado com 100% de D0 - naturalmente mais forte. Ainda assim, o padrão relevante
  para a discussão de drift é a **queda entre D1 e D2** (-6.04%), que mostra que a
  degradação real de M0 se manifesta principalmente no conjunto mais distante no tempo,
  reforçando a decisão de tratar D2 como teste cego e justificando por que estratégias de
  atualização (MRT, MREC) trazem ganho relevante especificamente nesse horizonte.

---

## E13 - Verificação de Overfitting: F1 de Treino vs. F1 de Teste (D2)

- **Alteração realizada:** Comparação do F1 de cada estratégia no seu próprio conjunto de
  treino (M0 e MRT-base em D0/D0+D1; MFT e MREC em D1) contra o F1 obtido no teste cego (D2).
- **Hipótese:** Um gap grande entre F1 de treino e F1 de teste indicaria overfitting; espera-
  se que o MREC, com menor volume de dados de treino, seja o mais propenso a esse efeito.
- **Resultado:**

  | Modelo | Conjunto de treino | F1 (Treino) | F1 (Teste D2) | Gap (Treino − Teste) |
  |---|---|---|---|---|
  | M0 | D0 | 0.3244 ± 0.0061 | 0.2783 ± 0.0107 | +0.0462 |
  | MFT | D1 | 0.3152 ± 0.0203 | 0.2781 ± 0.0325 | +0.0371 |
  | **MRT** | D0+D1 | 0.3188 ± 0.0104 | 0.2962 ± 0.0123 | **+0.0226 (menor gap)** |
  | MREC | D1 | 0.3783 ± 0.0202 | 0.3260 ± 0.0148 | **+0.0523 (maior gap)** |

- **Interpretação:** A hipótese foi **confirmada**: o MREC apresenta o maior gap
  treino-teste (0.0523) - o sinal de overfitting mais evidente entre as quatro estratégias,
  coerente com seu menor volume de dados de treino. Curiosamente, é também o modelo com
  **melhor desempenho absoluto em teste** (E07). Isso deve ser destacado com cuidado no
  relatório: o MREC generaliza melhor para D2 apesar de (e não por causa de) seu maior
  overfitting relativo - o fator decisivo parece ser a proximidade temporal de D1 com D2, que
  compensa a menor quantidade de dados. Já o MRT tem o menor gap de todos, sugerindo que
  combinar D0+D1 é a estratégia mais estável/generalizável, mesmo não sendo a de maior F1
  absoluto - um ponto importante para a recomendação final de manutenção do modelo.

---

## Síntese para a Discussão do Relatório (item 11 do edital)

Com base nos experimentos acima, as respostas às perguntas mínimas exigidas seriam:

1. **Quais atributos mudaram entre D0, D1 e D2?** `weeks_on_chart` e `consecutive_weeks`
   (drift severo, PSI > 0.25), seguidos por `peak_rank`, `album_age_days` e `dist_from_peak`
   (drift moderado); `entry_status` também mudou significativamente (Qui² p<0.0001). `rank`
   e `country` permaneceram estáveis (E08, E09).
2. **O desempenho de M0 caiu ao longo do tempo?** Sim, especialmente entre D1 e D2 (-6.04%
   de F1), embora a trajetória completa não seja monotônica (E12).
3. **O fine-tuning melhorou ou prejudicou M0 em D2?** A configuração conservadora (A) foi
   neutra (-0.06%); a agressiva (B) **prejudicou** (E04, E05) - evidência de esquecimento
   catastrófico com learning rate alto.
4. **O retreinamento completo foi superior ao fine-tuning?** Sim, MRT (+6.45% vs. M0) superou
   ambas as configurações de MFT (E06).
5. **Treinar apenas com D1 foi melhor ou pior que usar dados históricos?** **Melhor** - MREC
   foi a estratégia de melhor desempenho absoluto (+17.15% vs. M0), sugerindo concept drift
   real entre os períodos, não apenas data drift (E07, E10).
6. **Há sinais de overfitting durante o fine-tuning?** A Config B de fine-tuning mostrou o
   pior desempenho e maior variância entre seeds, indício de instabilidade/overfitting ao
   padrão de D1 (E05); em termos de gap treino-teste, o MREC é o mais overfitted, mas ainda
   assim o de melhor generalização absoluta (E13).
7. **Quais características do problema explicam os resultados?** A mudança na *mecânica* de
   permanência no ranking (`weeks_on_chart`, `consecutive_weeks`, `entry_status`) e a
   inversão de importância de `rank` para `peak_rank` no SHAP (E10) sugerem uma mudança
   estrutural - possivelmente ligada a hits virais/K-pop com trajetórias atípicas (E11) -
   e não apenas ruído estatístico.
8. **Qual estratégia de manutenção seria recomendada em uma aplicação real?** Considerando
   desempenho absoluto (MREC) e estabilidade/generalização (MRT com menor gap de
   overfitting), a recomendação seria um **retreinamento periódico usando uma janela móvel
   recente** (ex: últimos 12-18 meses, similar ao volume de D1), equilibrando o ganho de
   desempenho do MREC com a maior robustez observada no MRT.

---

> **Observação sobre reprodutibilidade:** todos os experimentos acima são reprodutíveis a
> partir da execução sequencial dos notebooks 01 → 02 → 03 → 04, com as seeds fixas
> documentadas no início deste diário. Nenhum experimento envolveu o uso de D2 para ajuste
> de hiperparâmetros ou seleção de modelo - D2 foi utilizado exclusivamente para a avaliação
> final (Tabela 17), conforme RN12.1.