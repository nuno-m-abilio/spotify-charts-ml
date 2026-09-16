# Backlog do Projeto: AMMCI - Análise sob Mudança Temporal

---

## ÉPICO 1: Dados e Base

### US01: Setup Estrutural, Pipeline de Divisão Temporal e Controle de Data Leakage

* **Responsável:** 
* **Descrição:** Como pesquisador do projeto, quero estruturar o repositório e o pipeline de particionamento temporal para garantir que as bases D0, D1 e D2 sejam geradas sem contaminação futura.
* **Atividades:**
1. Inicializar o repositório Git com `.gitignore` adequado (ignorando a pasta `data/`).
2. Criar a árvore de diretórios padrão (`data/raw`, `data/processed`, `notebooks`, `src`, `docs`).
3. Desenvolver o script/notebook de partição temporal nos limites definidos (~60% D0, ~20% D1, ~20% D2).
4. Exportar e salvar fisicamente os arquivos `d0.csv`, `d1.csv` e `d2.csv`.


* **Regras de Negócio (RN):**
* **RN01.1:** A ordenação dos dados para corte deve ser puramente cronológica pela coluna `date`.
* **RN01.2:** Sob nenhuma hipótese o conjunto D2 pode ser carregado durante etapas de treinamento, ajuste de hiperparâmetros ou cálculo de parâmetros de pré-processamento.
* **RN01.3:** Os arquivos brutos originais não devem ser versionados no Git.


* **Critérios de Aceitação (CA):**
* [ ] O particionamento temporal respeita a proporção aproximada de 60% (D0), 20% (D1) e 20% (D2).
* [ ] Três arquivos CSV distintos (`d0.csv`, `d1.csv`, `d2.csv`) foram gerados na pasta de processados.
* [ ] Repositório inicializado e estruturado com `.gitignore` funcional.



---

### US02: Engenharia de Atributos e Construção do Target

* **Responsável:** 
* **Descrição:** Como cientista de dados, quero construir as variáveis preditivas e definir a variável-alvo para transformar o histórico de rankings em um problema tabular de aprendizado de máquina.
* **Atividades:**
1. Formular e calcular a variável-alvo binária (`target_trend_up`) indicando se o álbum subiu de posição na semana seguinte.
2. Implementar variáveis de diferença temporal (ex: idade do álbum calculada a partir de `release_date`).
3. Criar variáveis de dinâmica de ranking (distância do pico `rank - peak_rank`, variação semanal `rank - previous_rank`, persistência `consecutive_weeks`).
4. Tratar variáveis categóricas (One-Hot Encoding para `country` e `entry_status`).


* **Regras de Negócio (RN):**
* **RN02.1:** A criação de `target_trend_up` deve respeitar o agrupamento por `country` e identificador do álbum (`uri`), garantindo que o cálculo de "semana seguinte" não misture músicas ou países distintos.
* **RN02.2:** Qualquer normalização, padronização ou codificador categórico deve ter seu `fit` realizado estritamente em D0; em D1 e D2 aplica-se apenas `transform`.
* **RN02.3:** Linhas com valores ausentes resultantes do deslocamento temporal (`shift`) no final do ciclo do álbum devem ser descartadas.


* **Critérios de Aceitação (CA):**
* [ ] Pelo menos 6 variáveis explicativas úteis e não redundantes criadas no dataset.
* [ ] Target binário validado e sem valores nulos.
* [ ] Pipeline de transformação exportado ou estruturado para evitar vazamento entre partições.



---

### US03: Limpeza de Dados e Análise Exploratória dos Dados (AED)

* **Responsável:** 
* **Descrição:** Como analista de dados, quero auditar a integridade da base e explorar suas características para identificar desbalanceamento, valores espúrios e padrões de consumo nos países analisados.
* **Atividades:**
1. Realizar auditoria quantitativa da base: contagem de nulos, duplicatas, tipos de dados e cardinalidade.
2. Analisar o balanceamento da variável-alvo no conjunto histórico (D0).
3. Gerar gráficos de distribuição temporal das variáveis numéricas e de contagem para as variáveis categóricas.
4. Documentar hipóteses e observações sobre diferenças de comportamento de consumo entre os 5 países (Brasil, Grã-Bretanha, Japão, Estados Unidos e África do Sul).


* **Regras de Negócio (RN):**
* **RN03.1:** A análise exploratória principal de distribuições deve ser focada no conjunto D0 para evitar enviesamento analítico com base em D2.
* **RN03.2:** Identificadores únicos sem valor semântico (como URLs ou URIs brutas) devem ser desconsiderados como variáveis de entrada dos modelos.


* **Critérios de Aceitação (CA):**
* [ ] Notebook de AED contendo gráficos de distribuição, correlação e contagem de classes.
* [ ] Relatório sucinto de auditoria com contagem de instâncias, nulos tratados e dimensionalidade final.



---

## ÉPICO 2: Modelos Históricos (D0)

### US04: Implementação e Ajuste do Modelo Principal (MLP)

* **Responsável:** 
* **Descrição:** Como engenheiro de aprendizado de máquina, quero implementar e otimizar um Multilayer Perceptron (MLP) em D0 para servir como Modelo Principal atualizável por gradiente.
* **Atividades:**
1. Definir e instanciar a arquitetura do MLP (usando PyTorch ou `MLPClassifier` com suporte a `warm_start`).
2. Implementar validação temporal (ex: `TimeSeriesSplit`) em D0 para ajuste de hiperparâmetros (número de camadas, neurônios, taxa de aprendizado, regularização).
3. Executar o treinamento do modelo histórico base (**M0**) a partir de pesos inicializados aleatoriamente (sem pesos pré-treinados).
4. Executar os treinamentos com pelo menos 3 seeds diferentes para medição de variabilidade (média e desvio-padrão).


* **Regras de Negócio (RN):**
* **RN04.1:** O modelo M0 deve ser treinado única e exclusivamente em D0.
* **RN04.2:** O ajuste de hiperparâmetros não pode utilizar K-Fold aleatório tradicional; deve preservar a ordem temporal dos dados.
* **RN04.3:** O modelo deve suportar salvamento de pesos para continuidade de treinamento nas etapas de fine-tuning.


* **Critérios de Aceitação (CA):**
* [ ] Modelo M0 treinado com sucesso em D0.
* [ ] Métricas históricas em D0 e validação temporal devidamente registradas para 3 seeds.
* [ ] Parâmetros e arquitetura documentados no diário de experimentos.



---

### US05: Implementação do Algoritmo do Zero (Didático)

* **Responsável:** 
* **Descrição:** Como desenvolvedor de algoritmos, quero construir um algoritmo de aprendizado de máquina do zero com NumPy para cumprir o requisito didático de compreensão dos mecanismos de otimização.
* **Atividades:**
1. Escrever o módulo do algoritmo (ex: Regressão Logística) contendo: inicialização de pesos, função sigmoide, função de custo (Binary Cross-Entropy), gradiente descendente e critério de parada.
2. Implementar funções públicas de interface consistentes (`fit`, `predict`, `predict_proba`).
3. Adicionar tratamento de erros e comentários didáticos explicativos no código.
4. Treinar o algoritmo do zero nos dados de treino de D0.


* **Regras de Negócio (RN):**
* **RN05.1:** O algoritmo não pode importar nenhum estimador pronto de bibliotecas de ML (como scikit-learn); apenas NumPy e Pandas para manipulação matemática e de matrizes.
* **RN05.2:** O código deve estar isolado em arquivo executável/módulo dentro de `src/` ou em célula limpa no notebook.


* **Critérios de Aceitação (CA):**
* [ ] Código do algoritmo implementado funcionalmente com convergência do gradiente demonstrada.
* [ ] Métodos de inicialização, treinamento e predição implementados e testados com matrizes de D0.



---

### US06: Modelos de Baseline e Comparação com Biblioteca

* **Responsável:** 
* **Descrição:** Como avaliador de modelos, quero treinar um modelo de biblioteca comparável e um modelo baseado em árvores para validar a corretude da implementação do zero e estabelecer um baseline competitivo.
* **Atividades:**
1. Instanciar e treinar o modelo equivalente da biblioteca (ex: `LogisticRegression` do scikit-learn) com hiperparâmetros semelhantes aos da US05.
2. Comparar desempenho, tempo de execução e convergência entre a versão do zero e a de biblioteca.
3. Treinar uma terceira família de modelo (ex: Random Forest ou LightGBM) em D0 como baseline forte.
4. Consolidar o comparativo inicial de famílias em D0.


* **Regras de Negócio (RN):**
* **RN06.1:** A comparação entre a biblioteca e a implementação do zero deve usar rigorosamente as mesmas matrizes de treino e validação.
* **RN06.2:** O total de famílias de modelos treinadas no projeto deve atender ao requisito de pelo menos $n$ famílias (onde $n$ é o número de integrantes da equipe = 3).


* **Critérios de Aceitação (CA):**
* [ ] Comparação documentada entre a implementação própria e a implementação de biblioteca (tempo, acurácia, convergência).
* [ ] Terceiro modelo de biblioteca (ex: árvores) treinado e avaliado em D0.



---

## ÉPICO 3: Drift e Atualização Temporal

### US07: Pipelines de Atualização MFT (Fine-Tuning) e MREC (Apenas Recente)

* **Responsável:** 
* **Descrição:** Como pesquisador de adaptação de modelos, quero executar as estratégias de Fine-Tuning e Treinamento Recente para comparar adaptação contínua contra esquecimento catastrófico.
* **Atividades:**
1. Carregar os pesos prévios de M0 e aplicar continuidade de treinamento utilizando exclusivamente o conjunto D1 (**MFT**).
2. Implementar e testar pelo menos duas configurações distintas de fine-tuning (variando learning rate, número de épocas ou regularização).
3. Inicializar uma nova instância da mesma arquitetura do zero e treiná-la exclusivamente em D1 (**MREC**).
4. Executar os treinamentos com 3 seeds e registrar métricas e tempos de execução.


* **Regras de Negócio (RN):**
* **RN07.1:** MFT não pode reiniciar pesos; deve partir estritamente do estado final de M0.
* **RN07.2:** MREC deve ter pesos inicializados aleatoriamente e não pode ver nenhuma linha de D0.
* **RN07.3:** A arquitetura do modelo deve ser rigorosamente idêntica à de M0 para assegurar comparabilidade.


* **Critérios de Aceitação (CA):**
* [ ] Duas variantes de MFT executadas e documentadas em D1.
* [ ] Modelo MREC treinado exclusivamente em D1 com 3 seeds.
* [ ] Pesos de M0, MFT e MREC salvos para avaliação posterior.



---

### US08: Pipeline de Retreinamento Completo (MRT)

* **Responsável:** 
* **Descrição:** Como engenheiro de dados, quero implementar o retreinamento completo do modelo utilizando os dados acumulados para servir de padrão de comparação com o fine-tuning.
* **Atividades:**
1. Construir a rotina de união cronológica de D0 e D1 ($D0 \cup D1$).
2. Inicializar o modelo MLP do zero (pesos aleatórios) com a mesma arquitetura de M0.
3. Treinar o modelo completo (**MRT**) sobre a base combinada $D0 + D1$.
4. Repetir o treinamento para as 3 seeds, registrando tempo total de processamento e curvas de perda.


* **Regras de Negócio (RN):**
* **RN08.1:** MRT deve ter os pesos reinicializados; não pode aproveitar pesos de treinamentos anteriores.
* **RN08.2:** As mesmas transformações de pré-processamento aplicadas em D0 devem ser estendidas a D0+D1 de forma consistente.


* **Critérios de Aceitação (CA):**
* [ ] Modelo MRT treinado com sucesso no conjunto acumulado D0 + D1.
* [ ] Tempos de execução e métricas de convergência registrados para as 3 seeds.



---

### US09: Investigação e Quantificação de Data Drift e Concept Drift

* **Responsável:** 
* **Descrição:** Como analista estatístico, quero mensurar as mudanças na distribuição dos atributos e do target ao longo do tempo para fundamentar se as variações de desempenho decorrem de data drift ou concept drift.
* **Atividades:**
1. Aplicar testes quantitativos de distribuição (Kolmogorov-Smirnov para numéricas e Qui-Quadrado ou PSI para categóricas) comparando D0 vs D1 e D0 vs D2.
2. Identificar e listar os atributos que sofreram maior alteração distributiva ao longo do tempo.
3. Analisar a distribuição temporal da variável-alvo (`target_trend_up`) entre D0, D1 e D2.
4. Correlacionar visualmente as quedas de desempenho observadas com os desvios de atributos calculados.


* **Regras de Negócio (RN):**
* **RN09.1:** A quantificação de drift deve ser feita por métodos estatísticos formalmente reconhecidos, não apenas por inspeção visual.
* **RN09.2:** Diferenciar explicitamente no relatório eventuais alterações na distribuição de $X$ (data drift) de alterações na relação $P(Y\vert{}X)$ (concept drift).


* **Critérios de Aceitação (CA):**
* [ ] Tabela comparativa com valores estatísticos (p-valor do KS ou valor do PSI) para as principais features entre os blocos temporais.
* [ ] Discussão formal identificando quais variáveis sofreram drift estatisticamente significativo.



---

## ÉPICO 4: Pós-Processamento e Avaliação Final

### US10: Revisão de Reprodutibilidade, Diário de Experimentos e README

* **Responsável:** 
* **Descrição:** Como mantenedor do projeto, quero padronizar o repositório, documentar as instruções de execução e registrar o diário de experimentos para garantir a reprodutibilidade integral do trabalho.
* **Atividades:**
1. Estruturar o `README.md` com guia passo a passo de reprodução do pipeline, requisitos e declaração de uso de IA generativa.
2. Congelar as versões de dependências em um arquivo `requirements.txt`.
3. Consolidar o Diário de Experimentos contendo no mínimo 8 execuções relevantes (com alteração realizada, hipótese, resultado e interpretação).
4. Verificar a integridade e legibilidade de todos os notebooks e scripts.


* **Regras de Negócio (RN):**
* **RN10.1:** Todas as sementes aleatórias (seeds) utilizadas devem estar explícitas no código e no README.
* **RN10.2:** O diário deve conter pelo menos 8 registros de experimentos técnicos válidos, não triviais.


* **Critérios de Aceitação (CA):**
* [ ] `requirements.txt` gerado e testado em ambiente limpo.
* [ ] `README.md` detalhado com instruções claras e tabela do diário de 8 experimentos preenchida.



---

### US11: Explicabilidade do Modelo (SHAP) e Análise de Erros

* **Responsável:** 
* **Descrição:** Como cientista de dados, quero aplicar métodos de explicabilidade e inspecionar erros para interpretar a lógica de decisão dos modelos e entender as causas de falsos positivos e negativos.
* **Atividades:**
1. Aplicar a biblioteca SHAP (ex: `TreeExplainer` ou `KernelExplainer`/`DeepExplainer`) para extrair a importância dos atributos em M0 e no modelo atualizado.
2. Comparar se as variáveis mais determinantes mudaram entre o período histórico e o período recente.
3. Realizar a análise de instâncias de erro: isolar e analisar os casos extremos de falsos positivos e falsos negativos.
4. Redigir uma síntese qualitativa explicando por que certos álbuns geraram erros de predição severos.


* **Regras de Negócio (RN):**
* **RN11.1:** A análise de explicabilidade deve comparar o comportamento entre o modelo antes e depois da atualização temporal.
* **RN11.2:** A análise de erros deve contemplar exemplos reais da base (nomes de álbuns/artistas reais presentes nos dados).


* **Critérios de Aceitação (CA):**
* [ ] Gráficos de importância global e local gerados via SHAP (summary plot, force plot ou bar plot).
* [ ] Seção de análise de erros documentando padrões observados em previsões incorretas.



---

### US12: Avaliação Cega em D2 e Consolidação das Métricas Finais

* **Responsável:** 
* **Descrição:** Como auditor de qualidade, quero executar a avaliação final das quatro estratégias (M0, MFT, MRT, MREC) no conjunto D2 para gerar a tabela comparativa definitiva exigida pelo edital.
* **Atividades:**
1. Carregar o conjunto D2 (futuro) e aplicar as transformações já ajustadas.
2. Submeter D2 aos quatro modelos: M0, MFT (melhor variante), MRT e MREC.
3. Calcular as métricas exigidas: Accuracy, Precision, Recall, F1-Score, Matriz de Confusão e PR-AUC.
4. Calcular as médias e desvios-padrão entre as 3 seeds executadas e calcular a variação percentual relativa a M0.
5. Construir a Tabela 17 do edital consolidando todas as estratégias em D2.


* **Regras de Negócio (RN):**
* **RN12.1:** D2 só é aberto e processado nesta etapa, sem que haja qualquer retreinamento ou alteração de hiperparâmetros após a observação dos resultados.
* **RN12.2:** Todas as métricas devem reportar média $\pm$ desvio-padrão derivados das seeds estocásticas.


* **Critérios de Aceitação (CA):**
* [ ] Tabela consolidada contendo M0, MFT, MRT e MREC avaliados em D2.
* [ ] Matrizes de confusão e métricas completas calculadas e salvas.



---

## ÉPICO 5: Relatório no Formato SBC (Overleaf)

### US13: Escrita da Seção de Metodologia e Protocolo Experimental

* **Responsável:** 
* **Descrição:** Como autor do relatório, quero documentar a arquitetura dos modelos, a formulação matemática e o protocolo dos experimentos temporais para cumprir a Seção 3 do short paper.
* **Atividades:**
1. Escrever o detalhamento do particionamento cronológico D0/D1/D2 e justificar as proporções adotadas.
2. Documentar a arquitetura do Modelo Principal (camadas, funções de ativação, otimizador, learning rate).
3. Descrever as regras dos experimentos M0, MFT, MRT e MREC.
4. Redigir o Apêndice de "Uso de IA Generativa" detalhando ferramentas, finalidades e validações executadas.


* **Regras de Negócio (RN):**
* **RN13.1:** Seguir rigorosamente o template LaTeX da Sociedade Brasileira de Computação (SBC).
* **RN13.2:** Declarar explicitamente a utilização de ferramentas de IA generativa no apêndice conforme exigência do item 14 da especificação.


* **Critérios de Aceitação (CA):**
* [ ] Seção "Materiais e Métodos" concluída no Overleaf.
* [ ] Apêndice de IA generativa preenchido em conformidade com o edital.



---

### US14: Escrita das Seções de Resultados e Discussão

* **Responsável:** 
* **Descrição:** Como autor do relatório, quero redigir as seções de Resultados e Discussão para detalhar o comportamento dos modelos sob drift e interpretar a análise de explicabilidade e erros.
* **Atividades:**
1. Inserir e descrever a Tabela Comparativa de D2 no formato exigido pelo edital.
2. Descrever a análise de explicabilidade SHAP e contrastar a relevância dos atributos entre modelos.
3. Discutir as hipóteses de degradação temporal, respondendo se o retreinamento superou o fine-tuning e discutindo overfitting.
4. Inserir e debater os casos de falsos positivos e negativos identificados na US11.


* **Regras de Negócio (RN):**
* **RN14.1:** Todos os dados numéricos do texto devem coincidir estritamente com os valores consolidados nos experimentos.
* **RN14.2:** A discussão deve responder diretamente às perguntas mínimas do item 11 da especificação (sinais de overfitting, fine-tuning vs retreinamento, recomendação prática).


* **Critérios de Aceitação (CA):**
* [ ] Seções "Resultados" e "Discussão" redigidas no Overleaf com tabelas e gráficos em formato vetorial/alta resolução.
* [ ] Interpretação técnica profunda conectando as métricas com o domínio de streaming musical.



---

### US15: Escrita de Introdução, Fundamentação Teórica e Conclusão

* **Responsável:** 
* **Descrição:** Como autor do relatório, quero estruturar a narrativa inicial, a base conceitual e o desfecho do artigo para situar o leitor e responder à hipótese do projeto.
* **Atividades:**
1. Redigir a "Introdução" contextualizando o domínio de streaming, a volatilidade temporal e a hipótese do trabalho.
2. Redigir a "Fundamentação Teórica" definindo: Data Drift vs Concept Drift, Fine-Tuning, Retreinamento Contínuo e métricas de classificação sob desbalanceamento.
3. Escrever as "Conclusões" respondendo formalmente se a hipótese inicial foi confirmada ou refutada e recomendando uma estratégia de manutenção para produção.
4. Compilar e formatar as referências bibliográficas no padrão BibTeX/SBC.


* **Regras de Negócio (RN):**
* **RN15.1:** A hipótese enunciada na Introdução deve ser rigorosamente a mesma cadastrada no registro inicial do Classroom.
* **RN15.2:** O short paper não deve exceder o limite de páginas estipulado pelas diretrizes da disciplina/SBC.


* **Critérios de Aceitação (CA):**
* [ ] Seções de Introdução, Fundamentação Teórica e Conclusões integradas no Overleaf.
* [ ] Artigo compilando perfeitamente em PDF sem erros de sintaxe LaTeX ou referências quebradas.



---

## ÉPICO 6: Apresentação e Slides (PDF)

### US16: Elaboração dos Slides de Metodologia, Arquitetura e Engenharia

* **Responsável:** 
* **Descrição:** Como apresentador do projeto, quero elaborar os slides relativos à modelagem técnica e aos experimentos de fine-tuning para comunicar a complexidade do pipeline durante a defesa oral.
* **Atividades:**
1. Criar o slide de apresentação institucional (Universidade, departamento, docente, integrantes, RAs, data).
2. Desenvolver os slides de Materiais e Métodos (diagrama do split D0/D1/D2 e fluxo temporal).
3. Desenvolver os slides de Desenvolvimento (arquitetura do MLP, estratégias de Fine-Tuning MFT vs MRT vs MREC).
4. Estruturar a síntese de decisões de engenharia de software e versionamento.


* **Regras de Negócio (RN):**
* **RN16.1:** Os slides devem priorizar diagramas visuais e tabelas legíveis, evitando blocos densos de texto.
* **RN16.2:** Seguir a ordem de tópicos definida no item 18 da especificação.


* **Critérios de Aceitação (CA):**
* [ ] Slides de abertura, metodologia e desenvolvimento prontos e revisados.



---

### US17: Elaboração dos Slides do Algoritmo do Zero, Resultados e SHAP

* **Responsável:** 
* **Descrição:** Como apresentador do projeto, quero preparar o material visual relativo ao algoritmo implementado do zero, análise de erros e explicabilidade para demonstrar domínio prático na apresentação.
* **Atividades:**
1. Elaborar slide explicativo do Algoritmo do Zero (equações implementadas, convergência e comparação com scikit-learn).
2. Estruturar os slides com os gráficos de importância de atributos SHAP.
3. Preparar slides demonstrativos de análise de erros reais (exemplos de álbuns que enganaram o modelo).
4. Revisar as explicações técnicas visando a preparação para perguntas da banca/professor.


* **Regras de Negócio (RN):**
* **RN17.1:** As figuras do SHAP e matrizes de confusão devem ter resolução suficiente para leitura projetada.


* **Critérios de Aceitação (CA):**
* [ ] Slides do algoritmo do zero, explicabilidade e erros concluídos e integrados ao arquivo mestre.



---

### US18: Elaboração dos Slides de Abertura, Análise de Drift, Conclusão e Fechamento do PDF

* **Responsável:** 
* **Descrição:** Como apresentador e consolidador final, quero elaborar os slides de contexto, análise estatística de drift, fechamento e gerar o arquivo final em PDF.
* **Atividades:**
1. Criar os slides de Introdução e contextualização do problema no Spotify e hipótese de trabalho.
2. Desenvolver o slide de Fundamentação (conceitos de Drift e Métricas).
3. Montar os slides de resultados quantitativos de Data Drift (tabela de testes KS/PSI).
4. Elaborar os slides de Resultados Finais (Tabela consolidada em D2), Conclusões e Referências.
5. Consolidar todos os slides dos três membros em um único arquivo PDF final formatado.


* **Regras de Negócio (RN):**
* **RN18.1:** O arquivo final entregável deve ser estritamente em formato PDF.
* **RN18.2:** O layout visual deve ser uniforme em tipografia, cores e proporções em todos os slides da apresentação.


* **Critérios de Aceitação (CA):**
* [ ] Apresentação completa finalizada cobrindo todos os itens do roteiro da Seção 18.
* [ ] Arquivo PDF exportado, revisado e validado pela equipe antes do ensaio da defesa.