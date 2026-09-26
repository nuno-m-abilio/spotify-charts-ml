# Diário de Experimentos - AMMCI

Este documento registra as principais iterações do ciclo de modelagem, conforme o Item 13 do edital. Os valores finais nas hipóteses dependem da execução de testes, mas refletem a lógica de decisão do projeto.

| Exp. | Alteração Realizada | Hipótese | Resultado | Interpretação |
| :--- | :--- | :--- | :--- | :--- |
| **E01** | Teste de baseline clássico (`LogisticRegression` em D0). | Modelos lineares não conseguirão capturar a complexidade temporal da persistência de álbuns no topo. | F1-Score: Moderado. | Baseline linear estabelecido como piso. Rejeitamos uso final. |
| **E02** | Treinamento da implementação manual `LogisticRegressionCustom`. | O algoritmo próprio convergirá para resultados $\approx$ ao Scikit-learn. | Métricas pareadas (diferença < 1%). | Algoritmo validado matematicamente. Atende requisito didático. |
| **E03** | Teste de Baseline Baseado em Árvore (Random Forest). | Modelos de particionamento (RF) superarão os modelos paramétricos (LR) em $D_0$. | F1-Score: Alto. | Confirmado. Árvores lidam melhor com variáveis não linearizadas (`album_age_days`). |
| **E04** | Treinamento do Modelo Principal (MLP $M_0$) em $D_0$. | O MLP (64x32 neurônios) criará abstrações suficientes para generalização futura. | Estabilizado no Treino. | Arquitetura oficializada para os testes de Fine-Tuning. |
| **E05** | Avaliação cega de $M_0$ na partição recente ($D_1$). | A mudança de consumo pós-2023 degradará as métricas do modelo treinado na era pandêmica. | Queda no F1-Score e PR-AUC. | Hipótese confirmada. Concept Drift validado via SHAP/Performance. |
| **E06** | Fine-Tuning ($M_{FT}$) aplicando apenas $D_1$ sobre os pesos de $M_0$. | Ajustes marginais nos pesos farão o modelo recuperar métricas sem perder o histórico. | Recuperação moderada de F1-Score. | Mantido. Demonstra resiliência à degradação contínua. |
| **E07** | Retreinamento Completo ($M_{RT}$) em $D_0 \cup D_1$. | Refazer a otimização global em todos os dados gerará o melhor modelo. | Melhor métrica global. | Aceito. O custo computacional compensa o ganho estatístico. |
| **E08** | Treinamento estrito em dados recentes ($M_{REC}$ em $D_1$). | Dados velhos ($D_0$) são ruído; treinar apenas no recente seria superior. | Degradação severa. | Rejeitado. O volume reduzido de $D_1$ causa subajuste em redes neurais. O passado importa. |