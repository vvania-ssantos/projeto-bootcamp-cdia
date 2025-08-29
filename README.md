Projeto de Manutenção Preditiva

O objetivo deste projeto é criar um sistema de manutenção preditiva para identificar e prever falhas em máquinas industriais, utilizando dados de sensores. A análise e a modelagem foram realizadas seguindo a metodologia CRISP-DM.

Preparação dos Dados
Os dados foram limpos, valores inconsistentes foram unificados e valores faltantes nas colunas numéricas foram preenchidos com a mediana.

Colunas irrelevantes para o modelo, como id_produto, foram removidas.

Modelagem
Um modelo de Decision Tree Classifier foi utilizado para fazer as previsões.

Para tratar o desbalanceamento dos dados, utilizei a técnica SMOTE para balancear o conjunto de treino.

Avaliação e Resultados
A acurácia do modelo foi de 98%.

O recall para a classe de falha (classe 1) foi de 73%, o que demonstra que o modelo é eficaz em identificar a maioria das falhas que realmente ocorrem.

A precisão para a classe de falha foi de 41%, indicando que 41% das previsões de falha estavam corretas.

Em um cenário de manutenção preditiva, a alta taxa de recall é crucial, pois é mais importante identificar falhas potenciais (mesmo com alguns falsos positivos) do que deixar uma falha real passar despercebida. A aplicação do SMOTE foi fundamental para alcançar este resultado, e a acurácia, embora alta, deve ser analisada com cuidado devido ao desbalanceamento do conjunto de dados.









