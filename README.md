WINE QUALITY CLASSIFICATION
Tech Challenge - Fase 2 | POSTECH Data Analytics

Pipeline completa de Machine Learning para classificar a qualidade de vinhos a
partir de suas caracteristicas fisico-quimicas, transformando o problema em uma
classificacao binaria:

Alta Qualidade    - nota maior ou igual a 7
Baixa/Media Qualidade - nota menor que 7


RESULTADO PRINCIPAL

Modelo                 ROC-AUC   Acuracia   Precisao   Recall   F1
Random Forest (melhor)  0,916     0,90       0,71       0,44     0,55
Gradient Boosting       0,886     0,88       0,57       0,44     0,50
Regressao Logistica     0,884     0,79       0,36       0,74     0,48

O Random Forest foi o melhor modelo geral, com maior ROC-AUC e precisao. A
Regressao Logistica se destaca no recall, o que e util quando o objetivo e nao
deixar passar nenhum vinho premium.

Variaveis mais influentes na qualidade:
1. Sulfatos (conservacao/antioxidacao)
2. Acidez volatil (quanto menor, melhor - evita o defeito "avinagrado")
3. Alcool (maior teor tende a melhor avaliacao)


ESTRUTURA DO REPOSITORIO

wine-quality-classification/
  data/
    WineQT.csv                    - base de dados (Wine Quality Dataset)
  notebooks/
    wine_quality_analysis.ipynb   - notebook completo (EDA + modelagem)
  src/
    preprocessing.py              - carga, binarizacao, dedup, feature engineering
    eda.py                        - analise exploratoria (gera os graficos)
    train.py                      - treino, avaliacao e comparacao dos modelos
  results/
    *.png                         - graficos da EDA e da modelagem
    model_comparison.csv          - tabela comparativa de metricas
    feature_importance.csv        - importancia das variaveis
    classification_report.txt     - relatorio do melhor modelo
    metrics_summary.json          - resumo consolidado em JSON
  apresentacao_executiva.pdf      - storytelling executivo da analise
  requirements.txt
  README.txt


COMO EXECUTAR

1. Criar ambiente e instalar dependencias
   pip install -r requirements.txt

2. Rodar a EDA (gera os graficos em results/)
   cd src
   python eda.py

3. Treinar e avaliar os modelos
   python train.py

Ou abrir o notebook completo:
   jupyter notebook notebooks/wine_quality_analysis.ipynb


METODOLOGIA

1. Compreensao do problema - binarizacao da nota de qualidade.
2. EDA - distribuicoes, correlacoes justificadas, deteccao de outliers (IQR),
   analise de balanceamento (~14% de positivos, classe desbalanceada).
3. Pre-processamento - sem nulos; remocao de 125 duplicatas (evita leakage);
   padronizacao via StandardScaler dentro do pipeline; 4 novas features com
   justificativa enologica.
4. Modelagem - 3 algoritmos com split estratificado e class_weight='balanced'.
5. Avaliacao - ROC-AUC, F1, precisao, recall, matriz de confusao e validacao
   cruzada estratificada (5 folds).
6. Interpretacao - permutation importance e recomendacoes para producao.


DECISOES TECNICAS RELEVANTES

- Duplicatas removidas considerando apenas as features (nao o Id), para evitar
  que a mesma amostra apareca em treino e teste (vazamento de dados).
- Outliers mantidos, pois representam variacao fisica real; modelos de arvore
  sao robustos a eles.
- Metricas alem da acuracia: com classes desbalanceadas, acuracia e enganosa;
  o foco e ROC-AUC e F1.
- Scaler dentro do pipeline: a padronizacao e ajustada apenas no treino.


DATASET

Wine Quality Dataset (Kaggle) - 1.143 amostras, 11 variaveis fisico-quimicas e
nota de qualidade. Apos limpeza: 1.018 amostras unicas.
