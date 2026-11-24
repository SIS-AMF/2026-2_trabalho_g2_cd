# Planejamento do Trabalho de Ciência de Dados

## 1. O que o professor pediu (Enunciado)

> Entregar em 26 de novembro um Colab ou repositório contendo a aplicação em um conjunto de dados (livre escolha):
> 1. O **Pipeline de pré-processamento de dados incluindo engenharia de recursos** *(mesmo que o modelo escolhido posteriormente não necessite de algumas técnicas, demonstre-as mesmo assim).*
> 2. Uma **rotina adequada de processamento dos dados**, incluindo **validação cruzada**, **ajuste dos hiperparâmetros** e **avaliação do modelo** *(visto em sala).*

**Regra de ouro deste trabalho:** manter tudo **simples, objetivo e 100% fiel ao que foi apresentado em aula** (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap` e `joblib`). Todo o trabalho será entregue em **um único notebook limpo e bem explicado**.

---

## 2. Traduzindo o pedido com base nas aulas

O trabalho reproduz no nosso dataset exatamente o fluxo ensinado nos materiais de `class_material/`:

| Parte do Trabalho | O que precisa conter | Referência direta nas aulas (`class_material/`) |
| :--- | :--- | :--- |
| **Parte 1 — Pré-processamento e Engenharia de Recursos** | Limpeza enxuta, criação de novas features, demonstração curta das técnicas de pré-processamento da aula e montagem do `Pipeline` + `ColumnTransformer` sem *data leakage*. | `pre.pdf`, `Preechimento_de_dados.ipynb`, `Pipeline_de_Pre_Processamento.ipynb` |
| **Parte 2 — Rotina de Processamento e Avaliação** | Split estratificado, baseline, comparação de 4 classificadores com `StratifiedKFold` e `GridSearchCV` (otimizando `roc_auc`), avaliação no teste, Nested CV (IC 95%), SHAP e salvamento do modelo final. | `aula_09.pdf`, `Supervisionados_01.ipynb`, `Árvore_de_Decisão.ipynb`, `Pepeline_processaemento.ipynb` |

> **Materiais fora de escopo:** `Pre_Imagens.ipynb` (imagens) e `Kmeans.ipynb` (não supervisionado — usado apenas na demonstração de discretização com `KBinsDiscretizer(strategy='kmeans')`).

---

## 3. Decisões Fechadas do Projeto (Escopo Definitivo)

| Decisão | Definição Escolhida | Justificativa |
| :--- | :--- | :--- |
| **Problema** | **Classificação Binária** de Recomendação (*Cross-Sell*) no nível do pedido | Simples, 100% compatível com o código da aula e permite recomendar via `.predict_proba()`. |
| **Alvo (`y`)** | **Família de produto** mais frequente em compras combinadas (`1` = levou, `0` = não levou) | Mais exemplos positivos (`y = 1`), folds mais estáveis na validação cruzada e leitura clara para negócio. |
| **Feature `evento_nome`** | **Incluída em `X`**, agrupando feiras raras em `"Outros"` | Permite demonstrar agrupamento de categorias raras, `OneHotEncoder` e `TargetEncoder` vistos em aula. |
| **Imputação (base sem nulos)** | `SimpleImputer` no `Pipeline` + **1 célula didática simulando ~5% de nulos numa cópia** | Cumpre a exigência de demonstrar a técnica na prática (como no exercício de `Preechimento_de_dados.ipynb`) sem alterar a base oficial. |
| **Modelos comparados** | **Regressão Logística**, **KNN**, **Random Forest** e **XGBoost** | Cobre 3 famílias ("tribos") da aula (linear, distância e árvores/boosting), justificando todo o pré-processamento. |
| **Método de Tuning** | **`GridSearchCV`** com grades enxutas | Mais simples, rápido (inclusive no Nested CV) e direto de explicar na apresentação. |
| **Métrica Principal** | **`roc_auc`** (no CV, tuning e Nested CV) | Avalia a qualidade das probabilidades de recomendação em todos os limiares, sem depender de um único corte fixo. |

### Dataset e Regras de Limpeza (`dataset.csv`)
- **Base bruta:** ~3.815 registros de itens vendidos em feiras (~3.003 pedidos), 13 colunas originais.
- **Limpeza enxuta:**
  1. Remover colunas irrelevantes: `pedido_vendedor`, `item_pedido_user_id` (apenas 2 logins genéricos de caixa) e `pedido_tipo_desconto` (após confirmar que os descontos já estão em R$).
  2. Remover as 2 linhas com `item_pedido_quantidade <= 0`.
  3. Padronizar textos simples (remover espaços extras em produtos/eventos e unificar `"Pix"`/`"PIX"` em `"PIX"`).
  4. Extrair **`produto_base`**, **`familia_produto`** e **`multiplicador`** (ex.: pacote x3, bandeja x5, unidade x1) e corrigir os erros de preço/quantidade em pacotes para garantir `qtd * preco == subtotal`.
  5. Filtrar poucos *outliers* operacionais óbvios (pedidos gratuitos com valor total `<= 0` e fechamentos consolidados de caixa muito fora do padrão de varejo).

### Regras Obrigatórias Anti-*Data Leakage*
1. **Filtrar pedidos exclusivos da família alvo:** manter na base de modelagem apenas pedidos que contenham **pelo menos 1 item de outra família** (`qtd_outros_itens >= 1`). Sem isso, pedidos que só têm a família alvo ficariam com todas as outras colunas zeradas, entregando `y = 1` trivialmente.
2. **Features de cesta calculadas sem o alvo:** `qtd_outros_itens`, `qtd_unidades_outros`, `valor_outros_itens`, `eh_pacote_outros` e flags das outras famílias são calculados **exclusivamente sobre os demais itens do pedido**.
3. **Divisão antes de qualquer ajuste estatístico:** executar `train_test_split` antes de qualquer `fit`/`transform` do pré-processamento.

---

## 4. Passo a Passo do Trabalho (Estrutura do Notebook)

### Etapa 1 — Carregamento, Inspeção Rápida e Limpeza Básica
*Referência: `Pipeline_de_Pre_Processamento.ipynb` e `pre.pdf`.*
- Carregar o CSV e inspecionar `.shape`, `.info()`, `.isnull().sum()` e `.describe()`.
- Aplicar a limpeza enxuta da Seção 3 (remover colunas inúteis, padronizar textos, extrair `produto_base`, `familia_produto` e `multiplicador`, reconciliar valores e remover pedidos inválidos/fechamentos de caixa).
- Conferir qual **família de produto** aparece com mais frequência em pedidos com 2+ produtos e fixá-la como o alvo binário (`y`).

### Etapa 2 — Engenharia de Recursos (*Feature Engineering*) e Base por Pedido
*Referência: `Pipeline_de_Pre_Processamento.ipynb` (Seção "Engenharia de atributos") e `pre.pdf`.*
Consolidar a base para **1 linha por pedido** (mantendo apenas `qtd_outros_itens >= 1`) com features simples e explicáveis:
1. **Features de Tempo** (de `pedido_data`):
   - `hora`, `dia_semana`, `mes`, `eh_fim_de_semana` (`0` ou `1`).
   - **Codificação cíclica da hora:** `hora_sin = sin(2 * pi * hora / 24)` e `hora_cos = cos(2 * pi * hora / 24)`.
   - **Discretização personalizada:** `turno` (`Manhã`, `Tarde`, `Noite`).
2. **Features da Cesta (somente sobre os outros itens, excluindo a família alvo):**
   - `qtd_outros_itens`: quantidade de SKUs/itens distintos de outras famílias no pedido.
   - `qtd_unidades_outros`: soma de unidades base (`quantidade * multiplicador`) dos outros itens.
   - `valor_outros_itens`: valor monetário acumulado dos outros itens já no carrinho.
   - `eh_pacote_outros`: flag binária (`1` se já há algum item com `multiplicador > 1`).
   - Flags binárias das **demais famílias de produto** presentes no pedido (ex.: `tem_cuca`, `tem_rapadura`, `tem_amendoim`, `tem_bolacha` — **exceto a família alvo**).
3. **Contexto da venda:** `pedido_metodo_pagamento` e `evento_nome`.

### Etapa 3 — Demonstração Curta das Técnicas de Pré-processamento da Aula
*Referência: `Pipeline_de_Pre_Processamento.ipynb`, `Preechimento_de_dados.ipynb` e `pre.pdf`.*

> **Disciplina de simplicidade:** limitar cada técnica abaixo a **1 célula curta e objetiva** (demonstrações feitas sobre cópias quando alteram dados, sem bagunçar a base oficial).

1. **Detecção de Outliers:** Boxplot/IQR (Tukey), Z-score ($\mu \pm 3\sigma$) e `IsolationForest` em `valor_outros_itens` / `qtd_unidades_outros`, justificando o tratamento aplicado aos fechamentos de caixa.
2. **Imputação de Dados Ausentes (célula didática):** mostrar que a base limpa não possui nulos, resumir MCAR/MAR/MNAR e simular **~5% de `NaN` em uma cópia** de `valor_outros_itens`, comparando o erro (`MAE`) entre `SimpleImputer(strategy='median')` e `IterativeImputer` (como no exercício de `Preechimento_de_dados.ipynb`).
3. **Discretização (Binning):** mostrar `turno` (regra de negócio) e comparar brevemente `KBinsDiscretizer` (`uniform`, `quantile`, `kmeans`) em `valor_outros_itens`.
4. **Codificação de Categóricas (Encoding):** demonstrar `OneHotEncoder`, `OrdinalEncoder` (`turno`: `['Manhã', 'Tarde', 'Noite']`), agrupamento de feiras raras (`< 5%`) em `"Outros"` e `TargetEncoder` (demonstrativo em `evento_nome`).
5. **Correção de Assimetria (*Skewness*):** calcular `.skew()` de `valor_outros_itens` e `qtd_unidades_outros`, mostrando histograma + Q-Q plot antes e depois de `np.log1p` e `PowerTransformer(method='yeo-johnson')`.
6. **Escalonamento (*Scaling*):** comparar brevemente `StandardScaler`, `MinMaxScaler` e `RobustScaler` em `valor_outros_itens` e `qtd_unidades_outros`.
7. **Seleção de Características:**
   - **Multicolinearidade:** matriz de correlação (`sns.heatmap`) removendo redundâncias com $r > 0.9$.
   - **Variância Quase Nula:** checagem com `VarianceThreshold` (limiar 1%–5%).
8. **Redução de Dimensionalidade (PCA):** aplicar `StandardScaler` + `PCA` nas numéricas, plotar variância explicada acumulada (90%/95%) e projeção 2D (`PC1` vs `PC2`), justificando por que o PCA fica apenas demonstrativo para preservar a interpretabilidade das features no SHAP.

### Etapa 4 — Split Estratificado e `Pipeline` + `ColumnTransformer` (Sem *Data Leakage*)
*Referência: `Pipeline_de_Pre_Processamento.ipynb` e `Pepeline_processaemento.ipynb` (Seção 1).*
1. **Divisão Treino/Teste antes de qualquer `fit`/`transform`:**
   - `train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)`.
2. **Grupos de colunas por tipo de tratamento no `ColumnTransformer`:**
   - `skewed_features` (`valor_outros_itens`, `qtd_unidades_outros`): `SimpleImputer('median')` $\rightarrow$ `PowerTransformer('yeo-johnson')` $\rightarrow$ `StandardScaler()`.
   - `numeric_features` (`qtd_outros_itens`, `hora_sin`, `hora_cos`, `mes`, `dia_semana`): `SimpleImputer('median')` $\rightarrow$ `StandardScaler()`.
   - `binary_features` (`eh_fim_de_semana`, `eh_pacote_outros`, flags `tem_*`): `SimpleImputer('most_frequent')` (mantém `0`/`1`, preservando a leitura direta no SHAP).
   - `ordinal_features` (`turno`): `SimpleImputer('most_frequent')` $\rightarrow$ `OrdinalEncoder(categories=[['Manhã', 'Tarde', 'Noite']])`.
   - `categorical_features` (`pedido_metodo_pagamento`, `evento_nome`): `SimpleImputer('most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', min_frequency=0.05)` *(agrupa feiras raras automaticamente sem vazamento!)*.
3. **Encapsular em `Pipeline([('preprocessor', preprocessor), ('model', ...)])`.**

### Etapa 5 — Rotina de Processamento: Baseline, Validação Cruzada e `GridSearchCV`
*Referência: `Pepeline_processaemento.ipynb` (Seções 2 e 3), `Supervisionados_01.ipynb` e `Árvore_de_Decisão.ipynb`.*
1. **Modelo Baseline:** treinar `LogisticRegression` (ou `RandomForestClassifier`) com parâmetros padrão dentro do `Pipeline` e registrar o `ROC-AUC` inicial como referência.
2. **Validação Cruzada e Métrica:**
   - `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
   - `scoring = 'roc_auc'`.
3. **Comparação dos 4 classificadores em loop com `GridSearchCV` (grades enxutas):**
   - **Regressão Logística** (`LogisticRegression(max_iter=1000, random_state=42)`): `C`: `[0.1, 1.0, 10.0]`, `class_weight`: `[None, 'balanced']`.
   - **KNN** (`KNeighborsClassifier()`): `n_neighbors`: `[5, 11, 21]`, `weights`: `['uniform', 'distance']`, `p`: `[1, 2]`.
   - **Random Forest** (`RandomForestClassifier(random_state=42, n_jobs=-1)`): `n_estimators`: `[100, 300]`, `max_depth`: `[None, 6, 12]`, `min_samples_leaf`: `[1, 5]`, `class_weight`: `[None, 'balanced']`.
   - **XGBoost** (`XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')`): `n_estimators`: `[100, 250]`, `max_depth`: `[3, 6]`, `learning_rate`: `[0.05, 0.1]`, `subsample`: `[0.8, 1.0]`.
4. **Tabela comparativa:** ordenar os 4 modelos pelo melhor `ROC-AUC` médio na validação cruzada, exibir os melhores hiperparâmetros e selecionar o **modelo campeão**.

### Etapa 6 — Avaliação Final, Nested CV, Explicabilidade (SHAP) e Salvamento
*Referência: `aula_09.pdf`, `Árvore_de_Decisão.ipynb` e `Pepeline_processaemento.ipynb`.*
1. **Avaliação no Conjunto de Teste (`X_test`, `y_test`):**
   - Comparar `ROC-AUC` de **Treino / CV vs Teste** para verificar se houve *overfitting*.
   - Exibir as métricas completas de classificação:
     - `classification_report` (Acurácia, Precisão, Recall e F1-Score).
     - **Matriz de Confusão** (`ConfusionMatrixDisplay`).
     - **Curva ROC e AUC** comparando os 4 modelos ajustados no mesmo gráfico (como no final de `Supervisionados_01.ipynb`).
2. **Validação Final Robusta com Nested CV (no modelo campeão):**
   - Seguir exatamente o padrão de `Pepeline_processaemento.ipynb`:
     - `outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
     - `inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
     - `GridSearchCV` com a grade do campeão e `scoring='roc_auc'`.
   - Reportar a métrica final confiável com **média $\pm$ IC 95%** (`mean ± 1.96 * std / sqrt(5)`).
3. **Explicabilidade com SHAP:**
   - Se o campeão for árvore (`RandomForest` ou `XGBoost`), usar `shap.TreeExplainer` + `shap.summary_plot(..., plot_type="bar")`.
   - Se o campeão for linear (`LogisticRegression`), exibir os coeficientes e/ou `shap.LinearExplainer` (e, se for `KNN`, complementar com `TreeExplainer` no melhor modelo de árvore para leitura das features).
4. **Retreino Final, Salvamento e Exemplo de Recomendação:**
   - Retreinar o `GridSearchCV` do modelo campeão com 100% dos dados (`final_search.fit(X, y)`).
   - Salvar o pipeline completo com `joblib.dump(final_search.best_estimator_, "modelo_recomendacao_final.pkl")`.
   - Mostrar 2 ou 3 exemplos práticos de pedidos com a probabilidade prevista (`predict_proba()[:, 1]`), a decisão de recomendar (`Sim`/`Não`) e o valor real.

---

## 5. O que NÃO fazer (para manter a simplicidade)

- **Não** usar redes neurais / PyTorch (`MLP`, `Autoencoder`, `DataLoader`) nem `Optuna` (mantemos só `GridSearchCV`).
- **Não** fazer expansão combinatória de cestas ($2^k - 2$), *cold-start* sintético, multi-task ou máscaras customizadas.
- **Não** espalhar o projeto em vários arquivos `.py`, `.parquet`, `.npz` ou API — manter tudo em **um único notebook**.
- **Não** inflar a Etapa 3 com dezenas de gráficos repetitivos nem testes estatísticos avançados fora das aulas.

---

## 6. Checklist Rápido antes da Entrega (26/11)

- [ ] Dataset limpo (erros de valor/embalagem corrigidos, `produto_base`/`familia_produto`/`multiplicador` extraídos e colunas inúteis removidas).
- [ ] Base consolidada por pedido com alvo binário na **família mais frequente em compras combinadas** e regras anti-*leakage* aplicadas (`qtd_outros_itens >= 1` e features de cesta sem o alvo).
- [ ] Engenharia de recursos aplicada (tempo + ciclo `sin`/`cos` + `turno` + features de cesta + `evento_nome`).
- [ ] Demonstração curta (1 célula por técnica) de todo o pré-processamento da aula, incluindo simulação didática de ~5% de nulos numa cópia.
- [ ] `train_test_split(..., stratify=y)` executado **antes** de ajustar o `Pipeline` + `ColumnTransformer`.
- [ ] Baseline + comparação de **Regressão Logística, KNN, Random Forest e XGBoost** com `StratifiedKFold(5)` e `GridSearchCV` otimizando `roc_auc`.
- [ ] Avaliação no teste (checar *overfitting*, `classification_report`, Matriz de Confusão e Curva ROC comparando os 4 modelos).
- [ ] Nested CV no campeão (média $\pm$ IC 95% de `ROC-AUC`), explicabilidade com SHAP, retreino em 100% dos dados e `joblib.dump`.
- [ ] Notebook executando do início ao fim sem erros e com explicações curtas e claras.
