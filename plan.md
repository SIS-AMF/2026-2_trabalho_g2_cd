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
| **Parte 1 — Pré-processamento e Engenharia de Recursos** | Limpeza enxuta, criação de novas features, demonstração curta das técnicas de pré-processamento da aula e montagem do `Pipeline` + `ColumnTransformer` sem *data leakage*. | `pre.pdf`, `Pos__Identificação_outlier.ipynb`, `Preechimento_de_dados.ipynb`, `Pipeline_de_Pre_Processamento.ipynb` |
| **Parte 2 — Rotina de Processamento e Avaliação** | Split estratificado, baseline, comparação de 4 classificadores com `StratifiedKFold` e `GridSearchCV` (otimizando `roc_auc`), avaliação no teste, Nested CV (IC 95%), SHAP e salvamento do modelo final. | `aula_09.pdf`, `Supervisionados_01.ipynb`, `Árvore_de_Decisão.ipynb`, `Pepeline_processaemento.ipynb` |

> **Materiais fora de escopo:** `Pre_Imagens.ipynb` (imagens) e `Kmeans.ipynb` (não supervisionado — usado apenas na demonstração de discretização com `KBinsDiscretizer(strategy='kmeans')`).

---

## 3. Decisões Fechadas do Projeto (Escopo Definitivo)

| Decisão | Definição Escolhida | Justificativa |
| :--- | :--- | :--- |
| **Problema** | **Classificação Binária** do Meio de Pagamento no nível do pedido | Objetivo simples, usa todos os pedidos pagos válidos (`2.989`) sem precisar filtrar itens da cesta, e apoia o planejamento de troco, maquininhas e internet nas feiras. |
| **Alvo (`y`)** | **`pagamento_digital`** (`1` = `Cartão` ou `PIX`; `0` = `Dinheiro`) | Classes equilibradas (**40,38% Digital** vs **59,62% Dinheiro**), evitando distorções de desbalanceamento extremo e mantendo a avaliação (ROC, Matriz de Confusão e SHAP) direta e clara. |
| **Feature `evento_nome`** | **Incluída em `X`** (unificando caixas da *Festa do Moranguinho* e agrupando feiras raras `< 5%`) | Forte preditor real (de 0%–17% digital em feiras rurais até 52%–56% em grandes feiras) e permite demonstrar agrupamento de raras, `OneHotEncoder` e `TargetEncoder`. |
| **Redundâncias Temporais** | Remover **`mes`** e **`dia_semana`** bruto do modelo final | No dataset cada feira ocorre em praticamente 1 único mês (`mes` $\approx$ `evento_nome`), e `dia_semana` tem correlação alta (`r = 0,83`) com `eh_fim_de_semana`. |
| **Imputação (base sem nulos)** | `SimpleImputer` no `Pipeline` + **1 célula didática simulando ~5% de `NaN` numa cópia** | A base bruta tem apenas 1 nulo em colunas descartadas (`pedido_vendedor`/`item_pedido_user_id`), ficando limpa (`0` nulos). A simulação cumpre a exigência prática de `Preechimento_de_dados.ipynb` sem alterar os dados oficiais. |
| **Modelos comparados** | **Regressão Logística**, **KNN**, **Random Forest** e **XGBoost** | Cobre 3 famílias ("tribos") da aula (linear, distância e árvores/boosting), justificando todo o pré-processamento. |
| **Método de Tuning** | **`GridSearchCV`** com grades enxutas | Simples, rápido (inclusive no Nested CV) e direto de explicar na apresentação. |
| **Métrica Principal** | **`roc_auc`** (no CV, tuning e Nested CV) | Avalia a capacidade de separação probabilística em todos os limiares, complementada por Acurácia, Precisão, Recall, F1 e Matriz de Confusão no teste. |

### Dataset e Regras de Limpeza (`dataset.csv`)
- **Base bruta:** `3.815` registros de itens vendidos em feiras (`3.003` pedidos únicos, `13` colunas originais, de mar/2026 a nov/2026).
- **Limpeza enxuta:**
  1. **Remover colunas irrelevantes:** `pedido_vendedor`, `item_pedido_user_id` (100% idênticas, apenas 2 logins genéricos de caixa e 1 `NaN`) e `pedido_tipo_desconto` (após confirmar matematicamente que 100% dos descontos já estão em R$).
  2. **Remover itens inválidos:** excluir as `2` linhas com `item_pedido_quantidade <= 0` (`3.815` $\rightarrow$ `3.813` linhas, mantendo os `3.003` pedidos).
  3. **Padronizar textos e categorias:**
     - Remover espaços extras em produtos e eventos.
     - Unificar sinônimo `"Amendoim salgado e temperado"` $\rightarrow$ `"Amendoim Temperado"`.
     - Unificar caixas da mesma feira: `"1 Moranguinho"` e `"2-Moranguinho"` $\rightarrow$ `"Festa do Moranguinho"`.
     - Padronizar meio de pagamento: `"Pix"`/`"PIX"` $\rightarrow$ `"PIX"`, e `"Cartão de Crédito"`/`"Cartão de Débito"` $\rightarrow$ `"Cartão"` (originando o alvo binário `pagamento_digital`: `Cartão`/`PIX` = `1`, `Dinheiro` = `0`).
  4. **Extrair `produto_base`, `familia_produto` e `multiplicador` e reconciliar pacotes:** aplicar a função enxuta já validada para corrigir os 139 pedidos divergentes (*ExpoDireto*), os 59 pacotes sem multiplicador (*1 Moranguinho*) e a exceção `Rapadura de Melado Grão Moído (Pacote x1)` a R$ 20,00, garantindo `sum(qtd_un * preco_un) == pedido_subtotal` com erro `R$ 0,00` (`assert`).
- *(Nota: a detecção e o tratamento de outliers ficam integralmente concentrados na **Etapa 3**, alinhados ao `README.md`.)*

### Regras Obrigatórias Anti-*Data Leakage*
1. **Não incluir derivadas do alvo em `X`:** `pedido_metodo_pagamento` e `pagamento_grupo` são usadas exclusivamente para construir `y` (`pagamento_digital`) e removidas de `X`.
2. **Divisão antes de qualquer ajuste estatístico:** executar `train_test_split` antes de qualquer `fit`/`transform` (`SimpleImputer`, `PowerTransformer`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`, etc.).
3. **Tudo encapsulado no `Pipeline` + `ColumnTransformer`:** inclusive o agrupamento de feiras raras (`min_frequency=0.05`), garantindo que as frequências sejam estimadas apenas dentro de cada fold de treino.

---

## 4. Passo a Passo do Trabalho (Estrutura do Notebook)

### Etapa 1 — Carregamento, Inspeção Rápida e Limpeza Básica
*Referência: `Pipeline_de_Pre_Processamento.ipynb` e `pre.pdf`.*
- Carregar `dataset.csv` e inspecionar `.shape`, `.info()`, `.isnull().sum()` (destacando o único nulo real nas colunas descartadas) e `.describe()`.
- Aplicar os passos 1 a 4 da limpeza (remover colunas inúteis e `qtd <= 0`, padronizar textos/eventos/pagamentos, extrair `produto_base`, `familia_produto` e `multiplicador`, e validar a reconciliação contábil com `assert`).
- Mapear os 20 produtos base em 6 famílias comerciais: `Cuca`, `Rapadura`, `Alfajor`, `Amendoim_Doces`, `Bolacha` e `Pao`.

### Etapa 2 — Engenharia de Recursos (*Feature Engineering*) e Base por Pedido
*Referência: `Pipeline_de_Pre_Processamento.ipynb` (Seção "Engenharia de atributos") e `pre.pdf`.*
Consolidar a base para **1 linha por pedido (`3.003` pedidos antes do filtro de outliers)** com o alvo **`pagamento_digital`** (`1` = Digital, `0` = Dinheiro) e features simples e explicáveis:
1. **Features de Tempo** (extraídas de `pedido_data` em `America/Sao_Paulo`):
   - `hora`, `dia_semana`, `eh_fim_de_semana` (`0` ou `1`).
   - **Codificação cíclica da hora:** `hora_sin = sin(2 * pi * hora / 24)` e `hora_cos = cos(2 * pi * hora / 24)`.
   - **Discretização personalizada:** `turno` (`Manhã` `< 12h`, `Tarde` `12h–17h`, `Noite` `>= 18h`).
2. **Features da Cesta (pedido completo):**
   - `pedido_valor_total` e `pedido_subtotal` *(mantida temporariamente para demonstrar multicolinearidade $r = 0{,}997$ na Etapa 3)*.
   - `qtd_itens`: número de itens/SKUs distintos no pedido.
   - `qtd_unidades`: soma total de unidades base (`qtd_un`) no pedido.
   - `eh_pacote`: flag binária (`1` se há algum item com `multiplicador > 1`).
   - `teve_desconto`: flag binária (`1` se `pedido_valor_desconto > 0`) *(76,9% dos pedidos com desconto são pagos em Dinheiro)*.
   - Flags binárias das **6 famílias de produto** presentes no pedido: `tem_cuca`, `tem_rapadura`, `tem_alfajor`, `tem_amendoim_doces`, `tem_bolacha` e `tem_pao` *(esta última com apenas 0,3% de frequência, usada na demonstração de variância quase nula)*.
3. **Contexto da venda:** `evento_nome` (já unificando *Festa do Moranguinho*).

### Etapa 3 — Demonstração Curta das Técnicas de Pré-processamento da Aula
*Referência: `Pos__Identificação_outlier.ipynb`, `Pipeline_de_Pre_Processamento.ipynb`, `Preechimento_de_dados.ipynb` e `pre.pdf`.*

> **Disciplina de simplicidade:** limitar cada técnica abaixo a **1 célula curta e objetiva** (demonstrações feitas sobre cópias quando alteram dados, sem bagunçar a base oficial).

1. **Detecção e Tratamento de Outliers (antes do filtro final — seguindo `Pos__Identificação_outlier.ipynb`):**
   - **Visualização e Análise Univariada inicial (`3.003` pedidos):** exibir `pedido_valor_total × qtd_unidades` e comparar **Quantis (`95%/99%`)**, **IQR (`1.5×IQR`)**, **Tukey Extremo (`3×IQR`)** e **Z-Score (`3σ`)**, destacando os `2` pedidos gratuitos e os fechamentos extremos de caixa.
   - **Regra de Domínio (pedidos gratuitos):** remover `pedido_valor_total <= 0` (`2` pedidos com 100% de desconto $\rightarrow$ `3.001` pedidos pagos).
   - **Análise Multivariada (`IsolationForest` com corte visual via `decision_function`):** rodar `IsolationForest(contamination='auto', random_state=42)` sobre `(pedido_valor_total, qtd_unidades)` escalonados nos `3.001` pedidos pagos, plotando o **histograma dos scores** e o scatterplot com corte em `score < -0.30` (12 eventos extremos, `0,40%`). *(Painel complementar com MAD, `DBSCAN` e `LOF` registrado em `extra_to_do.md`.)*
   - **Decisão Pós-Identificação (Estratégias da aula):**
     - **Estratégia 1 — Remoção (*Trimming*):** remover os `2` gratuitos + `12` extremos = **`14` pedidos (`0,47%` da base)** $\rightarrow$ **`2.989` pedidos** (`40,38%` Digital / `59,62%` Dinheiro).
     - **Estratégia 3 — Transformação (`PowerTransformer`):** manter a cauda alta natural restante e suavizá-la dentro do `Pipeline`.
2. **Imputação de Dados Ausentes (célula didática):**
   - Confirmar `0` nulos na base modelada, resumir MCAR/MAR/MNAR e simular **~5% de `NaN` em uma cópia** de `pedido_valor_total`.
   - Comparar o erro (`MAE`) entre **`SimpleImputer(strategy='median')`**, **`KNNImputer`** e **`IterativeImputer`** (MICE), espelhando o exercício de `Preechimento_de_dados.ipynb`.
3. **Discretização (Binning):**
   - Mostrar `turno` (regra de negócio) e comparar brevemente `KBinsDiscretizer` (`uniform`, `quantile`, `kmeans`) em `pedido_valor_total`.
4. **Codificação de Categóricas (Encoding):**
   - Demonstrar `OneHotEncoder`, `OrdinalEncoder` (`turno`: `['Manhã', 'Tarde', 'Noite']`), agrupamento de feiras raras (`< 5%`, que agrupa 6 feiras menores em `"Outros"`) e `TargetEncoder` (demonstrativo em `evento_nome` com `target_type='binary'`).
5. **Correção de Assimetria (*Skewness*):**
   - Calcular `.skew()` de `pedido_valor_total` (~`2,42`) e `qtd_unidades` (~`3,23`), mostrando histograma + Q-Q plot antes e depois de `np.log1p` e `PowerTransformer(method='yeo-johnson')`.
6. **Escalonamento (*Scaling*):**
   - Comparar brevemente `StandardScaler`, `MinMaxScaler` e `RobustScaler` em `pedido_valor_total` e `qtd_unidades`.
7. **Seleção de Características:**
   - **Multicolinearidade:** matriz de correlação de **Spearman** (`sns.heatmap`, adequada à assimetria das numéricas), identificando e removendo `pedido_subtotal` ($\rho = 0{,}997$ com `pedido_valor_total`), `hora` bruta ($\rho = -0{,}945$ / $0{,}864$) e `dia_semana` bruto ($\rho = 0{,}882$ com `eh_fim_de_semana`), mantendo o par cíclico `hora_sin`/`hora_cos`.
   - **Variância Quase Nula:** aplicar `VarianceThreshold(threshold=0.01)` nas binárias, identificando e removendo **`tem_pao`** (variância `0,0030`, apenas 9 pedidos).
   - **Seleção Supervisionada (demonstrativo):** 1 célula curta com `SelectKBest(score_func=mutual_info_classif)` para ranquear a relevância univariada das features em relação a `pagamento_digital`.
8. **Redução de Dimensionalidade (PCA):**
   - Aplicar `StandardScaler` + `PCA` nas variáveis numéricas, plotar variância explicada acumulada (90%/95%) e projeção 2D (`PC1` vs `PC2`), justificando por que o PCA fica apenas demonstrativo para preservar a interpretabilidade direta das features no SHAP.

### Etapa 4 — Split Estratificado e `Pipeline` + `ColumnTransformer` (Sem *Data Leakage*)
*Referência: `Pipeline_de_Pre_Processamento.ipynb` e `Pepeline_processaemento.ipynb` (Seção 1).*
1. **Divisão Treino/Teste antes de qualquer `fit`/`transform`:**
   - `train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)` sobre os `2.989` pedidos (`2.391` treino / `598` teste).
2. **Grupos de colunas por tipo de tratamento no `ColumnTransformer`:**
   - `skewed_features` (`pedido_valor_total`, `qtd_unidades`): `SimpleImputer('median')` $\rightarrow$ `PowerTransformer(method='yeo-johnson')` *(já padroniza com média 0 e desvio padrão 1 via `standardize=True`)*.
   - `numeric_features` (`qtd_itens`, `hora_sin`, `hora_cos`): `SimpleImputer('median')` $\rightarrow$ `StandardScaler()`.
   - `binary_features` (`eh_fim_de_semana`, `eh_pacote`, `teve_desconto`, `tem_cuca`, `tem_rapadura`, `tem_alfajor`, `tem_amendoim_doces`, `tem_bolacha`): `SimpleImputer('most_frequent')` *(mantém `0`/`1`, preservando a leitura direta no SHAP)*.
   - `ordinal_features` (`turno`): `SimpleImputer('most_frequent')` $\rightarrow$ `OrdinalEncoder(categories=[['Manhã', 'Tarde', 'Noite']])`.
   - `categorical_features` (`evento_nome`): `SimpleImputer('most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', min_frequency=0.05, sparse_output=False)` *(agrupa as 6 feiras raras `< 5%` sem vazamento e devolve matriz densa pronta para o SHAP)*.
3. **Encapsular em `Pipeline([('preprocessor', preprocessor), ('model', ...)])`.**

### Etapa 5 — Rotina de Processamento: Baseline, Validação Cruzada e `GridSearchCV`
*Referência: `Pepeline_processaemento.ipynb` (Seções 2 e 3), `Supervisionados_01.ipynb` e `Árvore_de_Decisão.ipynb`.*
1. **Modelos Baseline (referência inicial):**
   - `DummyClassifier(strategy='most_frequent')` (piso de referência: ~59,6% acurácia / `0,500` ROC-AUC).
   - `LogisticRegression` com parâmetros padrão dentro do `Pipeline` (referência linear sem tuning).
2. **Validação Cruzada e Métrica:**
   - `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
   - `scoring = 'roc_auc'`.
3. **Comparação dos 4 classificadores em loop com `GridSearchCV` (grades enxutas):**
   - **Regressão Logística** (`LogisticRegression(max_iter=1000, random_state=42)`): `C`: `[0.1, 1.0, 10.0]`, `class_weight`: `[None, 'balanced']`.
   - **KNN** (`KNeighborsClassifier()`): `n_neighbors`: `[5, 11, 21, 31]`, `weights`: `['uniform', 'distance']`, `p`: `[1, 2]`.
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
   - Extrair o modelo (`best_pipe.named_steps['model']`) e transformar `X_test` em `DataFrame` com `best_pipe.named_steps['preprocessor'].get_feature_names_out()`.
   - Se o campeão for árvore (`RandomForest` ou `XGBoost`), usar `shap.TreeExplainer` + `shap.summary_plot(..., plot_type="bar")` e *beeswarm*.
   - Se o campeão for linear (`LogisticRegression`), usar `shap.LinearExplainer` + coeficientes (e, se for `KNN`, complementar com `TreeExplainer` no melhor modelo de árvore para leitura das features).
4. **Retreino Final, Salvamento e Exemplos Práticos:**
   - Retreinar o `GridSearchCV` do modelo campeão com 100% dos dados (`final_search.fit(X, y)`).
   - Salvar o pipeline completo com `joblib.dump(final_search.best_estimator_, "modelo_pagamento_digital_final.pkl")`.
   - Mostrar 3 exemplos práticos de pedidos com a probabilidade prevista de pagamento digital (`predict_proba()[:, 1]`), a classe prevista (`Digital`/`Dinheiro`) e o valor real, encerrando com uma discussão honesta das limitações do sinal (~0,66–0,68 de AUC devido à preferência pessoal do cliente não observada na base).

---

## 5. O que NÃO fazer (para manter a simplicidade)

- **Não** usar redes neurais / PyTorch (`MLP`, `Autoencoder`, `DataLoader`) nem `Optuna` (mantemos só `GridSearchCV`).
- **Não** complicar com classificação multiclasse (`Dinheiro`/`Cartão`/`PIX`) nem expansão combinatória de cestas.
- **Não** espalhar o projeto em vários arquivos `.py`, `.parquet`, `.npz` ou API — manter tudo em **um único notebook**.
- **Não** inflar a Etapa 3 com dezenas de gráficos repetitivos nem testes estatísticos avançados fora das aulas.

---

## 6. Checklist Rápido antes da Entrega (26/11)

- [ ] Dataset limpo (colunas inúteis e `qtd <= 0` removidas, textos/eventos/pagamentos padronizados, `produto_base`/`familia_produto`/`multiplicador` extraídos e reconciliação validada com erro `R$ 0,00`).
- [ ] Base consolidada por pedido com alvo binário **`pagamento_digital`** (`1` = `Cartão`/`PIX`, `0` = `Dinheiro`) e tratamento de outliers baseado em `Pos__Identificação_outlier.ipynb` (visualização inicial + remoção de `2` gratuitos + `IsolationForest` via `decision_function < -0.30` removendo `12` extremos $\rightarrow$ `2.989` pedidos, `40,38%` Digital; cauda natural tratada com `PowerTransformer`).
- [ ] Engenharia de recursos aplicada (`hora_sin`/`hora_cos`, `turno`, `eh_fim_de_semana`, perfil da cesta, `teve_desconto`, flags de famílias e `evento_nome`).
- [ ] Demonstração curta (1 célula por técnica) de todo o pré-processamento da aula: outliers, imputação simulada (`SimpleImputer` vs `KNNImputer` vs `IterativeImputer`), discretização, encoding (`OneHot`, `Ordinal`, `TargetEncoder`, raras `< 5%`), assimetria, scaling, multicolinearidade (`subtotal` removido, `r = 0,997`), variância quase nula (`tem_pao` removido, var `0,003`), `SelectKBest` e PCA.
- [ ] `train_test_split(..., stratify=y)` executado **antes** de ajustar o `Pipeline` + `ColumnTransformer` (sem redundância no `PowerTransformer` e com `sparse_output=False` para o SHAP).
- [ ] Baseline (`DummyClassifier` + `LogisticRegression` default) + comparação de **Regressão Logística, KNN, Random Forest e XGBoost** com `StratifiedKFold(5)` e `GridSearchCV` otimizando `roc_auc`.
- [ ] Avaliação no teste (checar *overfitting*, `classification_report`, Matriz de Confusão e Curva ROC comparando os 4 modelos).
- [ ] Nested CV no campeão (média $\pm$ IC 95% de `ROC-AUC`), explicabilidade com SHAP, retreino em 100% dos dados e `joblib.dump`.
- [ ] Notebook executando do início ao fim sem erros e com explicações curtas, claras e honestas.
