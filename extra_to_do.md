# Extra To-Do — Demonstrações Complementares de Pré-processamento (Exigência do Enunciado)

Técnicas vistas em aula que **não alteram a base oficial nem entram no pipeline final**, mas precisam ser demonstradas no [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb) para cumprir integralmente o enunciado (*"mesmo que o modelo escolhido posteriormente não necessite de algumas técnicas, demonstre-as mesmo assim"*).

> **Quando executar:** após o fluxo principal completo (`Pipeline` + `ColumnTransformer`, modelos, `GridSearchCV`, avaliação no teste, Nested CV, SHAP e `joblib.dump`) estar funcionando de ponta a ponta. Todas rodam sobre **cópias** dos dados, em células curtas e objetivas.

---

## 1. Outliers — Comparativo Completo da Aula
*Referência: [Pos__Identificação_outlier.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pos__Identificação_outlier.ipynb).*

- [ ] Acrescentar o cálculo de **MAD (`mediana + 3×MAD`)** na tabela univariada.
- [ ] Rodar **`DBSCAN`** (estrutural) e **`LocalOutlierFactor` / LOF** (local) sobre `(pedido_valor_total, qtd_unidades)` escalonados e montar o painel de 3 gráficos lado a lado com **`IsolationForest`** (global).
- [ ] Incluir nota curta explicando as outras duas estratégias pós-identificação (**Estratégia 2 — Capping/Winsorization** e **Estratégia 4 — Manutenção com flag**) e por que não foram escolhidas para os fechamentos de caixa.

---

## 2. Imputação de Dados Ausentes (Simulação Didática)
*Referência: [Preechimento_de_dados.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Preechimento_de_dados.ipynb) e [pre.pdf](file:///home/lucas/Projects/cd_g2/class_material/pre.pdf).*

- [ ] Resumir brevemente os mecanismos **MCAR, MAR e MNAR** (citando `add_indicator=True` para MNAR) após confirmar `0` nulos reais na base modelada.
- [ ] Em uma cópia de `df_varejo`, inserir **~5% de `NaN`** aleatórios (`seed=42`) em `pedido_valor_total`.
- [ ] Comparar o **`MAE`** de reconstrução entre **`SimpleImputer(strategy='median')`**, **`KNNImputer(n_neighbors=5)`** e **`IterativeImputer(random_state=42)`** (MICE usando features correlacionadas da cesta).

---

## 3. Discretização / *Binning* (Comparativo `KBinsDiscretizer`)
*Referência: [Pipeline_de_Pre_Processamento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pipeline_de_Pre_Processamento.ipynb).*

- [ ] Em uma cópia de `pedido_valor_total`, aplicar **`KBinsDiscretizer(n_bins=4, encode='ordinal')`** comparando as 3 estratégias da aula (**`uniform`**, **`quantile`** e **`kmeans`**), mostrando limites e contagem por faixa (complementando a discretização personalizada `turno` já usada no modelo).

---

## 4. Codificação Categórica — Comparativo com `TargetEncoder`
*Referência: [Pipeline_de_Pre_Processamento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pipeline_de_Pre_Processamento.ipynb).*

- [ ] Demonstrar **`TargetEncoder(smooth='auto', target_type='binary', random_state=42)`** em uma cópia de `evento_nome` frente a `pagamento_digital`, comparando com a média simples por feira (e justificando por que o `ColumnTransformer` oficial usa `OneHotEncoder` para preservar interpretabilidade direta no SHAP).

---

## 5. Correção de Assimetria — Comparativo Visual (`log1p` vs `Yeo-Johnson`)
*Referência: [Pipeline_de_Pre_Processamento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pipeline_de_Pre_Processamento.ipynb).*

- [ ] Comparar em cópia o coeficiente `.skew()` de `pedido_valor_total` e `qtd_unidades` antes vs. depois de **`np.log1p`** e **`PowerTransformer(method='yeo-johnson')`**.
- [ ] Plotar 1 figura compacta com **histograma + Q-Q plot** antes e depois da transformação em `pedido_valor_total`.

---

## 6. Escalonamento — Comparativo de 3 *Scalers*
*Referência: [Pipeline_de_Pre_Processamento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pipeline_de_Pre_Processamento.ipynb).*

- [ ] Comparar em cópia **`StandardScaler`**, **`MinMaxScaler`** e **`RobustScaler`** sobre `pedido_valor_total` e `qtd_unidades` em 1 tabela resumo (`min`, `mediana`, `média`, `std`, `max`) + 1 gráfico comparativo.

---

## 7. Seleção Supervisionada de Variáveis (`SelectKBest`)
*Referência: [Pipeline_de_Pre_Processamento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pipeline_de_Pre_Processamento.ipynb).*

- [ ] Rodar **`SelectKBest(score_func=mutual_info_classif)`** sobre a base tratada para exibir um ranking univariado de relevância das features frente a `pagamento_digital` (apenas diagnóstico complementar).

---

## 8. Redução de Dimensionalidade com PCA
*Referência: [Pipeline_de_Pre_Processamento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pipeline_de_Pre_Processamento.ipynb) e [pre.pdf](file:///home/lucas/Projects/cd_g2/class_material/pre.pdf).*

- [ ] Aplicar `StandardScaler` + **`PCA()`** nas variáveis numéricas selecionadas.
- [ ] Plotar 1 figura com 2 painéis: **variância explicada acumulada** (marcando `90%` e `95%`) e **projeção 2D (`PC1` vs `PC2`)** colorida por `pagamento_digital`.
- [ ] Justificar brevemente por que o PCA fica apenas demonstrativo (baixa dimensionalidade e preservação da interpretabilidade no SHAP).
