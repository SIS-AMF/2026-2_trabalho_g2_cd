# To-Do — Etapa 5: Baseline, Validação Cruzada e `GridSearchCV`

Escopo baseado na Etapa 5 do [plan.md](file:///home/lucas/Projects/cd_g2/plan.md) e nos notebooks [Pepeline_processaemento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pepeline_processaemento.ipynb), [Supervisionados_01.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Supervisionados_01.ipynb) e [Árvore_de_Decisão.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Árvore_de_Decisão.ipynb), rodando **exclusivamente sobre `X_train`, `y_train`** (`2.391` pedidos) no [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb).

---

## 1. Validação Cruzada e Modelos Baseline

- [ ] Definir `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` e `scoring = "roc_auc"`.
- [ ] Avaliar com `cross_val_score` sobre `X_train`, `y_train`:
  - [ ] **`DummyClassifier(strategy="most_frequent")`** (piso de referência: `ROC-AUC = 0,500`).
  - [ ] **`LogisticRegression(max_iter=1000, random_state=42)`** default dentro de `criar_pipeline(...)` (referência linear sem tuning).
- [ ] Exibir tabela com `ROC-AUC` médio e desvio padrão dos baselines.

---

## 2. Comparação dos 4 Classificadores com `GridSearchCV`

Rodar em loop `GridSearchCV(criar_pipeline(modelo), param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, return_train_score=True)` sobre `X_train`, `y_train`:

- [ ] **Regressão Logística** (`LogisticRegression(max_iter=1000, random_state=42)`):
  - `model__C`: `[0.1, 1.0, 10.0]`, `model__class_weight`: `[None, "balanced"]`.
- [ ] **KNN** (`KNeighborsClassifier()`):
  - `model__n_neighbors`: `[5, 11, 21, 31]`, `model__weights`: `["uniform", "distance"]`, `model__p`: `[1, 2]`.
- [ ] **Random Forest** (`RandomForestClassifier(random_state=42, n_jobs=-1)`):
  - `model__n_estimators`: `[100, 300]`, `model__max_depth`: `[None, 6, 12]`, `model__min_samples_leaf`: `[1, 5]`, `model__class_weight`: `[None, "balanced"]`.
- [ ] **XGBoost** (`XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss")`):
  - `model__n_estimators`: `[100, 250]`, `model__max_depth`: `[3, 6]`, `model__learning_rate`: `[0.05, 0.1]`, `model__subsample`: `[0.8, 1.0]`.

---

## 3. Tabela Comparativa e Seleção do Campeão

- [ ] Montar tabela ordenada pelo melhor **`ROC-AUC` médio no CV** exibindo: modelo, `ROC-AUC CV (média ± std)`, `ROC-AUC Treino` (checando indício de *overfitting*) e melhores hiperparâmetros.
- [ ] Armazenar `melhores_modelos` (os 4 pipelines ajustados) e destacar o **modelo campeão** (`campeao_nome`, `campeao_pipeline`) para a Etapa 6.
- [ ] Executar o [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb) de ponta a ponta no kernel `cd_g2` sem erros.
