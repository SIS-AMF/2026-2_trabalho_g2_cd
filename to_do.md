# To-Do — Etapa 4: Split Estratificado e `Pipeline` + `ColumnTransformer` (Sem *Data Leakage*)

Escopo baseado na Etapa 4 do [plan.md](file:///home/lucas/Projects/cd_g2/plan.md) e nos notebooks [Pipeline_de_Pre_Processamento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pipeline_de_Pre_Processamento.ipynb) e [Pepeline_processaemento.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pepeline_processaemento.ipynb), partindo da base selecionada `df_model` (`2.989` pedidos × `16` colunas) no [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb).

---

## 1. Separação `X`/`y` e Split Estratificado (80/20) — Antes de Qualquer `.fit()`

- [ ] Separar features `X` (`15` colunas) e alvo `y` (`pagamento_digital`).
- [ ] Executar `train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)` gerando **`2.391` pedidos de treino (`80%`)** e **`598` pedidos de teste (`20%`)**.
- [ ] Validar via `assert` que:
  - [ ] Não há sobreposição de índices entre treino e teste;
  - [ ] A proporção da classe positiva (`~40,4%` Digital) é preservada em treino e teste (estratificação funcionando).

---

## 2. Definição dos Grupos de Colunas e Transformadores (`ColumnTransformer`)

Aplicar exatamente as decisões diagnosticadas na Seção 3.2:

- [ ] **`skewed_features`** (`pedido_valor_total`, `qtd_unidades`):
  - [ ] `Pipeline`: `SimpleImputer(strategy="median")` $\rightarrow$ `PowerTransformer(method="yeo-johnson")` *(já padroniza com média `0` e desvio `1` via `standardize=True`)*.
- [ ] **`numeric_features`** (`qtd_itens`, `hora_sin`, `hora_cos`):
  - [ ] `Pipeline`: `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`.
- [ ] **`binary_features`** (`eh_fim_de_semana`, `eh_pacote`, `teve_desconto`, `tem_cuca`, `tem_rapadura`, `tem_alfajor`, `tem_amendoim_doces`, `tem_bolacha`):
  - [ ] `SimpleImputer(strategy="most_frequent")` *(mantém `0`/`1`, preservando interpretação direta no SHAP)*.
- [ ] **`ordinal_features`** (`turno`):
  - [ ] `Pipeline`: `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OrdinalEncoder(categories=[["Manhã", "Tarde", "Noite"]])`.
- [ ] **`categorical_features`** (`evento_nome`):
  - [ ] `Pipeline`: `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore", min_frequency=0.05, sparse_output=False)` *(agrupa feiras raras `< 5%` dentro de cada fold/treino sem vazamento e devolve matriz densa pronta para o SHAP)*.
- [ ] Validar via `assert` que os `5` grupos cobrem exatamente as `15` colunas de `X` (sem duplicatas nem omissões).
- [ ] Montar `preprocessor = ColumnTransformer(..., remainder="drop", verbose_feature_names_out=False)` e função auxiliar `criar_pipeline(modelo)` usando `clone(preprocessor)` dentro de `Pipeline([("preprocessor", ...), ("model", modelo)])`.

---

## 3. Teste de Integridade do Pré-processamento (Zero *Data Leakage*)

- [ ] Ajustar uma cópia isolada (`prep_check = clone(preprocessor)`) chamando **`fit_transform(X_train)`** apenas no treino e **`transform(X_test)`** apenas no teste (mantendo `preprocessor` original virgem para o CV).
- [ ] Validar via `assert`:
  - [ ] `0` valores nulos (`NaN`) nas matrizes transformadas de treino e teste;
  - [ ] Mesmo número de colunas em `X_train` e `X_test` após transformação (e exibir lista completa de `get_feature_names_out()`, confirmando agrupamento das feiras raras `< 5%`);
  - [ ] Média $\approx 0$ e desvio padrão $\approx 1$ em `pedido_valor_total` e `qtd_unidades` no treino transformado.
- [ ] Exibir amostra (`head()`) do `DataFrame` transformado e o diagrama do `preprocessor` no notebook.
- [ ] Executar o [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb) de ponta a ponta no kernel `cd_g2` sem erros.
