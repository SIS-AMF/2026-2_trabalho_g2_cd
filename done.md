# Done — Tarefas Concluídas

## 0. Configuração do Ambiente

- [x] Criar [requirements.txt](file:///home/lucas/Projects/cd_g2/requirements.txt) e [.gitignore](file:///home/lucas/Projects/cd_g2/.gitignore).
- [x] Criar ambiente virtual `.venv` (Python `3.14.10`), instalar todas as dependências (`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`, `joblib`, `jupyter`, `ipykernel`, `nbconvert`) e registrar o kernel Jupyter `Python (cd_g2)`.
- [x] Criar o notebook único [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb) com cabeçalho alinhado ao [README.md](file:///home/lucas/Projects/cd_g2/README.md).

---

## 1. Limpeza dos Dados de Vendas (`dataset.csv`)

- [x] **Carregar e inspecionar a base bruta:** verificado `.shape` (`3.815 x 13`), `.info()`, `.isnull().sum()` (`1` nulo real em colunas descartadas) e `.describe()`.
- [x] **Remover colunas irrelevantes de caixa:** descartadas `pedido_vendedor` e `item_pedido_user_id`.
- [x] **Validar e limpar descontos:** confirmado via `assert` que `subtotal - pedido_valor_desconto == pedido_valor_total` (`100%` em R$, erro `R$ 0,000000`) e removida `pedido_tipo_desconto`.
- [x] **Remover itens cancelados/inválidos:** excluídas `2` linhas com `item_pedido_quantidade <= 0` (`3.815` $\rightarrow$ `3.813` linhas, `3.003` pedidos).
- [x] **Padronizar textos e categorias:**
  - [x] Espaços extras removidos em `item_pedido_nome_produto` e `evento_nome`.
  - [x] Unificado sinônimo `"Amendoim salgado e temperado"` $\rightarrow$ `"Amendoim Temperado"`.
  - [x] Unificados caixas da mesma feira: `"1 Moranguinho"` e `"2-Moranguinho"` $\rightarrow$ `"Festa do Moranguinho"`.
  - [x] Padronizado meio de pagamento em `pagamento_grupo`: `Dinheiro`, `Cartão` e `PIX`.
- [x] **Extrair estrutura dos produtos e reconciliar pacotes:**
  - [x] Extraídos `20` `produto_base` mapeados em `6` famílias (`Cuca`, `Rapadura`, `Alfajor`, `Amendoim_Doces`, `Bolacha` e `Pao`).
  - [x] Extraído `multiplicador` da embalagem (incluindo os `59` casos sem número e a exceção de `Rapadura de Melado Grão Moído (Pacote x1)` a R$ 20,00 $\rightarrow$ `m = 3`).
  - [x] Reconciliados `qtd_unidades` e `preco_unitario` nos `139` pedidos divergentes, validado com `assert` (`0` divergências, erro máximo `R$ 0,000000`).

---

## 2. Consolidação por Pedido e Criação de Variáveis Novas

- [x] **Criar a variável alvo (`y`):** `pagamento_digital` (`1` = `Cartão`/`PIX`, `0` = `Dinheiro`) com colunas originais de pagamento removidas.
- [x] **Criar variáveis de tempo** (`America/Sao_Paulo`): `hora`, `dia_semana`, `eh_fim_de_semana`, `hora_sin`, `hora_cos` e `turno` (`Manhã`, `Tarde`, `Noite`).
- [x] **Criar variáveis de perfil da cesta:** `pedido_valor_total`, `pedido_subtotal`, `qtd_itens`, `qtd_unidades`, `eh_pacote`, `teve_desconto`.
- [x] **Criar flags binárias das 6 famílias:** `tem_cuca`, `tem_rapadura`, `tem_alfajor`, `tem_amendoim_doces`, `tem_bolacha`, `tem_pao`.
- [x] **Validar integridade da tabela consolidada (`df_pedidos`):** `3.003` pedidos únicos × `21` colunas, `0` valores nulos, notebook executado e validado de ponta a ponta.

---

## 3. Pré-processamento Efetivo e Tomada de Decisão (Etapa 3)

- [x] **Outliers — Visualização Inicial, Regra de Domínio e `IsolationForest` (`3.003` $\rightarrow$ `2.989`):**
  - [x] Plotada distribuição inicial (`boxplot` + scatterplot `qtd_unidades × pedido_valor_total` com escala `symlog` calibrada e ticks em reais) destacando visualmente os `2` valores zerados e os eventos extremos.
  - [x] Removidos os `2` pedidos gratuitos (`pedido_valor_total <= 0`) por regra de domínio $\rightarrow$ `3.001` pedidos pagos.
  - [x] Aplicado `IsolationForest(contamination='auto', random_state=42)` sobre `(pedido_valor_total, qtd_unidades)` escalonados, com histograma de `decision_function` e corte visual em `score < -0.30`, removendo `12` eventos extremos (`0,40%`, Estratégia 1 — *Trimming*) $\rightarrow$ **`df_varejo`** com **`2.989` pedidos** (`40,38%` Digital / `59,62%` Dinheiro; `14` removidos no total = `0,47%`).
  - [x] Registrada Estratégia 3 (*Transformação*) para suavizar a cauda natural restante via `PowerTransformer`.
- [x] **Diagnóstico de Assimetria, Escala e Categóricas (em cópia, sem *data leakage*):**
  - [x] Comparado `skew` antes/depois e $\lambda$ do Yeo-Johnson em cópia: `pedido_valor_total` (`3,98 -> 0,00`, $\lambda = 0{,}02$) e `qtd_unidades` (`3,20 -> 0,33`, $\lambda = -1{,}18$) mapeadas para `PowerTransformer`; `qtd_itens` ($\lambda = -9{,}41$, fora de `[-5, 5]`) e `hora_sin`/`hora_cos` (coordenadas cíclicas) mapeadas para `StandardScaler`.
  - [x] Confirmada ordem ordinal de `turno` (`['Manhã', 'Tarde', 'Noite']`) e frequência das `14` feiras em `evento_nome` (`7` principais $\ge 5\%$ e `7` raras $< 5\%$ para `min_frequency=0.05`).
- [x] **Seleção de Variáveis (Spearman + `VarianceThreshold`):**
  - [x] Heatmap de correlação de **Spearman** (adequado à assimetria/não-normalidade): removidas `pedido_subtotal` ($\rho = 0{,}997$), `hora` ($\rho = -0{,}945$ / $0{,}864$) e `dia_semana` ($\rho = 0{,}882$), mantendo o par cíclico complementar `hora_sin × hora_cos` ($\rho = -0{,}808$).
  - [x] `VarianceThreshold(threshold=0.01)` nas binárias: removida `tem_pao` (`0,30%`, variância `0,0030 < 0,01`).
- [x] **Validação da Base Selecionada (`df_model`):** `2.989` linhas × `16` colunas (`15` features + alvo `pagamento_digital`), `0` valores nulos, notebook [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb) executado de ponta a ponta sem erros.

---

## 4. Split Estratificado e `Pipeline` + `ColumnTransformer` (Etapa 4)

- [x] **Separação `X`/`y` e Split Estratificado (80/20):** `X` (`15` features) e `y` (`pagamento_digital`) divididos em `2.391` treino (`40,40%` Digital) e `598` teste (`40,30%` Digital) com índices disjuntos antes de qualquer `.fit()`.
- [x] **Montagem do `ColumnTransformer` (`preprocessor`) + `criar_pipeline(modelo)`:** 5 grupos cobertos sem duplicatas (`PowerTransformer` em valor/unidades, `StandardScaler` em `qtd_itens`/`hora_sin`/`hora_cos`, `SimpleImputer` como blindagem em todos os ramos, `OrdinalEncoder` em `turno` e `OneHotEncoder(min_frequency=0.05, sparse_output=False)` em `evento_nome`).
- [x] **Teste de integridade (zero *data leakage*):** validado em `clone(preprocessor)` (`fit_transform` só em `X_train`, `transform` em `X_test`): `15` $\rightarrow$ `22` features (`7` feiras principais + `evento_nome_infrequent_sklearn`), `0` nulos, média `0,000` e std `1,000` no treino transformado.
