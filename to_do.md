# To-Do — Pré-processamento Efetivo (Decisões que Afetam a Base e o Pipeline)

Foco exclusivo nas análises e transformações que **alteram o dataset oficial** ou **fundamentam diretamente a configuração do `ColumnTransformer`** no [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb).

> As demonstrações didáticas extras exigidas pelo enunciado (que não entram no modelo final) foram movidas para [extra_to_do.md](file:///home/lucas/Projects/cd_g2/extra_to_do.md) para serem adicionadas após o fluxo completo de modelagem estar pronto.

---

## 1. Outliers — Diagnóstico Objetivo e Corte Oficial (`3.003` $\rightarrow$ `2.984`)
*Afeta diretamente as linhas da base oficial (`df_varejo`).*

- [ ] **Diagnóstico univariado e multivariado objetivo** sobre `pedido_valor_total` e `qtd_unidades` (`3.003` pedidos):
  - [ ] Exibir tabela curta comparando **Quantis (`95%/99%`)**, **IQR (`1.5×`)**, **Tukey Extremo (`3×`)**, **Z-Score (`3σ`)** e **`IsolationForest`**, evidenciando os fechamentos de caixa (até R$ 3.693 / 565 unidades) versus a cauda natural do varejo.
  - [ ] Plotar 1 gráfico compacto (boxplot/dispersão) destacando os `19` pedidos operacionais anômalos.
- [ ] **Aplicar Tratamento Oficial (Estratégias 1 e 3 de [Pos__Identificação_outlier.ipynb](file:///home/lucas/Projects/cd_g2/class_material/Pos__Identificação_outlier.ipynb)):**
  - [ ] **Estratégia 1 (*Trimming*):** remover os `19` pedidos fora do varejo (`pedido_valor_total <= 0`, `pedido_subtotal >= 200`, `qtd_unidades >= 40` ou `evento_nome` $\in$ `{'Feira Polêsine', 'ExpoSerra'}`), gerando **`df_varejo`** (`2.984` pedidos: `40,4%` Digital / `59,6%` Dinheiro) validado com `assert`.
  - [ ] **Estratégia 3 (*Transformação*):** registrar a decisão de manter a cauda alta legítima do varejo (até R$ 175) e suavizá-la com `PowerTransformer(method='yeo-johnson')` dentro do `Pipeline`.

---

## 2. Diagnóstico para Decisão do `ColumnTransformer` (Assimetria e Categorias Raras)
*Define quais transformadores cada coluna receberá na Etapa 4.*

- [ ] **Assimetria (*Skewness*):** calcular `.skew()` das variáveis numéricas em `df_varejo`, confirmando que `pedido_valor_total` (`~2,42`) e `qtd_unidades` (`~3,23`) precisam de `PowerTransformer(method='yeo-johnson')` no `ColumnTransformer` (enquanto `qtd_itens`, `hora_sin` e `hora_cos` vão para `StandardScaler`).
- [ ] **Cardinalidade e categorias raras (`evento_nome`):** exibir a tabela de frequência das `13` feiras em `df_varejo`, fundamentando o uso de `OneHotEncoder(handle_unknown='ignore', min_frequency=0.05, sparse_output=False)` (`7` feiras $\ge 5\%$ mantidas + `6` feiras raras agrupadas sem *data leakage*).

---

## 3. Seleção de Variáveis (Remoção de Redundâncias e Baixa Variância)
*Afeta diretamente as colunas da base oficial (`df_model`).*

- [ ] **Multicolinearidade:** gerar o `sns.heatmap` de correlação das variáveis numéricas/binárias; comprovar e **remover**:
  - [ ] `pedido_subtotal` ($r = 0{,}997$ com `pedido_valor_total`);
  - [ ] `dia_semana` ($r = 0{,}83$ com `eh_fim_de_semana`);
  - [ ] `hora` bruta (já representada por `hora_sin`/`hora_cos` e `turno`).
- [ ] **Variância Quase Nula:** aplicar `VarianceThreshold(threshold=0.01)` nas colunas binárias; comprovar e **remover `tem_pao`** (`0,30%` dos pedidos, variância `0,0030 < 0,01`).

---

## 4. Validação da Base Final para Modelagem (`df_model`)

- [ ] Montar `df_model` (sem `pedido_id`, `pedido_subtotal`, `dia_semana`, `hora` e `tem_pao`) e validar via `assert`:
  - [ ] Exatamente **`2.984` linhas × `16` colunas** (`15` features + alvo `pagamento_digital`) e **`0` valores nulos**.
- [ ] Executar o [trabalho_g2.ipynb](file:///home/lucas/Projects/cd_g2/trabalho_g2.ipynb) de ponta a ponta no kernel `cd_g2` sem erros, pronto para iniciar a **Etapa 4 (`train_test_split` + `Pipeline` + `ColumnTransformer`)**.
