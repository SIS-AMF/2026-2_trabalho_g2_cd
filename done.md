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
