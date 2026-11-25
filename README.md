# Trabalho G2 - Ciência de Dados

## Objetivo

1. **Pré-processamento e Engenharia de Recursos (preparar os dados):**
   - Limpar os dados de vendas das feiras e criar variáveis novas.
   - Demonstrar na prática as técnicas vistas em aula (outliers, imputação de nulos, discretização, encoding, correção de assimetria, escalonamento, seleção de variáveis e PCA) — mesmo as que o modelo final não precisa.
   - Empacotar o pré-processamento real em um `Pipeline` + `ColumnTransformer` (sem *data leakage*).

2. **Treino, Otimização e Avaliação de Modelos (aplicar ML):**
   - Separar treino e teste antes de qualquer ajuste.
   - Rodar um baseline e comparar classificadores vistos em aula (Regressão Logística, KNN, Random Forest, XGBoost) com validação cruzada e ajuste de hiperparâmetros (`GridSearchCV`).
   - Avaliar no teste (métricas, Matriz de Confusão, Curva ROC), validar o modelo campeão com Nested CV, explicar o resultado com SHAP e salvar o pipeline final (`joblib`).

