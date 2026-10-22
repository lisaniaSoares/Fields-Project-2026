# Avaliação do baseline

Este documento registra os resultados iniciais do baseline de previsão e estabelece critérios para comparação futura com o modelo final baseado em GraphRAG e GenAI.

## Experimento XGBoost

- **Produto:** GASOLINA
- **Município:** São Paulo
- **Período dos dados:** 2026-01-05 a 2026-08-31
- **Treino:** 24 semanas
- **Teste:** 7 semanas
- **MAE:** R$ 0,0591/litro
- **RMSE:** R$ 0,06375/litro
- **Previsões no teste:** aproximadamente R$ 6,41761/litro

## Resultados iniciais

| Modelo | MAE | RMSE | Observação |
|---|---:|---:|---|
| XGBoost | R$ 0,0591/litro | R$ 0,06375/litro | As previsões no teste foram praticamente constantes, em aproximadamente R$ 6,41761/litro. |
| Baseline ingênuo walk-forward | R$ 0,01858/litro | R$ 0,02613/litro | Teve desempenho melhor que o XGBoost neste experimento. |

## Comparação futura

| Abordagem | Tipo | Pergunta | Métrica | Resultado | Observação |
|---|---|---|---|---|---|
| XGBoost | ML / baseline | Previsão do preço da gasolina | MAE/RMSE | Já disponível | Baseline inicial |
| GraphRAG + GenAI | Modelo final | A definir | A definir | A definir | Comparação futura |

## Critérios de avaliação

1. **Perguntas preditivas:** comparar a previsão com o valor real usando MAE e RMSE.
2. **Perguntas subjetivas:** avaliar relevância, coerência, uso correto dos dados e ausência de informações inventadas.
3. Não afirmar que o modelo final é melhor antes de comparar os resultados.
4. Registrar as limitações do experimento ao interpretar e comparar os resultados.

## Limitações atuais

- O conjunto de treino possui apenas 24 semanas.
- O XGBoost apresentou previsões praticamente constantes no teste.
- O baseline ingênuo teve desempenho melhor neste experimento.
- Portanto, o XGBoost atual deve ser tratado como protótipo de referência, não como baseline definitivo.
