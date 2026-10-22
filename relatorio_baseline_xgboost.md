# Relatório do Baseline XGBoost

Este relatório documenta a implementação e a avaliação inicial de um modelo de referência para previsão do preço semanal da gasolina. Os resultados descritos são os do experimento já executado; nenhum novo treinamento foi realizado para produzir este documento.

## 1. Objetivo

Um baseline é uma abordagem de referência contra a qual outras soluções podem ser comparadas. Sua função é estabelecer um resultado mensurável antes de avaliar modelos mais complexos e ajudar a identificar se uma nova abordagem realmente melhora o desempenho para a mesma tarefa.

Neste projeto, o baseline foi criado para prever o preço médio semanal da gasolina em um município. Posteriormente, seus resultados poderão ser comparados aos do sistema final baseado em GraphRAG + GenAI. Essa comparação deverá usar o mesmo alvo, períodos comparáveis e critérios adequados a cada tipo de pergunta. O baseline, por si só, não mede a qualidade de respostas subjetivas.

O problema tratado neste primeiro experimento é a previsão do preço médio semanal do produto identificado nos dados como `GASOLINA`, no município de São Paulo. O rótulo do arquivo é `GASOLINA`; os dados disponíveis não confirmam que ele seja equivalente a `GASOLINA C`.

## 2. Contexto do experimento

- **Produto:** `GASOLINA`, exatamente conforme registrado na coluna `Produto`.
- **Município:** `SAO PAULO`, no estado `SP`.
- **Período das observações selecionadas:** 2026-01-05 a 2026-08-31.
- **Arquivo-fonte:** `Dados_Brutos_Oil_Gas/Processados/etanol_gasolina_2026_teste_limpo.csv`.
- **Registros selecionados antes da agregação:** 7.252.
- **Semanas após agregação:** 35.

O recorte foi escolhido para ser um primeiro teste de previsão usando a base de teste de 2026 e uma combinação específica de produto e município. Restringir produto e localidade define um alvo consistente e reduz a mistura de mercados diferentes. A conversão para semanas produz uma série com menos pontos do que os registros individuais de coleta, adequada ao alvo semanal pretendido no experimento.

O arquivo tem outras colunas, como revenda, CNPJ, endereço, unidade de medida e bandeira. O modelo implementado não usa essas colunas como variáveis explicativas: filtra produto, município e estado, e usa data e valor de venda para preparar a série.

## 3. Preparação dos dados

O preparo descrito abaixo segue o que está implementado em `baseline_xgboost.py`:

1. **Leitura do CSV:** o arquivo é lido com `;` como separador, vírgula como separador decimal, codificação `utf-8-sig` e a coluna `Data da Coleta` convertida para data. Isso corresponde ao formato observado da fonte.
2. **Validação das colunas necessárias:** o script verifica a presença de `Produto`, `Municipio`, `Estado - Sigla`, `Data da Coleta` e `Valor de Venda`. A checagem evita continuar silenciosamente se o esquema esperado não estiver disponível.
3. **Filtro de produto e localidade:** valores textuais são removidos de espaços nas extremidades e convertidos para maiúsculas antes da comparação. O filtro mantém somente `Produto == GASOLINA`, `Municipio == SAO PAULO` e `Estado - Sigla == SP`. A condição do estado evita misturar cidades homônimas de outra unidade federativa.
4. **Seleção das colunas para a série:** depois do filtro, são mantidas apenas `Data da Coleta` e `Valor de Venda`; os demais campos não participam do cálculo do preço semanal.
5. **Conversão do preço:** `Valor de Venda` é convertido para número. Como a leitura usa `decimal=","`, valores como `6,99` são tratados no padrão decimal esperado. Datas ou preços ausentes/inválidos nos registros selecionados causam um erro explícito; o script não os descarta silenciosamente. Preços não positivos também causam erro.
6. **Início da semana:** cada data é associada à segunda-feira da mesma semana. Essa data de início passa a identificar a observação semanal.
7. **Agregação:** os valores de venda de todos os registros selecionados dentro de cada semana são combinados pela média aritmética. O alvo do modelo é, portanto, a média semanal dos preços observados, não o preço de um posto individual.
8. **Verificação de semanas sem dados:** é criada uma sequência semanal completa entre a primeira e a última semana. Se uma semana não tiver preço agregado, o script interrompe com erro, em vez de preencher ou interpolar o valor.
9. **Criação das linhas supervisionadas:** as features são combinadas com o preço semanal que se quer prever. A remoção por `dropna()` elimina as primeiras semanas para as quais não há histórico suficiente para calcular os lags e a média móvel. Não é uma remoção de linhas por dados originais ausentes: valores ausentes ou inválidos nas colunas-alvo já provocam erro antes desta etapa.

Foram obtidos 7.252 registros de `GASOLINA` em São Paulo/SP no período e 35 semanas com preço agregado. As primeiras quatro semanas não têm todos os históricos necessários para as features de lag 4 e média móvel de quatro semanas; por isso, 31 linhas ficam disponíveis para treino e teste depois da preparação supervisionada.

## 4. Por que frequência semanal?

**Interpretação metodológica:** a frequência semanal corresponde ao alvo pretendido — preço médio semanal — e resume múltiplas observações de revendas em um valor por período. Isso reduz a granularidade e a quantidade de linhas em comparação com os registros de coleta individuais.

Essa escolha também faz com que a previsão seja avaliada por semana, em vez de tratar cada preço de posto como se fosse uma observação temporal independente. A média, contudo, pode ocultar diferenças entre postos, regiões da cidade e composição das observações coletadas. O código implementa essa agregação, mas não demonstra que a frequência semanal seja superior a outras frequências; isso exigiria avaliação adicional.

## 5. Variáveis utilizadas pelo XGBoost

O código usa exatamente sete features:

| Feature | O que representa e por que foi usada | Informação temporal e risco de futuro |
|---|---|---|
| `tendencia_semanal` | Índice sequencial da semana na série, começando em zero. Fornece uma coordenada de tempo simples para o modelo. | Identifica a posição da semana prevista; não contém preço futuro. No teste, os índices extrapolam a faixa observada no treino, mas árvores de decisão não extrapolam uma tendência linear como faria uma reta. |
| `semana_seno` | Seno do número da semana ISO do ano, escalado pelo período de 52 semanas. | Usa apenas o calendário da semana prevista. Não utiliza valores de preço. A aproximação por 52 semanas não representa explicitamente anos com semana ISO 53. |
| `semana_cosseno` | Cosseno do número da semana ISO do ano, com a mesma escala de 52 semanas. Em conjunto com o seno, representa a posição cíclica no calendário anual. | Usa apenas o calendário. Não contém observações futuras de preço. |
| `preco_lag_1` | Preço médio semanal da semana imediatamente anterior. | Usa um preço passado em relação à semana-alvo. No teste walk-forward, semanas de teste já observadas podem fornecer histórico para semanas seguintes. |
| `preco_lag_2` | Preço médio de duas semanas antes da semana-alvo. | Usa preço passado; não usa o preço da semana prevista. |
| `preco_lag_4` | Preço médio de quatro semanas antes da semana-alvo. | Usa preço passado; não usa o preço da semana prevista. |
| `media_movel_4_semanas` | Média dos preços das quatro semanas anteriores à semana-alvo. O código aplica `shift(1)` antes da janela móvel de quatro observações. | Usa somente preços anteriores. O `shift(1)` exclui da janela o preço da própria semana-alvo e previne vazamento direto do alvo. |

Não há variáveis externas de câmbio, petróleo, impostos, oferta, demanda ou outras condições de mercado. A ausência dessas features limita o modelo ao calendário e ao histórico da própria série de preços.

### Uso de informações no teste

O código calcula os lags sobre toda a série temporal antes de separar treino e teste. Para as previsões de teste, o lag da segunda semana de teste pode incluir o preço **real observado** da primeira semana de teste; o mesmo ocorre sucessivamente nas semanas seguintes. Isso não usa o preço da própria semana prevista e é compatível com uma avaliação walk-forward de um passo, na qual uma nova previsão é feita após a observação da semana anterior.

Esse protocolo não equivale a prever as sete semanas de uma só vez na data de corte. Para uma previsão de horizonte fixo, os preços reais das semanas de teste não poderiam ser usados como entrada das previsões posteriores.

## 6. Divisão entre treino e teste

O script define a posição de corte como `int(35 * 0.8) = 28`, isto é, usa 80% das 35 semanas originais para localizar o início do teste. O corte é feito pela data, sem embaralhamento:

| Parte | Semanas supervisionadas | Período |
|---|---:|---|
| Treino | 24 | 2026-02-02 a 2026-07-13 |
| Teste | 7 | 2026-07-20 a 2026-08-31 |

A diferença entre as 28 semanas da posição do corte e as 24 semanas efetivamente usadas para treino decorre da remoção das quatro primeiras semanas sem features completas. Após essa remoção, há 31 linhas supervisionadas: 24 de treino e 7 de teste. Assim, a proporção observada nas linhas supervisionadas é aproximadamente 77,4% para treino e 22,6% para teste.

Em uma série temporal, o teste deve representar períodos posteriores aos usados para treinar. Uma divisão cronológica preserva essa ordem e simula, de forma simplificada, a previsão de observações futuras a partir do passado. Embaralhar as semanas poderia colocar informação de períodos posteriores no treino e períodos anteriores no teste, tornando a avaliação menos representativa do uso futuro.

## 7. Modelo XGBoost

XGBoost é uma implementação de boosting de árvores: árvores de decisão são adicionadas sequencialmente para reduzir erros da função objetivo. Neste experimento, foi escolhido como um modelo de referência de regressão para as features temporais e de histórico de preço. A escolha não implica que seja o melhor modelo para séries temporais nem que supere métodos mais simples.

O código usa a **API nativa** do XGBoost, criando matrizes `xgb.DMatrix` e chamando `xgb.train`; não usa a interface `XGBRegressor` do scikit-learn.

Parâmetros implementados:

| Parâmetro | Valor | Significado no experimento |
|---|---:|---|
| `objective` | `reg:squarederror` | Otimiza erro quadrático para regressão. |
| `eval_metric` | `rmse` | Métrica associada ao treinamento; o relatório final também calcula RMSE separadamente nas previsões. |
| `max_depth` | 2 | Limita a profundidade de cada árvore, restringindo a complexidade das divisões. |
| `eta` | 0,05 | Taxa de aprendizado que reduz a contribuição de cada árvore adicionada. |
| `num_boost_round` | 100 | Número fixo de iterações/árvores boosting; não há parada antecipada implementada. |
| `subsample` | 0,8 | Usa uma fração das linhas de treino em cada iteração. |
| `colsample_bytree` | 1,0 | Permite usar todas as features em cada árvore. |
| `lambda` | 1,0 | Regularização L2 dos pesos. |
| `seed` | 42 | Semente para tornar os componentes aleatórios do treinamento reproduzíveis. |
| `nthread` | 1 | Limita a execução a uma thread. |
| `tree_method` | `hist` | Usa o método baseado em histogramas para construir as árvores. |

Não há busca de hiperparâmetros, validação cruzada temporal nem early stopping no código. Com apenas 24 observações supervisionadas de treino, os resultados dependem de uma amostra pequena; os parâmetros devem ser entendidos como configuração inicial, não como configuração otimizada.

## 8. Como chegamos aos números

Para cada semana de teste, o script compara o preço médio semanal observado (`y_i`) com a previsão XGBoost (`ŷ_i`). Para este teste, `n = 7`.

### MAE

O erro absoluto médio (MAE) calcula a diferença absoluta entre previsão e valor real e tira a média:

`MAE = (1/n) × Σ |y_i - ŷ_i|`

Em palavras: calcula-se, para cada semana, quantos reais por litro a previsão ficou distante do valor real, sem distinguir erro acima ou abaixo; depois se calcula a média desses erros. O valor obtido pelo código e registrado na avaliação foi **R$ 0,0591/litro**.

### RMSE

A raiz do erro quadrático médio (RMSE) eleva os erros ao quadrado, calcula a média e extrai a raiz:

`RMSE = √[(1/n) × Σ (y_i - ŷ_i)²]`

Ele também é expresso em reais por litro. Ao elevar erros ao quadrado, dá peso relativamente maior aos erros maiores. O valor reportado para o XGBoost foi **R$ 0,06375/litro**.

O script calcula ambas as métricas usando os vetores de sete preços reais e sete previsões, e depois grava as previsões no CSV com seis casas decimais. Os MAE e RMSE acima são os valores registrados na execução/avaliação; recalculá-los com os valores arredondados do arquivo pode produzir diferença mínima de arredondamento.

Para o baseline ingênuo walk-forward, o valor previsto em cada semana é o preço real da semana imediatamente anterior. Na primeira semana de teste, usa-se o último preço semanal disponível no treino; nas semanas seguintes, usa-se o preço real da semana de teste anterior. As sete previsões ingênuas são comparadas com os mesmos sete valores reais. A avaliação anterior registrou **MAE de R$ 0,01858/litro** e **RMSE de R$ 0,02613/litro**.

**Nota de rastreabilidade:** o CSV de resultados contém as datas, os preços reais, as previsões XGBoost e os erros absolutos, mas não contém uma coluna de previsões ingênuas. Os valores do método ingênuo estão registrados em `avaliacao_baseline.md` e foram obtidos na comparação walk-forward anterior; não foram calculados pela execução XGBoost descrita no script.

## 9. Resultado das previsões

As sete previsões registradas no arquivo de saída são iguais a **R$ 6,417761/litro** (seis casas decimais). Os preços reais variaram entre R$ 6,334048 e R$ 6,396667 por litro no teste. A tabela mostra os valores diretamente do CSV:

| Data (início da semana) | Valor real (R$/litro) | Previsão XGBoost (R$/litro) | Erro absoluto (R$/litro) |
|---|---:|---:|---:|
| 2026-07-20 | 6,396667 | 6,417761 | 0,021094 |
| 2026-07-27 | 6,391524 | 6,417761 | 0,026237 |
| 2026-08-03 | 6,334048 | 6,417761 | 0,083713 |
| 2026-08-10 | 6,344667 | 6,417761 | 0,073094 |
| 2026-08-17 | 6,341190 | 6,417761 | 0,076570 |
| 2026-08-24 | 6,339143 | 6,417761 | 0,078618 |
| 2026-08-31 | 6,363697 | 6,417761 | 0,054063 |

O arquivo `avaliacao_baseline.md` descreve as previsões como aproximadamente R$ 6,41761/litro. O CSV de saída, que preserva seis casas decimais, registra R$ 6,417761; por isso, este relatório usa o valor do CSV e o distingue do arredondamento aproximado anterior.

Previsões constantes são importantes para interpretar as métricas: o erro médio isolado não revela que o modelo deixou de acompanhar a variação semanal. Uma árvore particiona o espaço de features em regiões e prevê valores associados às folhas. Em particular, o índice temporal do teste está além da maior posição temporal observada no treino; árvores não extrapolam automaticamente uma tendência além dessa faixa. Isso é uma interpretação compatível com o comportamento observado, não prova de que uma única causa explique toda a saída constante.

## 10. Comparação com o baseline ingênuo

O baseline ingênuo walk-forward prevê cada semana usando o preço real da semana imediatamente anterior. É uma referência simples de persistência: serve para verificar se o modelo treinado acrescenta valor em relação a manter como estimativa o último preço observado.

| Modelo | MAE | RMSE |
|---|---:|---:|
| XGBoost | R$ 0,0591/litro | R$ 0,06375/litro |
| Walk-forward | R$ 0,01858/litro | R$ 0,02613/litro |

**Neste experimento, o baseline ingênuo apresentou erros menores que o XGBoost** tanto em MAE quanto em RMSE. Essa comparação é walk-forward: ambos podem usar o valor observado da semana anterior em cada previsão semanal. Ela não deve ser confundida com uma previsão fixa de sete semanas feita de uma só vez no início do período de teste.

## 11. Interpretação

### Fatos observados

- O script leu os registros de `GASOLINA` de São Paulo/SP, agregou os preços por semana e treinou o XGBoost.
- O conjunto supervisionado teve 24 semanas de treino e 7 semanas de teste.
- As previsões gravadas no teste foram iguais a R$ 6,417761/litro nas sete semanas.
- Os erros registrados para o XGBoost foram maiores que os erros do baseline ingênuo walk-forward neste experimento.

### Interpretação

O baseline funcionou tecnicamente no sentido de executar o fluxo de preparação, treinamento, previsão, cálculo das métricas e gravação dos resultados. No entanto, suas previsões praticamente constantes não acompanharam as diferenças observadas entre semanas. A avaliação walk-forward mostrou que a regra simples de usar o preço anterior foi mais precisa neste recorte.

### Hipótese, não conclusão geral

O pequeno número de semanas de treino e a dificuldade de árvores em extrapolar podem ter contribuído para as previsões constantes e para o desempenho observado. São explicações plausíveis para esta configuração e esta amostra, mas não demonstram que XGBoost seja necessariamente inferior em geral. A conclusão limitada é que **este XGBoost, com estes dados e parâmetros, teve desempenho inferior ao walk-forward neste teste**.

## 12. Limitações

- Há apenas 24 semanas supervisionadas de treino e 7 semanas de teste.
- O conjunto temporal cobre menos de um ano; não permite avaliar ciclos anuais completos.
- As previsões XGBoost ficaram praticamente constantes no teste.
- O baseline ingênuo walk-forward teve desempenho melhor neste experimento.
- A série semanal de teste foi avaliada como previsões de um passo com atualização a cada semana, e não como um único horizonte fixo de sete semanas.
- O protocolo depende da disponibilidade do preço real da semana anterior para prever a próxima.
- Não foi feita validação cruzada temporal, busca de hiperparâmetros ou early stopping.
- O modelo usa somente tempo e histórico do próprio preço; não inclui fatores externos que possam estar relacionados à variação.
- A média municipal pode ocultar diferenças entre postos, bairros e composição das coletas.
- A última semana está identificada por 2026-08-31, data final da fonte. Não há informação posterior no conjunto para confirmar a completude de uma semana de segunda-feira a domingo.
- O arquivo identifica o produto como `GASOLINA`; a correspondência com `GASOLINA C` não está confirmada pelas colunas examinadas.
- O CSV de previsões registra XGBoost, não as previsões do método ingênuo. A avaliação ingênua está documentada separadamente.

Por esses motivos, o XGBoost atual deve ser tratado como **protótipo de referência**, e não como baseline definitivo.

## 13. Relação com o restante do projeto

O resultado atual fornece uma referência quantitativa inicial para a pergunta preditiva sobre preço de gasolina. Quando o sistema final estiver disponível, a comparação poderá ser organizada como:

```text
Baseline ML (XGBoost)
        ↓
resultado de referência

GraphRAG + GenAI
        ↓
resultado final a avaliar
```

Para perguntas preditivas, as previsões devem ser comparadas aos mesmos valores reais e períodos, com métricas como MAE e RMSE. O protocolo temporal e o horizonte também precisam ser iguais para que a comparação seja justa.

Para perguntas subjetivas, MAE e RMSE não são adequados. Devem ser avaliadas dimensões como relevância, coerência, uso correto dos dados e ausência de informação inventada. O resultado do sistema final não deve ser declarado melhor antes de realizar essas avaliações.

## 14. Próximos passos

Os itens abaixo são sugestões; não foram executados neste relatório:

- ampliar o conjunto temporal de treinamento e avaliação;
- avaliar outras configurações do XGBoost, mantendo a separação temporal;
- comparar diferentes abordagens de previsão, incluindo referências simples;
- definir um protocolo consistente para previsão de um passo ou de horizonte fixo;
- testar as perguntas oficiais do projeto com critérios apropriados para cada pergunta;
- comparar posteriormente os resultados com o sistema final GraphRAG + GenAI.

## 15. Resumo executivo

Foi implementado um baseline XGBoost para prever a média semanal do produto `GASOLINA` em São Paulo, usando dados de 2026-01-05 a 2026-08-31. O baseline foi criado para estabelecer uma referência quantitativa para uma comparação futura com GraphRAG + GenAI. Após filtrar 7.252 registros e agregá-los em 35 semanas, foram usadas 24 semanas para treino e 7 para teste. O XGBoost obteve MAE de R$ 0,0591/litro e RMSE de R$ 0,06375/litro, mas suas previsões ficaram constantes em R$ 6,417761/litro. O baseline ingênuo walk-forward obteve MAE de R$ 0,01858/litro e RMSE de R$ 0,02613/litro, desempenho melhor neste experimento. A principal descoberta é que o modelo mais complexo não superou a referência simples neste recorte. O XGBoost deve servir como protótipo de referência enquanto se amplia e robustece o protocolo de avaliação.

## Fontes dos resultados

Arquivos do projeto utilizados para produzir este relatório:

- `baseline_xgboost.py` — código de leitura, filtragem, agregação semanal, features, divisão temporal, treinamento, previsão e métricas XGBoost.
- `baseline_results/previsoes_gasolina_sao_paulo.csv` — previsões XGBoost, valores reais e erros absolutos do conjunto de teste.
- `avaliacao_baseline.md` — registro dos resultados anteriores, incluindo as métricas do baseline ingênuo walk-forward.
- `Dados_Brutos_Oil_Gas/Processados/etanol_gasolina_2026_teste_limpo.csv` — base original consultada para validar o esquema, a contagem e o período dos registros selecionados.
