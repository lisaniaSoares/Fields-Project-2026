from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

pasta = Path("Dados/Teste/Óleo Diesel + GNV")

pasta_saida = Path("Dados/Processados")
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivo_saida = pasta_saida / "diesel_gnv_2026_teste_limpo.csv"

# ============================================================
# 2. LOCALIZAR CSVs
# ============================================================

arquivos = list(pasta.glob("*.csv"))

if not arquivos:
    raise SystemExit(f"Nenhum CSV encontrado em {pasta}")

print("\nQuantidade de arquivos encontrados:")
print(len(arquivos))


# ============================================================
# 3. CONSOLIDAR ARQUIVOS
# ============================================================

dados = []

for arquivo in arquivos:

    df = pd.read_csv(
        arquivo,
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
    )

    dados.append(df)


dados_completos = pd.concat(
    dados,
    ignore_index=True
)

print("\nTamanho inicial:")
print(dados_completos.shape)


# ============================================================
# 4. REMOVER LINHAS TOTALMENTE VAZIAS
# ============================================================

linhas_vazias = dados_completos.isna().all(axis=1).sum()

print("\nLinhas totalmente vazias:")
print(linhas_vazias)

dados_completos = dados_completos.dropna(
    how="all"
)


# ============================================================
# 5. LIMPAR ESPAÇOS DOS TEXTOS
# ============================================================

colunas_texto = dados_completos.select_dtypes(
    include="object"
).columns

for coluna in colunas_texto:

    dados_completos[coluna] = (
        dados_completos[coluna]
        .str.strip()
    )


# ============================================================
# 6. TRANSFORMAR TEXTOS VAZIOS EM AUSENTES
# ============================================================

for coluna in colunas_texto:

    dados_completos[coluna] = (
        dados_completos[coluna]
        .replace("", pd.NA)
    )


# ============================================================
# 7. REMOVER DUPLICATAS
# ============================================================

print("\nDuplicatas antes da remoção:")
print(dados_completos.duplicated().sum())

dados_completos = (
    dados_completos
    .drop_duplicates()
    .reset_index(drop=True)
)


# ============================================================
# 8. REMOVER VALOR DE COMPRA
# ============================================================

if "Valor de Compra" in dados_completos.columns:

    print(
        "\nPercentual vazio em Valor de Compra:"
    )

    percentual = (
        dados_completos["Valor de Compra"]
        .isna()
        .mean()
        * 100
    )

    print(f"{percentual:.2f}%")

    if dados_completos["Valor de Compra"].isna().all():

        dados_completos = dados_completos.drop(
            columns=["Valor de Compra"]
        )

        print("Valor de Compra removido.")


# ============================================================
# 9. REMOVER COMPLEMENTO
# ============================================================

if "Complemento" in dados_completos.columns:

    percentual = (
        dados_completos["Complemento"]
        .isna()
        .mean()
        * 100
    )

    print(
        "\nPercentual vazio em Complemento:"
    )

    print(f"{percentual:.2f}%")

    if percentual > 70:

        dados_completos = dados_completos.drop(
            columns=["Complemento"]
        )

        print("Complemento removido.")


# ============================================================
# 10. CONVERTER DATA
# ============================================================

dados_completos["Data da Coleta"] = pd.to_datetime(
    dados_completos["Data da Coleta"],
    format="%d/%m/%Y",
    errors="coerce"
)

print("\nDatas inválidas:")
print(
    dados_completos["Data da Coleta"]
    .isna()
    .sum()
)


# ============================================================
# 11. VALIDAÇÃO FINAL
# ============================================================

print("\nTamanho final:")
print(dados_completos.shape)

print("\nValores ausentes:")
print(dados_completos.isna().sum())

print("\nProdutos:")
print(dados_completos["Produto"].unique())

print("\nUnidades:")
print(dados_completos["Unidade de Medida"].unique())

print("\nDuplicatas restantes:")
print(dados_completos.duplicated().sum())


# ============================================================
# 12. SALVAR
# ============================================================

dados_completos.to_csv(
    arquivo_saida,
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)

print("\nBase limpa salva em:")
print(arquivo_saida)