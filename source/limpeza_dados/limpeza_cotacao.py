from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

pasta = Path("Dados/Treinamento/Cotação do Dólar - BASEN PTAX")

pasta_saida = Path("Dados/Processados")
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivo_saida = pasta_saida / "cotacao_dolar_ptax_limpa.csv"


# ============================================================
# 2. LOCALIZAR ARQUIVOS
# ============================================================

arquivos = list(pasta.glob("*.csv"))

if not arquivos:
    raise SystemExit(f"Nenhum CSV encontrado em {pasta}")

print("\nArquivos encontrados:")
print(len(arquivos))


# ============================================================
# 3. LER E CONSOLIDAR CSVs
# ============================================================

dados = []

for arquivo in arquivos:

    df = pd.read_csv(
        arquivo,
        sep=";",
        decimal=",",
        header=None,
        names=[
            "Data",
            "Codigo_Moeda",
            "Tipo_Cotacao",
            "Moeda",
            "Compra",
            "Venda",
            "Paridade_Compra",
            "Paridade_Venda"
        ],
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

print("\nLinhas totalmente vazias:")
print(
    dados_completos.isna()
    .all(axis=1)
    .sum()
)

dados_completos = dados_completos.dropna(
    how="all"
)


# ============================================================
# 5. LIMPAR TEXTOS
# ============================================================

colunas_texto = dados_completos.select_dtypes(
    include="object"
).columns

for coluna in colunas_texto:

    dados_completos[coluna] = (
        dados_completos[coluna]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 6. CORRIGIR E CONVERTER DATA
# ============================================================

# Adiciona zero à esquerda quando necessário:
# Exemplo:
# 1102025 -> 01102025

dados_completos["Data"] = (
    dados_completos["Data"]
    .astype(str)
    .str.zfill(8)
)


dados_completos["Data"] = pd.to_datetime(
    dados_completos["Data"],
    format="%d%m%Y",
    errors="coerce"
)


print("\nDatas inválidas:")
print(
    dados_completos["Data"]
    .isna()
    .sum()
)


# ============================================================
# 7. CONVERTER VALORES NUMÉRICOS
# ============================================================

for coluna in [
    "Compra",
    "Venda",
    "Paridade_Compra",
    "Paridade_Venda"
]:

    dados_completos[coluna] = pd.to_numeric(
        dados_completos[coluna],
        errors="coerce"
    )


# ============================================================
# 8. REMOVER DUPLICATAS
# ============================================================

print("\nDuplicatas antes da remoção:")
print(
    dados_completos.duplicated()
    .sum()
)


dados_completos = (
    dados_completos
    .drop_duplicates()
    .reset_index(drop=True)
)


# ============================================================
# 9. VALIDAÇÃO FINAL
# ============================================================

print("\nTamanho final:")
print(dados_completos.shape)

print("\nTipos:")
print(dados_completos.dtypes)

print("\nValores ausentes:")
print(dados_completos.isna().sum())

print("\nDuplicatas restantes:")
print(
    dados_completos.duplicated()
    .sum()
)


# ============================================================
# 10. SALVAR
# ============================================================

dados_completos.to_csv(
    arquivo_saida,
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)


print("\nArquivo salvo:")
print(arquivo_saida)