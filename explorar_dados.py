from pathlib import Path
import argparse
import re

import pandas as pd


LIMITE_CATEGORIAS = 20


def ler_csv(arquivo: Path) -> pd.DataFrame:
    """Lê CSVs com separador detectado, sem alterar os valores carregados."""
    ultimo_erro = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            primeira_linha = arquivo.read_text(encoding=encoding).splitlines()[0]
            primeiro_campo = re.split(r"[;,\t]", primeira_linha)[0].strip()
            sem_cabecalho = bool(re.fullmatch(r"\d{8}", primeiro_campo))
            opcoes = {
                "sep": None,
                "engine": "python",
                "decimal": ",",
                "encoding": encoding,
            }
            if sem_cabecalho:
                quantidade_colunas = len(re.split(r"[;,\t]", primeira_linha))
                opcoes["header"] = None
                opcoes["names"] = [
                    f"coluna_{indice}"
                    for indice in range(1, quantidade_colunas + 1)
                ]
            return pd.read_csv(
                arquivo,
                **opcoes,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as erro:
            ultimo_erro = erro
    raise ultimo_erro


def identificar_datas(dados):
    colunas = [
        coluna for coluna in dados.columns
        if any(palavra in str(coluna).lower() for palavra in ("data", "date"))
    ]
    for coluna in dados.columns:
        valores = dados[coluna].dropna().astype(str).str.strip()
        if coluna in colunas or valores.empty:
            continue
        parecem_datas = valores.str.fullmatch(r"\d{8}")
        if parecem_datas.mean() >= 0.8:
            colunas.append(coluna)
    return colunas


def imprimir_relatorio(pasta: Path, arquivos, tabelas, colunas_por_arquivo):
    dados = pd.concat(tabelas, ignore_index=True, sort=False)
    colunas_referencia = colunas_por_arquivo[0]
    mesmas_colunas = all(colunas == colunas_referencia for colunas in colunas_por_arquivo)

    print("\n" + "=" * 72)
    print(f"PASTA: {pasta}")
    print(f"Arquivos CSV: {len(arquivos)}")
    print(f"Mesmas colunas em todos os arquivos: {'SIM' if mesmas_colunas else 'NAO'}")
    if not mesmas_colunas:
        for arquivo, colunas in zip(arquivos, colunas_por_arquivo):
            if colunas != colunas_referencia:
                print(f"  Colunas diferentes em {arquivo.name}: {list(colunas)}")

    print(f"Consolidado: {len(dados)} linhas x {len(dados.columns)} colunas")
    print("Colunas: " + ", ".join(map(str, dados.columns)))

    print("\nTipos das colunas:")
    for coluna, tipo in dados.dtypes.items():
        print(f"  - {coluna}: {tipo}")

    ausentes = dados.isna().sum()
    percentual = (ausentes / len(dados) * 100) if len(dados) else ausentes
    print("\nValores ausentes por coluna (quantidade; percentual):")
    for coluna in dados.columns:
        print(f"  - {coluna}: {ausentes[coluna]}; {percentual[coluna]:.2f}%")

    print(f"\nDuplicatas exatas: {dados.duplicated().sum()}")

    print("\nValores únicos de colunas categóricas com poucos valores:")
    encontrou_categorica = False
    for coluna in dados.select_dtypes(include=["object", "category", "bool"]).columns:
        valores = dados[coluna].dropna().unique()
        if len(valores) <= LIMITE_CATEGORIAS:
            encontrou_categorica = True
            print(f"  - {coluna} ({len(valores)}): {list(valores)}")
    if not encontrou_categorica:
        print("  - Nenhuma encontrada.")

    print("\nDatas inválidas:")
    colunas_data = identificar_datas(dados)
    encontrou_data = False
    for coluna in colunas_data:
        valores = dados[coluna].dropna()
        if valores.astype(str).str.fullmatch(r"\d{8}").mean() >= 0.8:
            convertidos = pd.to_datetime(valores, format="%d%m%Y", errors="coerce")
        else:
            convertidos = pd.to_datetime(valores, dayfirst=True, errors="coerce")
        invalidas = int(convertidos.isna().sum())
        encontrou_data = True
        print(f"  - {coluna}: {invalidas} inválidas em {len(valores)} valores preenchidos")
    if not encontrou_data:
        print("  - Nenhuma coluna de data identificada.")


def analisar_pasta(pasta: Path):
    arquivos = sorted(pasta.glob("*.csv"))
    if not arquivos:
        return

    tabelas = []
    colunas_por_arquivo = []
    for arquivo in arquivos:
        tabela = ler_csv(arquivo)
        tabelas.append(tabela)
        colunas_por_arquivo.append(tuple(tabela.columns))

    imprimir_relatorio(pasta, arquivos, tabelas, colunas_por_arquivo)


def main():
    parser = argparse.ArgumentParser(description="Relatório exploratório dos CSVs de treinamento.")
    parser.add_argument("--pasta", type=Path, default=Path("Dados/Treinamento"))
    args = parser.parse_args()

    pastas = sorted({arquivo.parent for arquivo in args.pasta.rglob("*.csv")})
    if not pastas:
        raise SystemExit(f"Nenhum CSV encontrado em {args.pasta}")

    print(f"Pastas com CSVs encontradas: {len(pastas)}")
    for pasta in pastas:
        analisar_pasta(pasta)


if __name__ == "__main__":
    main()
