from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

pasta = Path("Dados/Treinamento/Anuário Estatístico - 2026")

saida = Path("Dados/Processados/Anuario")
saida.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOCALIZAR EXCELS
# ============================================================

arquivos = sorted(pasta.glob("*.xlsx"))

print("Arquivos encontrados:")
print(len(arquivos))


# ============================================================
# PROCESSAR CADA ARQUIVO
# ============================================================

for arquivo in arquivos:

    print("\n" + "="*60)
    print("Processando:", arquivo.name)

    excel = pd.ExcelFile(arquivo)

    for aba in excel.sheet_names:

        df = pd.read_excel(
            arquivo,
            sheet_name=aba
        )


        # ignorar abas vazias
        if df.empty:
            print("Aba vazia ignorada:", aba)
            continue


        print("\nAba:", aba)
        print("Tamanho inicial:", df.shape)


        # remover linhas totalmente vazias
        df = df.dropna(
            how="all"
        )


        # remover colunas totalmente vazias
        df = df.dropna(
            axis=1,
            how="all"
        )


        # limpar espaços em textos
        colunas_texto = df.select_dtypes(
            include="object"
        ).columns

        for coluna in colunas_texto:

            df[coluna] = (
                df[coluna]
                .astype(str)
                .str.strip()
            )


        print("Tamanho final:", df.shape)


        # nome do arquivo de saída

        nome = (
            arquivo.stem
            + "_"
            + aba
            + "_limpo.csv"
        )

        caminho_saida = saida / nome


        df.to_csv(
            caminho_saida,
            sep=";",
            decimal=",",
            index=False,
            encoding="utf-8-sig"
        )


        print("Salvo em:")
        print(caminho_saida)


print("\nProcessamento finalizado.")