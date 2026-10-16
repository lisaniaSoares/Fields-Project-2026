import os
import pandas as pd
from llama_index.core import SimpleDirectoryReader

# =====================================================================
# 1. CONFIGURAÇÃO DOS CAMINHOS RELATIVOS
# =====================================================================
PATH_BRUTOS = r"..\Dados_Brutos_Oil_Gas\Treinamento"
PATH_INPUT_GRAPHRAG = r".\input"

os.makedirs(PATH_INPUT_GRAPHRAG, exist_ok=True)

print("⚙️ Iniciando o processador robusto anti-erros de tokenização...")
print(f"📂 Buscando dados em: {PATH_BRUTOS}")
print(f"📥 Salvando arquivos .txt limpos em: {PATH_INPUT_GRAPHRAG}\n")

arquivos_gerados = 0

# =====================================================================
# 2. TRATAMENTO RECURSIVO DE PLANILHAS (CSV) COM DETECÇÃO DE SEPARADOR
# =====================================================================
for raiz, pastas, arquivos in os.walk(PATH_BRUTOS):
    for arquivo in arquivos:
        if arquivo.endswith('.csv'):
            caminho_completo = os.path.join(raiz, arquivo)
            nome_subpasta = os.path.basename(raiz)
            print(f"📊 Convertendo CSV: [{nome_subpasta}] -> {arquivo}")
            
            df = None
            # Estratégia Avançada: Testa combinações de separadores comuns em Oil & Gas BR
            for separador in [';', ',', '\t']:
                for codificacao in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        # on_bad_lines='skip' ignora linhas malformadas (comentários ou cabeçalhos duplos) sem quebrar o script
                        df = pd.read_csv(
                            caminho_completo, 
                            sep=separador, 
                            encoding=codificacao, 
                            on_bad_lines='skip'
                        )
                        # Se leu e trouxe mais de uma coluna, o separador está correto!
                        if len(df.columns) > 1:
                            break
                    except Exception:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            
            if df is None or len(df.columns) <= 1:
                print(f"⚠️ Não foi possível determinar o formato correto do CSV {arquivo}. Tentando leitura forçada...")
                try:
                    df = pd.read_csv(caminho_completo, sep=None, engine='python', on_bad_lines='skip')
                except Exception as e:
                    print(f"❌ Falha crítica ao ler {arquivo}: {e}")
                    continue

            try:
                linhas_texto = []
                # Remove colunas ou linhas completamente vazias que geram ruído no Grafo
                df = df.dropna(how='all')
                
                for idx, row in df.iterrows():
                    detalhes_linha = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    if detalhes_linha.strip():
                        texto_narrativo = f"Dado de {nome_subpasta} (Origem {arquivo} - Registro {idx}): {detalhes_linha}.\n"
                        linhas_texto.append(texto_narrativo)
                
                if linhas_texto:
                    nome_txt = f"{nome_subpasta}_{arquivo.replace('.csv', '_csv_processado.txt')}"
                    # Remove caracteres inválidos para nomes de arquivos
                    nome_txt = "".join([c for c in nome_txt if c.isalpha() or c.isdigit() or c in '._- '])
                    caminho_saida_txt = os.path.join(PATH_INPUT_GRAPHRAG, nome_txt)
                    
                    with open(caminho_saida_txt, "w", encoding="utf-8") as f:
                        f.writelines(linhas_texto)
                    
                    print(f"✅ Sucesso! Gerado: {nome_txt}")
                    arquivos_gerados += 1
                else:
                    print(f"⚠️ Arquivo {arquivo} processado, mas não continha dados válidos.")
                
            except Exception as e:
                print(f"❌ Erro ao converter dados de {arquivo} para texto: {e}")

# =====================================================================
# 3. TRATAMENTO RECURSIVO DE PDFs
# =====================================================================
print("\n📄 Extraindo texto estruturado dos PDFs de treinamento...")
try:
    reader = SimpleDirectoryReader(input_dir=PATH_BRUTOS, required_exts=[".pdf"], recursive=True)
    pdf_docs = reader.load_data()
    
    documentos_agrupados = {}
    for doc in pdf_docs:
        nome_origem = doc.metadata.get('file_name', 'documento_pdf.pdf')
        nome_origem_limpo = os.path.basename(nome_origem)
        if nome_origem_limpo not in documentos_agrupados:
            documentos_agrupados[nome_origem_limpo] = []
        documentos_agrupados[nome_origem_limpo].append(doc.text)
        
    for nome_origem_limpo, paginas in documentos_agrupados.items():
        nome_txt = nome_origem_limpo.replace('.pdf', '_pdf_convertido.txt')
        caminho_saida_txt = os.path.join(PATH_INPUT_GRAPHRAG, nome_txt)
        
        texto_completo_pdf = "\n\n--- Nova Página ---\n\n".join(paginas)
        
        with open(caminho_saida_txt, "w", encoding="utf-8") as f:
            f.write(texto_completo_pdf)
            
        print(f"✅ PDF transformado com sucesso: {nome_txt}")
        arquivos_gerados += 1

except Exception as e:
    print(f"⚠️ Nota ou aviso sobre processamento de PDFs: {e}")

print("\n=======================================================")
print(f"✨ Concluído! {arquivos_gerados} arquivos de Treinamento foram limpos e injetados.")
print("🚀 Próximo passo: Execute 'graphrag index --root .' no seu terminal Conda!")
print("=======================================================")
