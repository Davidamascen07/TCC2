import pytesseract
import numpy as np
import cv2
import re
import os
import json
from pathlib import Path
from typing import Dict, List

# Configuração para usar as mesmas imagens do projeto
imagens_dir = Path("data/raw/exemplos/imagens")
anotacoes_path = "data/raw/exemplos/anotacao.json"

# Verifica se o diretório existe
if not imagens_dir.exists():
    print(f"Erro: Diretório não encontrado: {imagens_dir}")
    exit()

def carregar_anotacoes():
    """Carrega as anotações dos comprovantes"""
    import json
    from pathlib import Path
    
    anotacoes_path = Path("data/raw/exemplos/anotacao.json")
    
    if not anotacoes_path.exists():
        print(f"❌ Arquivo de anotações não encontrado: {anotacoes_path}")
        return []
    
    with open(anotacoes_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['anotacoes']

# Configuração simplificada do Tesseract
config_tesseract = "--psm 6"

def OCR_processa_simples(img, config_tesseract=""):
    """Versão simples do OCR - similar ao código original"""
    try:
        # Tenta português primeiro
        texto = pytesseract.image_to_string(img, config=config_tesseract, lang='por')
    except:
        print("Aviso: Idioma português não disponível, usando inglês")
        try:
            texto = pytesseract.image_to_string(img, config=config_tesseract, lang='eng')
        except:
            texto = pytesseract.image_to_string(img, config=config_tesseract)
    return texto

def OCR_processa_multiplas_configs(img):
    """Testa múltiplas configurações do Tesseract"""
    configs = [
        "",  # Padrão
        "--psm 6",  # Bloco uniforme de texto
        "--psm 4",  # Coluna única de texto de tamanhos variados
        "--psm 3",  # Página totalmente automática, mas sem OSD
        "--psm 8",  # Palavra única
        "--psm 7",  # Linha única de texto
        "--psm 11", # Palavra esparsa
        "--psm 13"  # Linha crua. Trata a imagem como uma única linha de texto
    ]
    
    resultados = []
    for i, config in enumerate(configs):
        try:
            texto = pytesseract.image_to_string(img, config=config, lang='por')
            if texto.strip():  # Só adiciona se não estiver vazio
                resultados.append({
                    'config': config or 'padrão',
                    'config_num': i,
                    'texto': texto,
                    'tamanho': len(texto)
                })
        except Exception as e:
            print(f"Erro na config {i}: {e}")
    
    return resultados

def preprocessar_imagem_simples(img):
    """Pré-processamento simples da imagem"""
    # Converter para escala de cinza se necessário
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    return gray

def preprocessar_imagem_avancado(img):
    """Múltiplas versões de pré-processamento"""
    versoes = {}
    
    # Versão original em escala de cinza
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    versoes['original'] = gray
    
    # Versão com threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versoes['threshold'] = thresh
    
    # Versão com denoising
    try:
        denoised = cv2.fastNlMeansDenoising(gray)
        versoes['denoised'] = denoised
    except:
        # Fallback se não tiver o método
        versoes['denoised'] = gray
    
    # Versão com contraste melhorado
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    versoes['enhanced'] = enhanced
    
    return versoes

def extrair_dados_basicos(texto):
    """Extrai dados básicos do texto usando regex simples"""
    dados = {}
    
    # Valor
    valores = re.findall(r'r\$\s*(\d+[.,]\d{2})', texto, re.IGNORECASE)
    if valores:
        dados['valor'] = f"R$ {valores[0]}"
    
    # Data
    datas = re.findall(r'(\d{2})[\/\-](\d{2})[\/\-](\d{4})', texto)
    if datas:
        dados['data'] = f"{datas[0][0]}/{datas[0][1]}/{datas[0][2]}"
    
    # CPF
    cpfs = re.findall(r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}', texto)
    if cpfs:
        dados['cpfs'] = cpfs
    
    # ID transação
    ids = re.findall(r'E\d{10,20}[a-zA-Z0-9]+', texto, re.IGNORECASE)
    if ids:
        dados['id_transacao'] = ids[0]
    
    return dados

def testar_comprovante_simples(arquivo: str, anotacao: dict = None):
    """Testa um comprovante específico com OCR simples"""
    import cv2
    import pytesseract
    from pathlib import Path
    import re
    
    caminho_imagem = Path("data/raw/exemplos/imagens") / arquivo
    
    if not caminho_imagem.exists():
        print(f"   ❌ Arquivo não encontrado: {arquivo}")
        return None
    
    try:
        print(f"📄 Processando: {arquivo}")
        
        # Carregar e processar imagem
        img = cv2.imread(str(caminho_imagem), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"   ❌ Erro ao carregar imagem")
            return None
        
        # OCR simples
        texto = pytesseract.image_to_string(img, lang='por')
        
        # Extrair dados básicos
        dados_extraidos = extrair_dados_basicos(texto)
        
        # Preparar comparação se anotação disponível
        comparacao = None
        if anotacao:
            comparacao = {
                'valor': anotacao.get('valor', ''),
                'data': anotacao.get('data', ''),
                'destinatario': anotacao.get('destinatario', {}).get('nome', ''),
                'remetente': anotacao.get('remetente', {}).get('nome', '')
            }
        
        resultado = {
            'arquivo': arquivo,
            'tamanho': len(texto),
            'texto_preview': texto[:200] + "..." if len(texto) > 200 else texto,
            'dados_extraidos': dados_extraidos,
            'comparacao': comparacao
        }
        
        print(f"   ✅ Processado com sucesso ({len(texto)} caracteres)")
        return resultado
        
    except Exception as e:
        print(f"   ❌ Erro no processamento: {e}")
        return None

def extrair_dados_basicos(texto: str) -> dict:
    """Extrai dados básicos do texto usando regex simples"""
    import re
    
    dados = {}
    texto_lower = texto.lower()
    
    # Extrair valor
    valores = re.findall(r'r\$\s*[\d.,]+', texto_lower)
    if valores:
        dados['valor'] = valores[0].upper()
    
    # Extrair data
    datas = re.findall(r'\d{2}/\d{2}/\d{4}', texto)
    if datas:
        dados['data'] = datas[0]
    
    # Extrair nomes (simplificado)
    linhas = [linha.strip() for linha in texto.split('\n') if linha.strip()]
    nomes_candidatos = []
    
    for linha in linhas:
        # Procurar por linhas que parecem nomes
        if len(linha) > 5 and not re.search(r'\d', linha) and linha.count(' ') >= 1:
            nomes_candidatos.append(linha.title())
    
    if nomes_candidatos:
        dados['destinatario'] = nomes_candidatos[0] if len(nomes_candidatos) > 0 else ''
        dados['remetente'] = nomes_candidatos[1] if len(nomes_candidatos) > 1 else ''
    
    return dados

# corrigir warning: in the working copy of 'src/ocr/extractor2.py', LF will be replaced by CRLF the next time Git touches it
def main():
    """Função principal para testar os comprovantes"""
    print("🚀 TESTANDO OCR SIMPLES NOS COMPROVANTES")
    print("=" * 80)
    
    # Carregar anotações
    anotacoes = carregar_anotacoes()
    anotacoes_dict = {anot['arquivo_origem']: anot for anot in anotacoes}
    
    # Lista de comprovantes para testar (primeiros 5 como no exemplo)
    comprovantes_teste = [
        'comprovante_inter_58_00.jpg',
        'comprovante_pix_35_00.jpg', 
        'comprovante_nubank_19_50.jpg',
        'comprovante_nubank_40_00.jpg',
        'comprovante_btg_38_00.jpg'
    ]
    
    resultados = []
    
    for arquivo in comprovantes_teste:
        anotacao = anotacoes_dict.get(arquivo)
        resultado = testar_comprovante_simples(arquivo, anotacao)
        if resultado:
            resultados.append(resultado)
    
    # Resumo final
    print(f"\n🎯 RESUMO DOS TESTES:")
    print("=" * 80)
    
    for resultado in resultados:
        arquivo = resultado['arquivo']
        tamanho = resultado['tamanho']
        dados = resultado['dados_extraidos']
        
        print(f"📄 {arquivo}")
        print(f"   📏 Tamanho: {tamanho} caracteres")
        print(f"   🔧 Dados extraídos: {list(dados.keys())}")
        
        if resultado['comparacao']:
            # Calcular score simples
            score_itens = []
            
            # Valor
            valor_esperado = resultado['comparacao'].get('valor', '').lower().replace(' ', '')
            valor_extraido = dados.get('valor', '').lower().replace(' ', '')
            if valor_esperado == valor_extraido:
                score_itens.append('Valor ✅')
            else:
                score_itens.append('Valor ❌')
            
            # Data
            data_esperada = resultado['comparacao'].get('data', '')
            data_extraida = dados.get('data', '')
            if data_esperada == data_extraida:
                score_itens.append('Data ✅')
            else:
                score_itens.append('Data ❌')
            
            print(f"   📊 Avaliação: {' | '.join(score_itens)}")
        
        print()
    
    print("✅ Teste concluído!")

if __name__ == "__main__":
    main()