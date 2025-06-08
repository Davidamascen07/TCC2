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

# Carrega as anotações para comparação
def carregar_anotacoes():
    try:
        with open(anotacoes_path, 'r', encoding='utf-8') as f:
            return json.load(f)['anotacoes']
    except FileNotFoundError:
        print(f"Arquivo de anotações não encontrado: {anotacoes_path}")
        return []

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

def testar_comprovante_simples(arquivo_imagem, anotacao_esperada=None):
    """Testa um comprovante com o método simples"""
    print(f"\n{'='*60}")
    print(f"📄 TESTANDO: {arquivo_imagem}")
    print(f"{'='*60}")
    
    caminho_completo = imagens_dir / arquivo_imagem
    
    if not caminho_completo.exists():
        print(f"❌ Arquivo não encontrado: {caminho_completo}")
        return None
    
    # Carregar imagem
    img = cv2.imread(str(caminho_completo))
    if img is None:
        print(f"❌ Erro ao carregar imagem: {caminho_completo}")
        return None
    
    print(f"📏 Dimensões da imagem: {img.shape}")
    
    # Teste 1: OCR simples na imagem original
    print(f"\n🔍 TESTE 1: OCR Simples (imagem original)")
    gray = preprocessar_imagem_simples(img)
    texto_simples = OCR_processa_simples(gray, "--psm 6")
    print(f"📝 Tamanho do texto: {len(texto_simples)} caracteres")
    print(f"🎯 Primeiros 200 chars: {texto_simples[:200]}...")
    
    # Teste 2: Múltiplas configurações
    print(f"\n🔍 TESTE 2: Múltiplas Configurações")
    resultados_configs = OCR_processa_multiplas_configs(gray)
    
    melhor_resultado = None
    if resultados_configs:
        melhor_resultado = max(resultados_configs, key=lambda x: x['tamanho'])
        print(f"🏆 Melhor config: {melhor_resultado['config']} ({melhor_resultado['tamanho']} chars)")
        
        for res in resultados_configs[:3]:  # Mostra top 3
            print(f"   Config {res['config_num']} ({res['config']}): {res['tamanho']} chars")
    
    # Teste 3: Múltiplas versões de pré-processamento
    print(f"\n🔍 TESTE 3: Diferentes Pré-processamentos")
    versoes = preprocessar_imagem_avancado(img)
    
    melhores_por_versao = {}
    for nome_versao, img_processada in versoes.items():
        try:
            texto_versao = OCR_processa_simples(img_processada, "--psm 6")
            melhores_por_versao[nome_versao] = {
                'texto': texto_versao,
                'tamanho': len(texto_versao)
            }
            print(f"   {nome_versao}: {len(texto_versao)} chars")
        except Exception as e:
            print(f"   {nome_versao}: ERRO - {e}")
    
    # Escolher o melhor texto
    if melhores_por_versao:
        melhor_versao = max(melhores_por_versao.items(), key=lambda x: x[1]['tamanho'])
        texto_final = melhor_versao[1]['texto']
        print(f"🎯 Melhor versão: {melhor_versao[0]} ({melhor_versao[1]['tamanho']} chars)")
    else:
        texto_final = texto_simples
        print(f"🎯 Usando texto simples ({len(texto_final)} chars)")
    
    # Extrair dados estruturados
    print(f"\n🔧 EXTRAÇÃO DE DADOS:")
    dados_extraidos = extrair_dados_basicos(texto_final)
    
    for campo, valor in dados_extraidos.items():
        print(f"   {campo}: {valor}")
    
    # Comparar com anotação esperada se disponível
    if anotacao_esperada:
        print(f"\n📊 COMPARAÇÃO COM DADOS ESPERADOS:")
        
        # Valor
        valor_esperado = anotacao_esperada.get('valor', '')
        valor_extraido = dados_extraidos.get('valor', '')
        valor_correto = valor_esperado.lower().replace(' ', '') == valor_extraido.lower().replace(' ', '')
        print(f"   💰 Valor: {valor_extraido} {'✅' if valor_correto else '❌'} (esperado: {valor_esperado})")
        
        # Data
        data_esperada = anotacao_esperada.get('data', '')
        data_extraida = dados_extraidos.get('data', '')
        data_correta = data_esperada == data_extraida
        print(f"   📅 Data: {data_extraida} {'✅' if data_correta else '❌'} (esperado: {data_esperada})")
        
        # CPFs
        cpfs_extraidos = dados_extraidos.get('cpfs', [])
        cpf_dest = anotacao_esperada.get('destinatario', {}).get('cpf', '')
        cpf_rem = anotacao_esperada.get('remetente', {}).get('cpf', '')
        
        cpf_dest_ok = cpf_dest in cpfs_extraidos if cpf_dest else False
        cpf_rem_ok = cpf_rem in cpfs_extraidos if cpf_rem else False
        
        print(f"   🆔 CPFs extraídos: {len(cpfs_extraidos)} - Dest: {'✅' if cpf_dest_ok else '❌'} Rem: {'✅' if cpf_rem_ok else '❌'}")
    
    # Mostrar texto completo
    print(f"\n📄 TEXTO COMPLETO EXTRAÍDO:")
    print(f"{'─'*60}")
    print(texto_final)
    print(f"{'─'*60}")
    
    return {
        'arquivo': arquivo_imagem,
        'texto_final': texto_final,
        'tamanho': len(texto_final),
        'dados_extraidos': dados_extraidos,
        'comparacao': anotacao_esperada
    }

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

# corrigir warning: in the working copy of 'src/ocr/extractor2.py', LF will be replaced by CRLF the next time Git touches it