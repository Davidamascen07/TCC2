"""
Compara o método OCR simples com o método avançado
"""

import sys
from pathlib import Path
import json
from ocr.extractor2 import testar_comprovante_simples, carregar_anotacoes
from src.evaluation.evaluator import OCRExtractorEvaluator

def comparar_metodos(max_comprovantes=5):
    """Compara os dois métodos de OCR"""
    
    print("🔥 COMPARAÇÃO ENTRE MÉTODOS DE OCR")
    print("=" * 80)
    
    # Paths
    anotacoes_path = "data/raw/exemplos/anotacao.json"
    imagens_dir = "data/raw/exemplos/imagens"
    
    # Carregar anotações
    anotacoes = carregar_anotacoes()
    if not anotacoes:
        print("❌ Não foi possível carregar anotações")
        return
    
    # Testar apenas os primeiros comprovantes
    anotacoes_teste = anotacoes[:max_comprovantes]
    
    print(f"📊 Testando {len(anotacoes_teste)} comprovantes\n")
    
    # Configurar avaliador avançado
    evaluator = OCRExtractorEvaluator(anotacoes_path, imagens_dir)
    
    resultados_comparacao = []
    
    for i, anotacao in enumerate(anotacoes_teste, 1):
        arquivo = anotacao['arquivo_origem']
        
        print(f"📄 {i}. COMPARANDO: {arquivo}")
        print("-" * 60)
        
        # MÉTODO SIMPLES
        print("🔹 MÉTODO SIMPLES:")
        resultado_simples = testar_comprovante_simples(arquivo, anotacao)
        
        if resultado_simples:
            dados_simples = resultado_simples['dados_extraidos']
            tamanho_simples = resultado_simples['tamanho']
            
            print(f"   📏 Tamanho: {tamanho_simples} chars")
            print(f"   🔧 Dados: {list(dados_simples.keys())}")
        else:
            print("   ❌ Falhou")
            dados_simples = {}
            tamanho_simples = 0
        
        print()
        
        # MÉTODO AVANÇADO
        print("🔸 MÉTODO AVANÇADO:")
        try:
            resultado_avancado = evaluator.extractor.process_comprovante(
                str(Path(imagens_dir) / arquivo)
            )
            
            if resultado_avancado['status'] == 'success':
                texto_avancado = resultado_avancado['texto_extraido']['tesseract']
                dados_avancados = resultado_avancado['dados_estruturados']
                tamanho_avancado = len(texto_avancado)
                
                print(f"   📏 Tamanho: {tamanho_avancado} chars")
                print(f"   🔧 Dados: {list(dados_avancados.keys())}")
            else:
                print(f"   ❌ Erro: {resultado_avancado.get('erro', 'Desconhecido')}")
                dados_avancados = {}
                tamanho_avancado = 0
                
        except Exception as e:
            print(f"   💥 Exceção: {e}")
            dados_avancados = {}
            tamanho_avancado = 0
        
        print()
        
        # COMPARAÇÃO
        print("📊 COMPARAÇÃO:")
        
        # Tamanho do texto extraído
        if tamanho_simples > tamanho_avancado:
            print(f"   📏 Tamanho: SIMPLES VENCE ({tamanho_simples} vs {tamanho_avancado})")
        elif tamanho_avancado > tamanho_simples:
            print(f"   📏 Tamanho: AVANÇADO VENCE ({tamanho_avancado} vs {tamanho_simples})")
        else:
            print(f"   📏 Tamanho: EMPATE ({tamanho_simples})")
        
        # Quantidade de dados extraídos
        qtd_simples = len(dados_simples)
        qtd_avancados = len(dados_avancados)
        
        if qtd_simples > qtd_avancados:
            print(f"   🔧 Dados extraídos: SIMPLES VENCE ({qtd_simples} vs {qtd_avancados})")
        elif qtd_avancados > qtd_simples:
            print(f"   🔧 Dados extraídos: AVANÇADO VENCE ({qtd_avancados} vs {qtd_simples})")
        else:
            print(f"   🔧 Dados extraídos: EMPATE ({qtd_simples})")
        
        # Verificar acurácia básica
        valor_esperado = anotacao.get('valor', '').lower().replace(' ', '')
        data_esperada = anotacao.get('data', '')
        
        # Simples
        valor_simples = dados_simples.get('valor', '').lower().replace(' ', '')
        data_simples = dados_simples.get('data', '')
        acertos_simples = 0
        if valor_esperado == valor_simples:
            acertos_simples += 1
        if data_esperada == data_simples:
            acertos_simples += 1
        
        # Avançado
        valor_avancado = dados_avancados.get('valor', '').lower().replace(' ', '')
        data_avancada = dados_avancados.get('data', '')
        acertos_avancados = 0
        if valor_esperado == valor_avancado:
            acertos_avancados += 1
        if data_esperada == data_avancada:
            acertos_avancados += 1
        
        if acertos_simples > acertos_avancados:
            print(f"   🎯 Acurácia: SIMPLES VENCE ({acertos_simples}/2 vs {acertos_avancados}/2)")
        elif acertos_avancados > acertos_simples:
            print(f"   🎯 Acurácia: AVANÇADO VENCE ({acertos_avancados}/2 vs {acertos_simples}/2)")
        else:
            print(f"   🎯 Acurácia: EMPATE ({acertos_simples}/2)")
        
        resultados_comparacao.append({
            'arquivo': arquivo,
            'simples': {
                'tamanho': tamanho_simples,
                'dados': qtd_simples,
                'acertos': acertos_simples
            },
            'avancado': {
                'tamanho': tamanho_avancado,
                'dados': qtd_avancados,
                'acertos': acertos_avancados
            }
        })
        
        print("\n" + "=" * 80 + "\n")
    
    # RESUMO FINAL
    print("🏆 RESUMO FINAL:")
    print("=" * 80)
    
    vitorias_simples = {'tamanho': 0, 'dados': 0, 'acertos': 0}
    vitorias_avancado = {'tamanho': 0, 'dados': 0, 'acertos': 0}
    empates = {'tamanho': 0, 'dados': 0, 'acertos': 0}
    
    for resultado in resultados_comparacao:
        s = resultado['simples']
        a = resultado['avancado']
        
        # Tamanho
        if s['tamanho'] > a['tamanho']:
            vitorias_simples['tamanho'] += 1
        elif a['tamanho'] > s['tamanho']:
            vitorias_avancado['tamanho'] += 1
        else:
            empates['tamanho'] += 1
        
        # Dados
        if s['dados'] > a['dados']:
            vitorias_simples['dados'] += 1
        elif a['dados'] > s['dados']:
            vitorias_avancado['dados'] += 1
        else:
            empates['dados'] += 1
        
        # Acertos
        if s['acertos'] > a['acertos']:
            vitorias_simples['acertos'] += 1
        elif a['acertos'] > s['acertos']:
            vitorias_avancado['acertos'] += 1
        else:
            empates['acertos'] += 1
    
    print(f"📏 Tamanho do texto: Simples {vitorias_simples['tamanho']} | Avançado {vitorias_avancado['tamanho']} | Empates {empates['tamanho']}")
    print(f"🔧 Dados extraídos: Simples {vitorias_simples['dados']} | Avançado {vitorias_avancado['dados']} | Empates {empates['dados']}")
    print(f"🎯 Acurácia: Simples {vitorias_simples['acertos']} | Avançado {vitorias_avancado['acertos']} | Empates {empates['acertos']}")
    
    # Determinar vencedor geral
    pontos_simples = sum(vitorias_simples.values())
    pontos_avancado = sum(vitorias_avancado.values())
    
    print(f"\n🏆 VENCEDOR GERAL:")
    if pontos_simples > pontos_avancado:
        print(f"   🥇 MÉTODO SIMPLES ({pontos_simples} vs {pontos_avancado} pontos)")
    elif pontos_avancado > pontos_simples:
        print(f"   🥇 MÉTODO AVANÇADO ({pontos_avancado} vs {pontos_simples} pontos)")
    else:
        print(f"   🤝 EMPATE ({pontos_simples} pontos cada)")

def main():
    max_comprovantes = 5
    if len(sys.argv) > 1:
        try:
            max_comprovantes = int(sys.argv[1])
        except ValueError:
            print("Uso: python compare_ocr_methods.py [numero_comprovantes]")
            return
    
    comparar_metodos(max_comprovantes)

if __name__ == "__main__":
    main()
