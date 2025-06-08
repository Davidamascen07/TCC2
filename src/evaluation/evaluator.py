import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import difflib
from datetime import datetime

from src.ocr.extractor import ComprovantePIXExtractor

@dataclass
class ExtractionMetrics:
    """Métricas de avaliação da extração REAL"""
    arquivo: str
    banco_correto: bool
    valor_correto: bool
    data_correta: bool
    cpf_destinatario_correto: bool
    cpf_remetente_correto: bool
    nome_destinatario_similar: float
    nome_remetente_similar: float
    id_transacao_correto: bool
    texto_extraido_qualidade: float
    score_total: float
    dados_extraidos_reais: Dict  # Novos dados reais extraídos

class OCRExtractorEvaluator:
    def __init__(self, anotacoes_path: str, imagens_dir: str):
        self.anotacoes_path = anotacoes_path
        self.imagens_dir = Path(imagens_dir)
        self.extractor = ComprovantePIXExtractor()
        self.anotacoes = self._load_anotacoes()

    def _load_anotacoes(self) -> Dict:
        """Carrega o arquivo de anotações"""
        try:
            with open(self.anotacoes_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Arquivo de anotações não encontrado: {self.anotacoes_path}")
            return {'anotacoes': []}
        except Exception as e:
            print(f"⚠️ Erro ao carregar anotações: {e}")
            return {'anotacoes': []}
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparação"""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    def _normalize_valor(self, valor: str) -> str:
        """Normaliza valores monetários"""
        if not valor:
            return ""
        # Remove caracteres não numéricos exceto vírgula e ponto
        valor_clean = re.sub(r'[^\d,.]', '', valor)
        # Padronizar formato para comparação
        if ',' in valor_clean:
            return valor_clean.replace('.', '').replace(',', '.')
        return valor_clean
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calcula similaridade entre dois textos"""
        if not text1 or not text2:
            return 0.0
        return difflib.SequenceMatcher(None, 
                                     self._normalize_text(text1), 
                                     self._normalize_text(text2)).ratio()
    
    def _extract_valor_from_data(self, dados: Dict) -> str:
        """Extrai valor dos dados estruturados"""
        if 'valor' in dados:
            return dados['valor']
        return ""
    
    def _extract_data_from_data(self, dados: Dict) -> str:
        """Extrai data dos dados estruturados"""
        if 'data' in dados:
            return dados['data']
        return ""
    
    def _extract_cpf(self, text: str) -> List[str]:
        """Extrai CPFs do texto"""
        return re.findall(r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}', text)

    def _extract_data(self, text: str) -> str:
        """Extrai data do texto bruto"""
        patterns = [
            r'(\d{2})/(\d{2})/(\d{4})',
            r'(\d{1,2})\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\.?\s+(\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3 and match.group(2).isdigit():
                    return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
                elif len(match.groups()) == 3:
                    # Converter mês texto para número
                    meses = {'jan':'01','fev':'02','mar':'03','abr':'04','mai':'05','jun':'06',
                            'jul':'07','ago':'08','set':'09','out':'10','nov':'11','dez':'12'}
                    mes_num = meses.get(match.group(2).lower().replace('.',''), '01')
                    return f"{match.group(1).zfill(2)}/{mes_num}/{match.group(3)}"
        return ""
    
    def _calculate_text_quality(self, extracted_text: str) -> float:
        """Calcula qualidade do texto extraído"""
        if not extracted_text:
            return 0.0
        
        score = 0.0
        # Pontuação por comprimento
        if len(extracted_text) > 100:
            score += 0.3
        
        # Pontuação por palavras-chave
        keywords = ['comprovante', 'pix', 'transferência', 'valor', 'destinatário']
        for keyword in keywords:
            if keyword in extracted_text.lower():
                score += 0.1
        
        # Pontuação por padrões estruturais
        if re.search(r'r\$\s*\d+[.,]\d{2}', extracted_text, re.IGNORECASE):
            score += 0.2
        if re.search(r'\d{2}/\d{2}/\d{4}', extracted_text):
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_expected_bank(self, anotacao: Dict) -> str:
        """Identifica o banco esperado baseado na anotação"""
        institutions = []
        
        if 'remetente' in anotacao and 'instituicao' in anotacao['remetente']:
            institutions.append(anotacao['remetente']['instituicao'].lower())
        
        if 'destinatario' in anotacao and 'instituicao' in anotacao['destinatario']:
            institutions.append(anotacao['destinatario']['instituicao'].lower())
        
        all_institutions = ' '.join(institutions)
        
        if any(term in all_institutions for term in ['nubank', 'nu pagamentos']):
            return 'nubank'
        elif any(term in all_institutions for term in ['inter', 'banco inter']):
            return 'inter'
        elif any(term in all_institutions for term in ['itau', 'unibanco']):
            return 'itau'
        elif any(term in all_institutions for term in ['btg', 'pactual']):
            return 'btg'
        elif 'will' in all_institutions:
            return 'will'
        elif 'picpay' in all_institutions:
            return 'picpay'
        else:
            return 'unknown'

    def show_extracted_text_real(self, max_comprovantes: int = 10) -> None:
        """Mostra o texto REAL extraído dos comprovantes (não fictício)"""
        anotacoes = self.anotacoes['anotacoes'][:max_comprovantes]
        
        print(f"📜 EXIBINDO DADOS REAIS EXTRAÍDOS de {len(anotacoes)} comprovantes:")
        print("=" * 80)
        
        for i, anotacao in enumerate(anotacoes, 1):
            arquivo = anotacao['arquivo_origem']
            image_path = self.imagens_dir / arquivo
            
            print(f"\n📄 {i}. COMPROVANTE: {arquivo}")
            print("-" * 60)
            
            if not image_path.exists():
                print(f"❌ Arquivo não encontrado: {image_path}")
                continue
            
            try:
                # EXTRAÇÃO REAL (não fictícia)
                resultado = self.extractor.process_comprovante(str(image_path))
                
                if resultado['status'] == 'error':
                    print(f"❌ Erro: {resultado.get('erro', 'Erro desconhecido')}")
                    continue
                
                texto = resultado['texto_extraido']['tesseract']
                banco = resultado['banco_identificado']
                dados = resultado['dados_estruturados']
                score = resultado.get('score_extracao', 0)
                
                # Mostrar informações REAIS extraídas
                print(f"🏦 Banco identificado: {banco}")
                print(f"📊 Score de extração: {score:.1%}")
                print(f"📋 Campos extraídos: {list(dados.keys())}")
                print(f"📝 Tamanho do texto: {len(texto)} caracteres")
                
                # Dados REAIS extraídos
                print(f"\n🔍 DADOS REAIS EXTRAÍDOS:")
                if dados:
                    for campo, valor in dados.items():
                        if valor:  # Só mostrar se extraiu algo
                            print(f"   ✅ {campo}: {valor}")
                else:
                    print("   ❌ Nenhum dado estruturado extraído")
                
                # Comparar com dados esperados
                print(f"\n💾 DADOS ESPERADOS (anotação):")
                print(f"   Valor: {anotacao.get('valor', 'N/A')}")
                print(f"   Data: {anotacao.get('data', 'N/A')}")
                print(f"   Destinatário: {anotacao.get('destinatario', {}).get('nome', 'N/A')}")
                print(f"   Banco esperado: {self._identify_expected_bank(anotacao)}")
                
                # Análise de qualidade da extração
                valor_real = dados.get('valor', '')
                valor_esperado = anotacao.get('valor', '')
                valor_match = self._compare_values(valor_real, valor_esperado)
                
                data_real = dados.get('data', '')
                data_esperada = anotacao.get('data', '')
                data_match = data_real == data_esperada
                
                print(f"\n📈 ANÁLISE DE QUALIDADE:")
                print(f"   Valor: {'✅' if valor_match else '❌'} ({valor_real} vs {valor_esperado})")
                print(f"   Data: {'✅' if data_match else '❌'} ({data_real} vs {data_esperada})")
                print(f"   Banco: {'✅' if banco == self._identify_expected_bank(anotacao) else '❌'}")
                
                # Mostrar uma amostra do texto extraído
                print(f"\n📝 TEXTO EXTRAÍDO (primeiros 200 chars):")
                print("─" * 40)
                print(texto[:200] + "..." if len(texto) > 200 else texto)
                print("─" * 40)
                
            except Exception as e:
                print(f"❌ Erro ao processar: {str(e)}")
        
        print("\n" + "=" * 80)
        print("✅ Exibição de dados REAIS concluída!")

    def _compare_values(self, extracted: str, expected: str) -> bool:
        """Compara valores monetários extraídos vs esperados"""
        if not extracted or not expected:
            return False
        
        # Normalizar ambos os valores
        ext_norm = self._normalize_valor(extracted)
        exp_norm = self._normalize_valor(expected)
        
        return ext_norm == exp_norm

    def evaluate_real_extraction(self) -> Dict:
        """Avalia a extração REAL de todos os comprovantes"""
        resultados_reais = []
        anotacoes = self.anotacoes['anotacoes']
        
        print(f"🔍 AVALIANDO EXTRAÇÃO REAL de {len(anotacoes)} comprovantes...")
        
        sucessos = 0
        falhas = 0
        
        for i, anotacao in enumerate(anotacoes, 1):
            arquivo = anotacao['arquivo_origem']
            image_path = self.imagens_dir / arquivo
            
            print(f"📄 Processando {i}/{len(anotacoes)}: {arquivo}")
            
            if not image_path.exists():
                print(f"   ❌ Arquivo não encontrado")
                falhas += 1
                continue
            
            try:
                # Extração REAL
                resultado = self.extractor.process_comprovante(str(image_path))
                
                if resultado['status'] == 'error':
                    print(f"   ❌ Erro: {resultado.get('erro', 'Desconhecido')}")
                    falhas += 1
                    continue
                
                # Calcular métricas reais
                dados_extraidos = resultado['dados_estruturados']
                banco_extraido = resultado['banco_identificado']
                
                # Verificar acertos
                valor_correto = self._compare_values(
                    dados_extraidos.get('valor', ''), 
                    anotacao.get('valor', '')
                )
                
                data_correta = dados_extraidos.get('data', '') == anotacao.get('data', '')
                
                banco_esperado = self._identify_expected_bank(anotacao)
                banco_correto = banco_extraido == banco_esperado
                
                score_real = resultado.get('score_extracao', 0)
                
                resultado_avaliacao = {
                    'arquivo': arquivo,
                    'banco_correto': banco_correto,
                    'valor_correto': valor_correto,
                    'data_correta': data_correta,
                    'score_extracao': score_real,
                    'dados_extraidos': dados_extraidos,
                    'texto_length': len(resultado['texto_extraido']['tesseract'])
                }
                
                resultados_reais.append(resultado_avaliacao)
                
                if score_real > 0.5:
                    sucessos += 1
                    print(f"   ✅ Sucesso (score: {score_real:.1%})")
                else:
                    print(f"   ⚠️ Parcial (score: {score_real:.1%})")
                
            except Exception as e:
                print(f"   ❌ Exceção: {str(e)}")
                falhas += 1
        
        # Relatório final REAL
        relatorio_real = {
            'total_processados': len(resultados_reais),
            'sucessos': sucessos,
            'falhas': falhas,
            'taxa_sucesso': sucessos / len(anotacoes) if anotacoes else 0,
            'score_medio': sum(r['score_extracao'] for r in resultados_reais) / len(resultados_reais) if resultados_reais else 0,
            'bancos_corretos': sum(1 for r in resultados_reais if r['banco_correto']),
            'valores_corretos': sum(1 for r in resultados_reais if r['valor_correto']),
            'datas_corretas': sum(1 for r in resultados_reais if r['data_correta']),
            'resultados_detalhados': resultados_reais
        }
        
        print(f"\n📊 RELATÓRIO FINAL - DADOS REAIS:")
        print(f"   Total processados: {relatorio_real['total_processados']}")
        print(f"   Sucessos: {sucessos} ({relatorio_real['taxa_sucesso']:.1%})")
        print(f"   Score médio: {relatorio_real['score_medio']:.1%}")
        print(f"   Bancos corretos: {relatorio_real['bancos_corretos']}")
        print(f"   Valores corretos: {relatorio_real['valores_corretos']}")
        print(f"   Datas corretas: {relatorio_real['datas_corretas']}")
        
        return relatorio_real

    def show_extracted_text(self, max_comprovantes: int = 10) -> None:
        """Método compatível para exibir texto extraído"""
        self.show_extracted_text_real(max_comprovantes)
    
    def evaluate_single_image(self, anotacao: Dict) -> ExtractionMetrics:
        """Avalia uma única imagem"""
        arquivo = anotacao['arquivo_origem']
        image_path = self.imagens_dir / arquivo
        
        if not image_path.exists():
            return self._create_error_metrics(arquivo, "Arquivo não encontrado")
        
        try:
            resultado = self.extractor.process_comprovante(str(image_path))
            return self._evaluate_extraction_results(anotacao, resultado)
        except Exception as e:
            return self._create_error_metrics(arquivo, str(e))

    def _create_error_metrics(self, arquivo: str, error_msg: str) -> ExtractionMetrics:
        """Cria métricas de erro"""
        return ExtractionMetrics(
            arquivo=arquivo,
            banco_correto=False,
            valor_correto=False,
            data_correta=False,
            cpf_destinatario_correto=False,
            cpf_remetente_correto=False,
            nome_destinatario_similar=0.0,
            nome_remetente_similar=0.0,
            id_transacao_correto=False,
            texto_extraido_qualidade=0.0,
            score_total=0.0,
            dados_extraidos_reais={'erro': error_msg}
        )

    def _evaluate_extraction_results(self, anotacao: Dict, resultado: Dict) -> ExtractionMetrics:
        """Avalia os resultados da extração"""
        dados = resultado.get('dados_estruturados', {})
        
        # Avaliar cada campo
        banco_correto = resultado.get('banco_identificado') == self._identify_expected_bank(anotacao)
        valor_correto = self._compare_values(dados.get('valor', ''), anotacao.get('valor', ''))
        data_correta = dados.get('data', '') == anotacao.get('data', '')
        
        # Similaridade de nomes
        nome_dest_esperado = anotacao.get('destinatario', {}).get('nome', '')
        nome_dest_extraido = dados.get('destinatario', '')
        nome_dest_similar = self._similarity_score(nome_dest_extraido, nome_dest_esperado)
        
        # Qualidade do texto
        texto_qualidade = self._calculate_text_quality(resultado.get('texto_extraido', {}).get('tesseract', ''))
        
        # Score total
        score_total = resultado.get('score_extracao', 0)
        
        return ExtractionMetrics(
            arquivo=anotacao['arquivo_origem'],
            banco_correto=banco_correto,
            valor_correto=valor_correto,
            data_correta=data_correta,
            cpf_destinatario_correto=False,  # TODO: implementar
            cpf_remetente_correto=False,     # TODO: implementar
            nome_destinatario_similar=nome_dest_similar,
            nome_remetente_similar=0.0,      # TODO: implementar
            id_transacao_correto=False,      # TODO: implementar
            texto_extraido_qualidade=texto_qualidade,
            score_total=score_total,
            dados_extraidos_reais=dados
        )

    def evaluate_all(self) -> List[ExtractionMetrics]:
        """Avalia todos os comprovantes"""
        resultados = []
        for anotacao in self.anotacoes['anotacoes']:
            metrics = self.evaluate_single_image(anotacao)
            resultados.append(metrics)
        return resultados

    def generate_report(self, resultados: List[ExtractionMetrics]) -> Dict:
        """Gera relatório das métricas"""
        if not resultados:
            return {'erro': 'Nenhum resultado para gerar relatório'}
        
        total = len(resultados)
        bancos_corretos = sum(1 for r in resultados if r.banco_correto)
        valores_corretos = sum(1 for r in resultados if r.valor_correto)
        datas_corretas = sum(1 for r in resultados if r.data_correta)
        
        return {
            'total_avaliados': total,
            'bancos_corretos': bancos_corretos,
            'valores_corretos': valores_corretos,
            'datas_corretas': datas_corretas,
            'taxa_banco': bancos_corretos / total,
            'taxa_valor': valores_corretos / total,
            'taxa_data': datas_corretas / total,
            'score_medio': sum(r.score_total for r in resultados) / total
        }
