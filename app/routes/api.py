from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import logging

from src.ocr.extractor import ComprovantePIXExtractor

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Configurações
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/extract', methods=['POST'])
def extract_data():
    """API endpoint para extrair dados de comprovantes PIX"""
    try:
        # Verificar se foi enviado um arquivo
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'erro': 'Nenhum arquivo foi enviado'
            }), 400
        
        file = request.files['file']
        
        # Verificar se arquivo foi selecionado
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'erro': 'Nenhum arquivo selecionado'
            }), 400
        
        # Verificar extensão do arquivo
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'erro': f'Tipo de arquivo não permitido. Use: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Verificar tamanho do arquivo
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'status': 'error',
                'erro': f'Arquivo muito grande. Máximo: {MAX_FILE_SIZE // (1024*1024)}MB'
            }), 400
        
        # Salvar arquivo temporariamente
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{filename.rsplit(".", 1)[1].lower()}') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        logger.info(f"📁 Arquivo salvo temporariamente: {temp_path}")
        
        try:
            # Processar com o extrator REAL
            extractor = ComprovantePIXExtractor()
            resultado = extractor.process_comprovante(temp_path)
            
            # Verificar se a extração foi bem-sucedida
            if resultado.get('status') == 'error':
                return jsonify({
                    'status': 'error',
                    'erro': resultado.get('erro', 'Erro desconhecido na extração')
                }), 500
            
            # Adicionar informações extras para a resposta
            response_data = {
                'status': resultado.get('status', 'success'),
                'arquivo_origem': filename,
                'banco_identificado': resultado.get('banco_identificado', 'unknown'),
                'dados_estruturados': resultado.get('dados_estruturados', {}),
                'nomes': resultado.get('nomes', {}),
                'score_extracao': resultado.get('score_extracao', 0),
                'texto_extraido': {
                    'tesseract': resultado.get('texto_extraido', {}).get('tesseract', ''),
                    'caracteres_extraidos': len(resultado.get('texto_extraido', {}).get('tesseract', ''))
                },
                'metodos_utilizados': resultado.get('metodos_utilizados', {}),
                'timestamp': resultado.get('timestamp'),
                'processing_info': {
                    'file_size': file_size,
                    'file_type': filename.rsplit('.', 1)[1].lower(),
                    'extraction_method': 'OCR + ML + Template Matching'
                }
            }
            
            logger.info(f"✅ Extração concluída com sucesso: {filename}")
            logger.info(f"📊 Score: {response_data['score_extracao']:.1%}")
            logger.info(f"🏦 Banco: {response_data['banco_identificado']}")
            
            return jsonify(response_data), 200
            
        except Exception as extraction_error:
            logger.error(f"❌ Erro durante extração: {str(extraction_error)}")
            return jsonify({
                'status': 'error',
                'erro': f'Erro no processamento: {str(extraction_error)}',
                'arquivo_origem': filename
            }), 500
            
        finally:
            # Limpar arquivo temporário
            try:
                os.unlink(temp_path)
                logger.info(f"🗑️ Arquivo temporário removido: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Erro ao remover arquivo temporário: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"❌ Erro geral na API: {str(e)}")
        return jsonify({
            'status': 'error',
            'erro': f'Erro interno do servidor: {str(e)}'
        }), 500

@api_bp.route('/status', methods=['GET'])
def api_status():
    """Endpoint para verificar status da API"""
    try:
        # Testar se o extrator pode ser inicializado
        extractor = ComprovantePIXExtractor()
        
        return jsonify({
            'status': 'online',
            'message': 'API funcionando corretamente',
            'services': {
                'ocr': 'available',
                'ml_classifier': 'available' if extractor.ml_classifier else 'not_loaded',
                'templates': f'{len(extractor.templates)} bancos'
            },
            'version': '2.0.0',
            'timestamp': str(datetime.now())
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro no sistema: {str(e)}',
            'timestamp': str(datetime.now())
        }), 500

@api_bp.route('/banks', methods=['GET'])
def get_supported_banks():
    """Retorna lista de bancos suportados"""
    try:
        extractor = ComprovantePIXExtractor()
        banks = list(extractor.bank_patterns.keys())
        
        return jsonify({
            'status': 'success',
            'bancos_suportados': banks,
            'total': len(banks),
            'timestamp': str(datetime.now())
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'erro': str(e)
        }), 500

# Adicionar import necessário
from datetime import datetime
