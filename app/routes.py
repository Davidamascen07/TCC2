from flask import Blueprint, request, jsonify, current_app, render_template
import os
from werkzeug.utils import secure_filename
import json
from pathlib import Path

from src.ocr.extractor import ComprovantePIXExtractor
from src.ml.trainer import PIXMLTrainer
from src.ml.yolo_detector import PIXYOLODetector

api_bp = Blueprint('api', __name__)

# Inicializar serviços
extractor = ComprovantePIXExtractor()
ml_trainer = PIXMLTrainer()
yolo_detector = PIXYOLODetector()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@api_bp.route('/upload')
def upload_page():
    """Página de upload"""
    return render_template('upload.html')

@api_bp.route('/chatbot')
def chatbot():
    """Página do chatbot"""
    return render_template('chatbot.html')

@api_bp.route('/analytics')
def analytics():
    """Página de analytics"""
    return render_template('analytics.html')

@api_bp.route('/upload', methods=['POST'])
def upload_comprovante():
    """Endpoint para upload e processamento de comprovante"""
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Processar comprovante
            result = extractor.process_comprovante(filepath)
            
            # Melhorar com ML se modelo estiver disponível
            if ml_trainer.load_models():
                combined_text = result.get('texto_extraido', {}).get('tesseract', '')
                ml_prediction = ml_trainer.predict_bank(combined_text)
                result['banco_ml_prediction'] = ml_prediction
            
            # Usar YOLO para detecção de regiões (opcional)
            try:
                yolo_regions = yolo_detector.extract_roi_text(filepath, extractor)
                result['yolo_regions'] = yolo_regions
            except:
                pass  # YOLO é opcional
            
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500
        finally:
            # Limpar arquivo temporário
            try:
                os.remove(filepath)
            except:
                pass
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@api_bp.route('/train', methods=['POST'])
def train_model():
    """Endpoint para treinar o modelo ML"""
    try:
        annotations_path = 'data/raw/exemplos/anotacao.json'
        if not Path(annotations_path).exists():
            return jsonify({'error': 'Arquivo de anotações não encontrado'}), 404
        
        ml_trainer.train_bank_classifier(annotations_path)
        return jsonify({'message': 'Modelo treinado com sucesso'}), 200
        
    except Exception as e:
        return jsonify({'error': f'Erro ao treinar modelo: {str(e)}'}), 500

@api_bp.route('/batch_process', methods=['POST'])
def batch_process():
    """Processa todos os comprovantes da pasta exemplos"""
    try:
        images_dir = Path('data/raw/exemplos/imagens')
        if not images_dir.exists():
            return jsonify({'error': 'Diretório de imagens não encontrado'}), 404
        
        results = []
        for image_file in images_dir.glob('*.jpg'):
            result = extractor.process_comprovante(str(image_file))
            results.append(result)
        
        # Salvar resultados
        output_path = 'data/processed/batch_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'message': f'Processados {len(results)} comprovantes',
            'results': results,
            'output_file': output_path
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Erro no processamento em lote: {str(e)}'}), 500

@api_bp.route('/chatbot', methods=['POST'])
def chatbot_analyze():
    """Endpoint do chatbot para análise de comprovante"""
    data = request.get_json()
    
    if 'image_url' in data:
        # Processar imagem via URL
        return jsonify({'error': 'Processamento via URL não implementado ainda'}), 501
    
    elif 'message' in data:
        message = data['message'].lower()
        
        if 'ajuda' in message or 'help' in message:
            return jsonify({
                'response': """
Olá! Sou o assistente de análise de comprovantes PIX. 
Posso ajudar você a extrair dados de comprovantes.

Comandos disponíveis:
- Envie uma imagem de comprovante PIX
- 'processar lote' - processa todas as imagens da pasta exemplos
- 'treinar' - treina o modelo de machine learning
- 'status' - verifica status do sistema
                """
            }), 200
        
        elif 'processar lote' in message:
            return batch_process()
        
        elif 'treinar' in message:
            return train_model()
        
        elif 'status' in message:
            return jsonify({
                'response': 'Sistema funcionando normalmente. Pronto para processar comprovantes PIX.',
                'models_loaded': ml_trainer.load_models()
            }), 200
        
        else:
            return jsonify({
                'response': 'Não entendi sua mensagem. Digite "ajuda" para ver os comandos disponíveis.'
            }), 200
    
    return jsonify({'error': 'Dados inválidos'}), 400