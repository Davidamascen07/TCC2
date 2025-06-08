from app import create_app
from src.ml.trainer import PIXMLTrainer
import os

def setup_environment():
    """Configura o ambiente inicial"""
    # Criar diretórios necessários
    os.makedirs('data/uploads', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # Treinar modelo inicial se as anotações existirem
    annotations_path = 'data/raw/exemplos/anotacao.json'
    if os.path.exists(annotations_path):
        print("Treinando modelo inicial...")
        trainer = PIXMLTrainer()
        try:
            trainer.train_bank_classifier(annotations_path)
            print("✅ Modelo treinado com sucesso!")
        except Exception as e:
            print(f"⚠️ Erro ao treinar modelo: {e}")
    else:
        print("⚠️ Arquivo de anotações não encontrado")

if __name__ == '__main__':
    print("🚀 Iniciando Sistema Extrator de Comprovantes PIX")
    
    setup_environment()
    
    app = create_app()
    
    print("\n📱 Sistema pronto! Acesse:")
    print("🌐 Interface Web: http://localhost:5000")
    print("🤖 API: http://localhost:5000/api")
    print("💬 Chatbot: http://localhost:5000/chatbot")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
