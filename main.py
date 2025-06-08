import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent))

from app import create_app
from src.ml.trainer import PIXMLTrainer

def initialize_system():
    """Inicializa o sistema com treinamento de modelos"""
    print("🚀 Iniciando Sistema Extrator de Comprovantes PIX")
    
    # Criar diretórios necessários
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/uploads', exist_ok=True)
    os.makedirs('data/extraction_logs', exist_ok=True)
    
    # Treinar modelo inicial se não existir
    try:
        print("Treinando modelo inicial...")
        trainer = PIXMLTrainer()
        
        # Verificar se modelos já existem
        if not trainer.load_models():
            # Usar o método correto
            trainer.train_bank_classifier()  # Método que existe na classe
        else:
            print("✅ Modelos ML já existem e foram carregados")
            
    except Exception as e:
        print(f"⚠️ Erro ao treinar modelo: {e}")
    
    print("\n📱 Sistema pronto! Acesse:")
    print("🌐 Interface Web: http://localhost:5000")
    print("🤖 API: http://localhost:5000/api")
    print("💬 Chatbot: http://localhost:5000/chatbot")

if __name__ == '__main__':
    initialize_system()
    
    # Criar e executar a aplicação Flask
    app = create_app()
    
    # Executar em modo debug
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
