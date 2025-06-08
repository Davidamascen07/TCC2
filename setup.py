import os
import sys
import subprocess
import logging
from pathlib import Path
from setuptools import setup, find_packages

def setup_logging():
    """Configura sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def print_banner():
    """Exibe banner de início"""
    print("=" * 70)
    print("🚀 SETUP DO SISTEMA EXTRATOR DE COMPROVANTES PIX")
    print("   Sistema completo com OCR + ML + YOLO + Flask")
    print("=" * 70)

def check_python_version():
    """Verifica versão do Python"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário!")
        print(f"   Versão atual: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detectado")
    return True

def setup_virtual_environment():
    """Verifica e configura ambiente virtual"""
    try:
        # Verifica se já está em um ambiente virtual
        in_venv = (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        if in_venv:
            print("✅ Ambiente virtual detectado")
            return True
        else:
            print("⚠️  ATENÇÃO: Execute este script em um ambiente virtual!")
            print("\n📋 Para criar e ativar um ambiente virtual:")
            print("   python -m venv venv")
            print("   venv\\Scripts\\activate     # Windows")
            print("   source venv/bin/activate  # Linux/Mac")
            print("\n   Depois execute novamente: python setup.py")
            return False
    except Exception as e:
        logger.error(f"Erro na verificação do ambiente: {e}")
        return False

def create_directories():
    """Cria estrutura de diretórios necessária"""
    print("\n📁 Criando estrutura de diretórios...")
    
    directories = [
        'data/raw/exemplos/imagens',
        'data/processed',
        'uploads',
        'models',
        'logs',
        'src/core',
        'src/ml',
        'src/web/templates',
        'src/web/static/css',
        'src/web/static/js',
        'src/web/static/images',
        'tests',
        'docs',
        'temp'
    ]
    
    created_count = 0
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {directory}")
            created_count += 1
        except Exception as e:
            print(f"  ❌ Erro ao criar {directory}: {e}")
    
    print(f"\n📊 {created_count}/{len(directories)} diretórios criados com sucesso")
    return created_count == len(directories)

def install_requirements():
    """Instala dependências Python com fallbacks"""
    print("\n📦 Instalando dependências Python...")
    
    if not os.path.exists('requirements.txt'):
        print("  ❌ Arquivo requirements.txt não encontrado!")
        return False
    
    try:
        # Atualiza pip, setuptools e wheel primeiro
        print("  🔄 Atualizando ferramentas básicas...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 
            'pip', 'setuptools', 'wheel'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Tenta instalação normal primeiro
        print("  🔄 Instalando dependências...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], timeout=600)  # 10 minutos timeout
            print("  ✅ Dependências instaladas com sucesso")
            return True
            
        except subprocess.CalledProcessError:
            print("  ⚠️  Falha na instalação normal. Tentando instalação individual...")
            return install_packages_individually()
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Erro na atualização das ferramentas: {e}")
        print("     Tentando instalação individual...")
        return install_packages_individually()

def install_packages_individually():
    """Instala pacotes individualmente para identificar problemas"""
    print("  🔄 Instalação individual de pacotes...")
    
    # Pacotes essenciais primeiro
    essential_packages = [
        'numpy>=1.24.0',
        'Pillow>=10.0.0',
        'opencv-python>=4.8.0',
        'Flask>=3.0.0',
        'pytesseract>=0.3.10',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0'
    ]
    
    # Pacotes opcionais
    optional_packages = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'ultralytics>=8.0.0',
        'easyocr>=1.7.0'
    ]
    
    success_count = 0
    
    # Instala pacotes essenciais
    for package in essential_packages:
        if install_single_package(package, essential=True):
            success_count += 1
    
    # Instala pacotes opcionais
    for package in optional_packages:
        install_single_package(package, essential=False)
    
    # Instala dependências restantes
    remaining_packages = [
        'flask-cors', 'regex', 'fuzzywuzzy', 'python-levenshtein',
        'requests', 'werkzeug', 'python-dotenv', 'jinja2', 'joblib'
    ]
    
    for package in remaining_packages:
        install_single_package(package, essential=False)
    
    print(f"  📊 {success_count}/{len(essential_packages)} pacotes essenciais instalados")
    return success_count >= len(essential_packages) - 1

def install_single_package(package, essential=True):
    """Instala um pacote individual"""
    try:
        print(f"    🔄 Instalando {package}...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', package
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        print(f"    ✅ {package}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if essential:
            print(f"    ❌ {package} (essencial)")
        else:
            print(f"    ⚠️  {package} (opcional)")
        return False

def check_system_dependencies():
    """Verifica dependências do sistema"""
    print("\n🔍 Verificando dependências do sistema...")
    
    # Verifica Tesseract
    tesseract_ok = check_tesseract()
    
    # Verifica Microsoft Visual C++
    vc_ok = check_visual_cpp()
    
    return tesseract_ok

def check_tesseract():
    """Verifica instalação do Tesseract"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"  ✅ Tesseract OCR encontrado: {version}")
            return True
        else:
            print("  ⚠️  Tesseract não encontrado")
            print_tesseract_install_instructions()
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ⚠️  Tesseract OCR não encontrado")
        print_tesseract_install_instructions()
        return False

def check_visual_cpp():
    """Verifica Microsoft Visual C++ (Windows)"""
    if sys.platform == "win32":
        try:
            import winreg
            # Verifica registros do Visual C++
            key_paths = [
                r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
            ]
            
            for path in key_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path):
                        print("  ✅ Microsoft Visual C++ Redistributable encontrado")
                        return True
                except WindowsError:
                    continue
            
            print("  ⚠️  Microsoft Visual C++ Redistributable pode estar faltando")
            print("     Baixe de: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return False
            
        except ImportError:
            return True  # Não consegue verificar, assume OK
    
    return True  # Linux/Mac não precisa

def print_tesseract_install_instructions():
    """Exibe instruções de instalação do Tesseract"""
    print("     📋 INSTRUÇÕES DE INSTALAÇÃO DO TESSERACT:")
    print("     Windows:")
    print("       1. Baixe: https://github.com/UB-Mannheim/tesseract/wiki")
    print("       2. Instale o executável")
    print("       3. Adicione ao PATH: C:\\Program Files\\Tesseract-OCR")
    print("     Ubuntu/Debian:")
    print("       sudo apt update")
    print("       sudo apt install tesseract-ocr tesseract-ocr-por")
    print("     macOS:")
    print("       brew install tesseract tesseract-lang")

def run_basic_tests():
    """Executa testes básicos de importação"""
    print("\n🧪 Executando testes básicos...")
    
    tests = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('flask', 'Flask'),
        ('pytesseract', 'Tesseract Python'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas')
    ]
    
    success_count = 0
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {name} - Erro: {str(e)[:50]}...")
    
    # Testes opcionais
    optional_tests = [
        ('easyocr', 'EasyOCR'),
        ('ultralytics', 'YOLO'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision')
    ]
    
    optional_success = 0
    for module, name in optional_tests:
        try:
            __import__(module)
            print(f"  ✅ {name} (opcional)")
            optional_success += 1
        except ImportError:
            print(f"  ⚠️  {name} (opcional) - não disponível")
    
    print(f"\n📊 {success_count}/{len(tests)} dependências principais OK")
    print(f"📊 {optional_success}/{len(optional_tests)} dependências opcionais OK")
    
    return success_count >= len(tests) - 2  # Permite 2 falhas

def create_installation_report():
    """Cria relatório de instalação"""
    print("\n📋 Criando relatório de instalação...")
    
    try:
        import json
        from datetime import datetime
        
        report = {
            "installation_date": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "virtual_env": bool(getattr(sys, 'base_prefix', sys.prefix) != sys.prefix),
            "installed_packages": [],
            "failed_packages": [],
            "system_dependencies": {
                "tesseract": check_tesseract_simple(),
                "visual_cpp": check_visual_cpp() if sys.platform == "win32" else "N/A"
            }
        }
        
        # Testa pacotes importantes
        test_packages = [
            'numpy', 'opencv-python', 'flask', 'pytesseract', 
            'scikit-learn', 'pandas', 'torch', 'ultralytics', 'easyocr'
        ]
        
        for package in test_packages:
            try:
                __import__(package.replace('-', '_'))
                report["installed_packages"].append(package)
            except ImportError:
                report["failed_packages"].append(package)
        
        with open('installation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("  ✅ Relatório salvo em: installation_report.json")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro ao criar relatório: {e}")
        return False

def check_tesseract_simple():
    """Verificação simples do Tesseract"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def download_yolo_model():
    """Baixa modelo YOLO se necessário"""
    print("\n🤖 Configurando modelo YOLO...")
    
    try:
        from ultralytics import YOLO
        
        # Tenta carregar o modelo (baixa automaticamente se não existir)
        print("  🔄 Baixando YOLOv8...")
        model = YOLO('yolov8n.pt')
        
        # Salva no diretório de modelos
        model_path = Path('models/yolov8n.pt')
        if not model_path.exists():
            # Move o modelo para o diretório correto
            import shutil
            yolo_cache = Path.home() / '.cache' / 'ultralytics'
            if (yolo_cache / 'yolov8n.pt').exists():
                shutil.copy2(yolo_cache / 'yolov8n.pt', model_path)
        
        print("  ✅ Modelo YOLO configurado")
        return True
        
    except Exception as e:
        print(f"  ⚠️  Aviso: Erro ao configurar YOLO: {e}")
        print("     O modelo será baixado automaticamente no primeiro uso")
        return False

def create_env_file():
    """Cria arquivo de configuração .env"""
    print("\n⚙️  Criando arquivo de configuração...")
    
    env_content = """# Configurações do Sistema Extrator de Comprovantes Pix
# Flask Configurações
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Upload Configurações
UPLOAD_FOLDER=uploads
MAX_FILE_SIZE=16777216
ALLOWED_EXTENSIONS=png,jpg,jpeg,gif,bmp,tiff

# OCR Configurações
TESSERACT_CMD=tesseract
OCR_LANGUAGE=por
OCR_CONFIG=--oem 3 --psm 6

# ML Configurações
MODEL_PATH=models/
CONFIDENCE_THRESHOLD=0.7
ML_ENABLED=True

# YOLO Configurações
YOLO_MODEL=yolov8n.pt
YOLO_CONFIDENCE=0.5
YOLO_ENABLED=True

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Processamento
BATCH_SIZE=10
PARALLEL_PROCESSING=True
TEMP_DIR=temp

# Cache
CACHE_ENABLED=True
CACHE_TTL=3600
"""
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("  ✅ Arquivo .env criado")
        return True
    except Exception as e:
        print(f"  ❌ Erro ao criar .env: {e}")
        return False

def create_gitignore():
    """Cria arquivo .gitignore"""
    print("\n📝 Criando arquivo .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
.spyderproject
.spyproject

# Sistema
.DS_Store
Thumbs.db
desktop.ini

# Logs
logs/
*.log

# Uploads e arquivos temporários
uploads/
temp/
temp_*
*.tmp

# Modelos treinados
models/*.pkl
models/*.pth
models/*.h5
models/*.model

# Dados sensíveis
.env
.env.local
config.local.py
secrets.py

# Cache
.cache/
.pytest_cache/
__pycache__/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Dados de teste grandes
data/raw/exemplos/imagens/*.jpg
data/raw/exemplos/imagens/*.png
data/raw/exemplos/imagens/*.jpeg

# Backup
*.bak
*.backup

# Coverage
.coverage
htmlcov/

# Profiling
*.prof
"""
    
    try:
        with open('.gitignore', 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("  ✅ Arquivo .gitignore criado")
        return True
    except Exception as e:
        print(f"  ❌ Erro ao criar .gitignore: {e}")
        return False

def run_basic_tests():
    """Executa testes básicos de importação"""
    print("\n🧪 Executando testes básicos...")
    
    tests = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('pytesseract', 'Tesseract Python'),
        ('flask', 'Flask'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas')
    ]
    
    success_count = 0
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            success_count += 1
        except ImportError:
            print(f"  ❌ {name} - não importado")
    
    # Testes opcionais
    optional_tests = [
        ('easyocr', 'YOLO'),
        ('ultralytics', 'YOLO'),
        ('torch', 'PyTorch')
    ]
    
    for module, name in optional_tests:
        try:
            __import__(module)
            print(f"  ✅ {name} (opcional)")
        except ImportError:
            print(f"  ⚠️  {name} (opcional) - não disponível")
    
    print(f"\n📊 {success_count}/{len(tests)} dependências principais OK")
    return success_count >= len(tests) - 1  # Permite 1 falha

def create_sample_files():
    """Cria arquivos de exemplo"""
    print("\n📋 Criando arquivos de exemplo...")
    
    try:
        # Cria arquivo de exemplo para teste
        sample_config = {
            "name": "Sistema Extrator de Comprovantes Pix",
            "version": "1.0.0",
            "status": "Configurado com sucesso",
            "components": {
                "ocr": "Tesseract + EasyOCR",
                "ml": "Scikit-learn",
                "detection": "YOLOv8",
                "web": "Flask"
            }
        }
        
        import json
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        
        print("  ✅ Arquivo config.json criado")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro ao criar arquivos: {e}")
        return False

def create_run_script():
    """Cria script de execução"""
    print("\n🚀 Criando script de execução...")
    
    # Script para Windows
    run_bat = """@echo off
echo Iniciando Sistema Extrator de Comprovantes Pix...
echo.

if not exist venv (
    echo Ambiente virtual nao encontrado!
    echo Execute: python -m venv venv
    pause
    exit /b 1
)

call venv\\Scripts\\activate
python src\\web\\app.py
pause
"""
    
    # Script para Linux/Mac
    run_sh = """#!/bin/bash
echo "Iniciando Sistema Extrator de Comprovantes Pix..."
echo

if [ ! -d "venv" ]; then
    echo "Ambiente virtual não encontrado!"
    echo "Execute: python -m venv venv"
    exit 1
fi

source venv/bin/activate
python src/web/app.py
"""
    
    try:
        with open('run.bat', 'w', encoding='utf-8') as f:
            f.write(run_bat)
        
        with open('run.sh', 'w', encoding='utf-8') as f:
            f.write(run_sh)
        
        # Torna o script executável no Linux/Mac
        import stat
        os.chmod('run.sh', stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print("  ✅ Scripts de execução criados (run.bat, run.sh)")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro ao criar scripts: {e}")
        return False

def display_final_instructions():
    """Exibe instruções finais"""
    print("\n" + "=" * 70)
    print("🎉 SETUP CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. 📁 Coloque imagens de comprovantes em: data/raw/exemplos/imagens/")
    print("2. 🚀 Execute o sistema:")
    print("   Windows: run.bat")
    print("   Linux/Mac: ./run.sh")
    print("   Ou manualmente: python src/web/app.py")
    print("\n3. 🌐 Acesse: http://localhost:5000")
    print("\n📚 FUNCIONALIDADES:")
    print("• 📤 Upload de comprovantes via interface web")
    print("• 🤖 Chatbot para análise interativa")
    print("• 🔍 OCR avançado (Tesseract + EasyOCR)")
    print("• 🎯 Detecção inteligente com YOLO")
    print("• 🧠 Machine Learning para melhor precisão")
    print("• 📊 Processamento em lote")
    print("\n⚙️  CONFIGURAÇÃO:")
    print("• Edite .env para personalizar configurações")
    print("• Logs em: logs/app.log")
    print("• Modelos em: models/")
    print("\n🆘 SUPORTE:")
    print("• Verifique logs em caso de erro")
    print("• Certifique-se que Tesseract está instalado")
    print("• Para dúvidas, consulte a documentação")
    print("\n" + "=" * 70)

def main():
    """Função principal do setup"""
    print_banner()
    
    # Lista de verificações atualizadas
    checks = [
        ("Verificando Python", check_python_version),
        ("Verificando ambiente virtual", setup_virtual_environment),
        ("Criando diretórios", create_directories),
        ("Verificando dependências do sistema", check_system_dependencies),
        ("Instalando dependências Python", install_requirements),
        ("Configurando YOLO", download_yolo_model),
        ("Criando arquivo .env", create_env_file),
        ("Criando .gitignore", create_gitignore),
        ("Executando testes básicos", run_basic_tests),
        ("Criando arquivos de exemplo", create_sample_files),
        ("Criando scripts de execução", create_run_script),
        ("Gerando relatório de instalação", create_installation_report)
    ]
    
    success_count = 0
    total_checks = len(checks)
    
    for i, (description, check_func) in enumerate(checks, 1):
        print(f"\n[{i}/{total_checks}] {description}...")
        try:
            if check_func():
                success_count += 1
        except Exception as e:
            logger.error(f"Erro em '{description}': {e}")
    
    print(f"\n📊 RESULTADO: {success_count}/{total_checks} verificações concluídas")
    
    if success_count >= total_checks - 3:  # Permite 3 falhas
        display_final_instructions()
        print("\n💡 DICA: Verifique installation_report.json para detalhes da instalação")
        return True
    else:
        print("\n❌ Setup incompleto. Verifique os erros acima e:")
        print("   1. Instale o Tesseract OCR se necessário")
        print("   2. Execute: pip install --upgrade pip setuptools wheel")
        print("   3. Tente: pip install -r requirements.txt")
        print("   4. Verifique installation_report.json para detalhes")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrompido pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro fatal no setup: {e}")
        sys.exit(1)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="extrator-comprovantes-pix",
    version="1.0.0",
    author="David Damasceno",
    author_email="davidamascen07@gmail.com",
    description="Sistema inteligente de extração de dados de comprovantes PIX usando OCR e ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Davidamascen07/TCC2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "gpu": [
            "torch[cuda]>=2.2.0",
            "torchvision[cuda]>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "extrator-pix=main:main",
        ],
    },
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
)
