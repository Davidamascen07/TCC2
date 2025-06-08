# 🏦 Extrator Inteligente de Comprovantes PIX

Sistema avançado de extração de dados de comprovantes PIX usando **OCR**, **Machine Learning** e **YOLO** para identificação precisa de informações bancárias.

## 🚀 Funcionalidades

- 🔍 **OCR Avançado**: Utiliza Tesseract e EasyOCR para extração de texto
- 🧠 **Machine Learning**: Classificação automática de diferentes bancos
- 🎯 **YOLO Detection**: Detecção de regiões específicas nos comprovantes
- 🌐 **Interface Web**: Sistema web completo com Flask
- 🤖 **Chatbot**: Assistente inteligente para análise de comprovantes
- 📊 **Multi-Bancos**: Suporte para diversos bancos (Nubank, Inter, Itaú, BTG, etc.)
- ⚡ **Processamento em Lote**: Análise de múltiplos comprovantes simultaneamente

## 🏗️ Arquitetura do Sistema

```
extrator-comprovantes-ocr-1/
├── app/                    # Aplicação Flask
│   ├── routes/            # Rotas da API e web
│   ├── templates/         # Templates HTML
│   └── __init__.py
├── src/                   # Código fonte principal
│   ├── ocr/              # Módulos de OCR
│   ├── ml/               # Machine Learning
│   └── utils/            # Utilitários
├── data/                 # Dados do projeto
│   ├── raw/              # Dados brutos
│   ├── processed/        # Dados processados
│   └── models/           # Modelos treinados
└── requirements.txt      # Dependências
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Flask** - Framework web
- **OpenCV** - Processamento de imagens
- **Tesseract/EasyOCR** - Reconhecimento óptico de caracteres
- **scikit-learn** - Machine Learning
- **YOLO (Ultralytics)** - Detecção de objetos
- **PyTorch** - Deep Learning
- **NumPy/Pandas** - Manipulação de dados

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Tesseract OCR instalado no sistema
- Git para controle de versão

### Instalação do Tesseract

**Windows:**
```bash
# Baixar e instalar do site oficial
# https://github.com/UB-Mannheim/tesseract/wiki
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-por
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

## 🚀 Instalação e Configuração

### 1. Clone o repositório
```bash
git clone https://github.com/Davidamascen07/TCC2.git
cd TCC2
```

### 2. Crie um ambiente virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente
```bash
# Crie um arquivo .env na raiz do projeto
cp .env.example .env
```

### 5. Execute o sistema
```bash
python main.py
```

## 📊 Uso do Sistema

### Interface Web
Acesse `http://localhost:5000` para usar a interface web completa.

### API REST
```bash
# Upload de comprovante
POST /api/upload
Content-Type: multipart/form-data

# Chatbot
POST /api/chatbot
{
  "message": "Sua mensagem aqui"
}

# Processamento em lote
POST /api/batch_process

# Treinamento do modelo
POST /api/train
```

### Chatbot
```bash
# Comandos disponíveis:
- "ajuda" - Lista todos os comandos
- "processar lote" - Processa todos os comprovantes
- "treinar" - Treina o modelo ML
- "status" - Verifica status do sistema
```

## 🏦 Bancos Suportados

| Banco | Status | Recursos Específicos |
|-------|--------|---------------------|
| 🟣 Nubank | ✅ | Agência, Conta, Chave PIX |
| 🟠 Inter | ✅ | ID Transação específico |
| 🔵 Itaú | ✅ | Código de autenticação |
| 🟡 BTG Pactual | ✅ | Hash de autenticação |
| 🟢 Will Bank | ✅ | Descrição de pagamento |
| 💜 PicPay | ✅ | Conta de pagamentos |
| 🔴 Caixa | ✅ | NSU, Chave de segurança |
| 🟨 Banco do Brasil | ✅ | Documento, Autenticação |

## 📈 Dados de Exemplo

O sistema inclui 104 comprovantes de exemplo com anotações manuais para treinamento:

- **Período**: 16/03/2025 a 04/06/2025
- **Valor Total**: R$ 4.705,10
- **Formatos**: JPG com diferentes layouts e bancos

## 🔧 Desenvolvimento

### Estrutura de Dados
```json
{
  "arquivo_origem": "comprovante_exemplo.jpg",
  "valor": "R$ 58,00",
  "destinatario": {
    "nome": "Ana Cleuma Sousa dos Santos",
    "cpf": "***120.983-**",
    "instituicao": "Nu Pagamentos - Ip"
  },
  "remetente": {
    "nome": "AIALA DE BRITO MARTINS",
    "cpf": "***975.343-**",
    "instituicao": "Banco Inter S.A."
  },
  "data": "03/06/2025",
  "horario": "19h51",
  "id_transacao": "E00416968202506032251NJ14663h0eR",
  "tipo": "Pix enviado"
}
```

### Adicionando Novo Banco
1. Atualize `bank_patterns` em `src/ocr/extractor.py`
2. Adicione padrões específicos do banco
3. Teste com comprovantes reais
4. Atualize a documentação

## 🧪 Testes

```bash
# Executar testes unitários
python -m pytest tests/

# Testar OCR específico
python tests/test_ocr.py

# Testar ML
python tests/test_ml.py
```

## 📝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👥 Autores

- **David Damasceno** - *Desenvolvedor Principal* - [@Davidamascen07](https://github.com/Davidamascen07)

## 🙏 Agradecimentos

- Equipe do Tesseract OCR
- Desenvolvedores do EasyOCR
- Comunidade Ultralytics (YOLO)
- Contribuidores do projeto

## 📞 Suporte

Para suporte, envie um email para [seu-email@exemplo.com] ou abra uma issue no GitHub.

---

⭐ Se este projeto te ajudou, considere dar uma estrela no GitHub!
