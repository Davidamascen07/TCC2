import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import re
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class ComprovantePIXExtractor:
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['pt', 'en'])
        self.bank_patterns = self._load_bank_patterns()
        
    def _load_bank_patterns(self) -> Dict:
        """Carrega padrões específicos para cada banco baseado nas anotações"""
        return {
            'nubank': {
                'identifiers': ['nubank', 'nu pagamentos'],
                'patterns': {
                    'valor': r'R\$\s*[\d.]+,\d{2}',
                    'data': r'\d{2}/\d{2}/\d{4}',
                    'horario': r'\d{2}:\d{2}:\d{2}',
                    'nome': r'[A-ZÁÊÇ][a-záêçãõ\s]+',
                    'cpf': r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}',
                    'cnpj': r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}',
                    'id_transacao': r'E\d{11,20}[a-zA-Z0-9]+',
                    'agencia': r'\d{4}',
                    'conta': r'\d{5,8}-?\d'
                }
            },
            'inter': {
                'identifiers': ['banco inter', 'inter'],
                'patterns': {
                    'valor': r'R\$\s*[\d.]+,\d{2}',
                    'data': r'\d{2}/\d{2}/\d{4}',
                    'horario': r'\d{2}h\d{2}',
                    'id_transacao': r'E\d{17}[a-zA-Z0-9]+',
                }
            },
            'itau': {
                'identifiers': ['itau', 'unibanco'],
                'patterns': {
                    'codigo_autenticacao': r'[A-F0-9]{40}',
                    'agencia': r'\d{4}',
                    'conta': r'\d{7}-\d'
                }
            },
            'will': {
                'identifiers': ['will bank'],
                'patterns': {
                    'chave_pix': r'\(\d{2}\)\s?\d{4,5}-\d{4}',
                    'descricao': r'[a-z\s]+',
                }
            },
            'picpay': {
                'identifiers': ['picpay'],
                'patterns': {
                    'tipo_conta': 'Conta de pagamentos'
                }
            },
            'btg': {
                'identifiers': ['btg pactual'],
                'patterns': {
                    'codigo_autenticacao': r'[a-f0-9]{32}'
                }
            },
            'caixa': {
                'identifiers': ['caixa', 'econômica federal'],
                'patterns': {
                    'codigo_operacao': r'\d{10}',
                    'chave_seguranca': r'[A-Z0-9]{16}',
                    'nsu': r'\d{11}'
                }
            },
            'bb': {
                'identifiers': ['banco do brasil', 'bco do brasil'],
                'patterns': {
                    'documento': r'\d{15}',
                    'autenticacao': r'[A-F0-9.]{3,20}'
                }
            }
        }

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Pré-processamento da imagem para melhorar OCR"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
        # Redimensionar se muito grande
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Melhorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarização adaptativa
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary

    def extract_text_ocr(self, image_path: str) -> Tuple[str, str]:
        """Extrai texto usando Tesseract e EasyOCR"""
        processed_img = self.preprocess_image(image_path)
        
        # Tesseract
        tesseract_text = pytesseract.image_to_string(
            processed_img, lang='por', config='--psm 6'
        )
        
        # EasyOCR
        easyocr_results = self.easyocr_reader.readtext(processed_img)
        easyocr_text = ' '.join([result[1] for result in easyocr_results])
        
        return tesseract_text, easyocr_text

    def identify_bank(self, text: str) -> str:
        """Identifica o banco baseado no texto extraído"""
        text_lower = text.lower()
        
        for bank, config in self.bank_patterns.items():
            for identifier in config['identifiers']:
                if identifier in text_lower:
                    return bank
        
        return 'unknown'

    def extract_structured_data(self, text: str, bank: str) -> Dict:
        """Extrai dados estruturados baseado no banco identificado"""
        data = {}
        
        if bank in self.bank_patterns:
            patterns = self.bank_patterns[bank]['patterns']
            
            for field, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    data[field] = matches[0] if len(matches) == 1 else matches
        
        # Padrões gerais para todos os bancos
        general_patterns = {
            'valor': r'R\$\s*[\d.]+,\d{2}',
            'data': r'\d{2}/\d{2}/\d{4}',
            'cpf': r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}',
            'cnpj': r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}',
        }
        
        for field, pattern in general_patterns.items():
            if field not in data:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    data[field] = matches[0] if len(matches) == 1 else matches
        
        return data

    def extract_names_advanced(self, text: str) -> Dict[str, str]:
        """Extração avançada de nomes usando padrões contextuais"""
        names = {}
        
        # Padrões para identificar seções de remetente e destinatário
        remetente_patterns = [
            r'(?:de|remetente|pagador)[:\s]*([A-ZÁÊÇ][a-záêçãõ\s]+)',
            r'nome[:\s]*([A-ZÁÊÇ][a-záêçãõ\s]+)',
        ]
        
        destinatario_patterns = [
            r'(?:para|destinat[aá]rio|recebedor)[:\s]*([A-ZÁÊÇ][a-záêçãõ\s]+)',
            r'(?:dados do pagador|dados da transação)[:\s\n]*([A-ZÁÊÇ][a-záêçãõ\s]+)',
        ]
        
        for pattern in remetente_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                names['remetente'] = match.group(1).strip()
                break
                
        for pattern in destinatario_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                names['destinatario'] = match.group(1).strip()
                break
        
        return names

    def process_comprovante(self, image_path: str) -> Dict:
        """Processa um comprovante completo"""
        try:
            tesseract_text, easyocr_text = self.extract_text_ocr(image_path)
            
            # Combinar textos para melhor precisão
            combined_text = f"{tesseract_text}\n{easyocr_text}"
            
            # Identificar banco
            bank = self.identify_bank(combined_text)
            
            # Extrair dados estruturados
            structured_data = self.extract_structured_data(combined_text, bank)
            
            # Extrair nomes
            names = self.extract_names_advanced(combined_text)
            
            # Montar resultado final
            result = {
                'arquivo_origem': Path(image_path).name,
                'banco_identificado': bank,
                'texto_extraido': {
                    'tesseract': tesseract_text,
                    'easyocr': easyocr_text
                },
                'dados_estruturados': structured_data,
                'nomes': names,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {
                'arquivo_origem': Path(image_path).name,
                'status': 'error',
                'erro': str(e)
            }
