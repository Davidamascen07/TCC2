import cv2
import numpy as np
import pytesseract
import re
import logging
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import os
from datetime import datetime

# Configurar logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprovantePIXExtractor:
    def __init__(self):
        self.easyocr_reader = None
        logger.info("EasyOCR desabilitado temporariamente (problema ANTIALIAS)")
            
        self.bank_patterns = self._load_bank_patterns()
        self.templates = self._load_templates()
        
        # Carregar modelos ML se disponíveis
        self.ml_classifier = None
        try:
            from src.ml.trainer import PIXMLTrainer
            self.ml_classifier = PIXMLTrainer()
            if self.ml_classifier.load_models():
                logger.info("Modelos ML carregados com sucesso")
            else:
                logger.warning("Modelos ML não encontrados")
        except ImportError:
            logger.warning("PIXMLTrainer não disponível")
        
        # Verificar se o Tesseract está disponível
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract detectado com sucesso")
        except Exception as e:
            logger.warning(f"Tesseract não encontrado: {e}")

    def _load_templates(self) -> Dict:
        """Templates específicos para cada banco baseados nas estruturas reais"""
        return {
            'nubank': {
                'header_patterns': ['comprovante de transferência', 'comprovante de', 'nu pagamentos'],
                'valor_patterns': [
                    r'valor\s+r\$\s*([\d.,]+)',
                    r'r\$\s*([\d.,]+)',
                    r'valor[\s:]*r\$\s*(\d+(?:\.\d{3})*,\d{2})',
                    r'(\d+,\d{2})'
                ],
                'data_patterns': [
                    r'(\d{1,2})\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\.?\s+(\d{4})',
                    r'(\d{1,2})/(\d{1,2})/(\d{4})',
                    r'(\d{2})\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\.?\s+(\d{4})',
                    r'(\d{2})/(\d{2})/(\d{4})\s*-\s*\d{2}:\d{2}:\d{2}'
                ],
                'nome_patterns': [
                    r'nome[\s:]*([a-záêçõã\s]+)',
                    r'destinatário[\s:]*([a-záêçõã\s]+)',
                    r'para[\s:]*([a-záêçõã\s]+)',
                    r'origem[\s\n]+nome[\s:]*([a-záêçõã\s]+)',
                    r'destino[\s\n]+nome[\s:]*([a-záêçõã\s]+)'
                ],
                'cpf_patterns': [
                    r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}',
                    r'\*{3}\.\d{3}\.\d{3}-\*{2}',
                    r'cpf[:\s]*\*{3}\.?\d{3}\.?\d{3}-?\*{2}'
                ],
                'id_patterns': [
                    r'E\d{11,20}[a-zA-Z0-9]+',
                    r'E\d{17}[a-zA-Z0-9]+',
                    r'id\s+da\s+transação[\s:]*E\d{11,20}[a-zA-Z0-9]+'
                ]
            },
            'inter': {
                'header_patterns': ['pix enviado', 'banco inter'],
                'valor_patterns': [
                    r'r\$\s*(\d+,\d{2})',
                    r'valor[\s:]*r\$\s*(\d+,\d{2})',
                    r'r\$[\s:]*(\d+,\d{2})'
                ],
                'data_patterns': [
                    r'(\d{1,2})/(\d{1,2})/(\d{4})',
                    r'(\d{2})/(\d{2})/(\d{4})',
                    r'data\s+do\s+pagamento[\s:]*\w+,\s*(\d{1,2})/(\d{1,2})/(\d{4})',
                    r'terça,\s*(\d{1,2})/(\d{1,2})/(\d{4})'
                ],
                'horario_patterns': [
                    r'(\d{2})h(\d{2})',
                    r'horário[\s:]*(\d{2})h(\d{2})'
                ],
                'id_patterns': [
                    r'E\d{11,20}[a-zA-Z0-9]+',
                    r'id\s+da\s+transação[\s:]*E\d{11,20}[a-zA-Z0-9]+'
                ]
            },
            'btg': {
                'header_patterns': ['btg', 'pactual', 'comprovante de transferência'],
                'valor_patterns': [
                    r'valor[\s\n]*r\$\s*(\d+,\d{2})',
                    r'r\$\s*(\d+,\d{2})'
                ],
                'data_patterns': [
                    r'(\d{1,2})\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\.?\s+(\d{4})',
                    r'(\d{2})/(\d{2})/(\d{4})'
                ],
                'id_patterns': [
                    r'ES\d{11,25}[a-zA-Z0-9]+',
                    r'E\d{11,20}[a-zA-Z0-9]+'
                ]
            },
            'bb': {
                'header_patterns': ['comprovante bb', 'pix enviado', 'bco do brasil'],
                'valor_patterns': [
                    r'r\$\s*(\d+,\d{2})',
                    r'valor[\s:]*r\$\s*(\d+,\d{2})'
                ],
                'data_patterns': [
                    r'(\d{2})/(\d{2})/(\d{4})',
                    r'às\s+(\d{2}):(\d{2}):(\d{2})'
                ]
            },
            'caixa': {
                'header_patterns': ['comprovante', 'pix', 'caixa'],
                'valor_patterns': [
                    r'r\$\s*(\d+,\d{2})',
                    r'valor[\s:]*r\$\s*(\d+,\d{2})'
                ],
                'data_patterns': [
                    r'(\d{2})/(\d{2})/(\d{4})',
                    r'data[\s:]*(\d{2})/(\d{2})/(\d{4})'
                ]
            },
            'itau': {
                'header_patterns': ['comprovante de transferência', 'pix por chave', 'itau'],
                'valor_patterns': [
                    r'r\$\s*(\d+,\d{2})',
                    r'valor[\s:]*r\$\s*(\d+,\d{2})'
                ],
                'data_patterns': [
                    r'(\d{2})\s*(mai|jun|jul)\s*(\d{4})',
                    r'(\d{2})/(\d{2})/(\d{4})'
                ]
            }
        }

    def _load_bank_patterns(self) -> Dict[str, List[str]]:
        """Carrega padrões para identificação de bancos"""
        return {
            'nubank': ['nubank', 'nu pagamentos', 'comprovante de transferência'],
            'inter': ['inter', 'banco inter', 'pix enviado'],
            'itau': ['itau', 'unibanco', 'pix por chave'],
            'bb': ['brasil', 'banco do brasil', 'comprovante bb', 'bco do brasil'],
            'caixa': ['caixa', 'econômica federal'],
            'btg': ['btg', 'pactual'],
            'will': ['will', 'will bank'],
            'picpay': ['picpay'],
            'pagbank': ['pagbank', 'pagseguro']
        }

    def identify_bank(self, text: str) -> str:
        """Identifica o banco usando ML primeiro, depois fallback para heurísticas"""
        text_lower = text.lower()
        
        # Tentar identificação via ML primeiro
        if self.ml_classifier:
            try:
                ml_prediction = self.ml_classifier.predict_bank(text)
                if ml_prediction != 'unknown':
                    logger.info(f"Banco identificado via ML: {ml_prediction}")
                    return ml_prediction
            except Exception as e:
                logger.warning(f"Erro na predição ML: {e}")
        
        # Fallback para identificação por padrões
        for bank, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    logger.info(f"Banco identificado via padrões: {bank}")
                    return bank
        
        logger.warning("Banco não identificado")
        return 'unknown'

    def _enhance_text_quality(self, img: np.ndarray) -> List[np.ndarray]:
        """Aplica múltiplas técnicas de melhoria específicas para texto de comprovantes"""
        enhanced_versions = []
        
        try:
            # Versão 1: Upscaling + Deblur
            height, width = img.shape[:2]
            # Aumentar resolução 2x usando interpolação cúbica
            upscaled = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
            
            # Aplicar filtro de nitidez (unsharp mask)
            gaussian = cv2.GaussianBlur(upscaled, (0,0), 2.0)
            sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
            enhanced_versions.append(sharpened)
            
            # Versão 2: Melhoria de contraste adaptativo
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(img)
            enhanced_versions.append(contrast_enhanced)
            
            # Versão 3: Remoção de ruído + Binarização Otsu
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
            _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_versions.append(binary_otsu)
            
            # Versão 4: Morfologia para melhorar caracteres
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            morph_close = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
            enhanced_versions.append(morph_close)
            
            # Versão 5: Dilatação leve para texto fino
            kernel_dilate = np.ones((1,1), np.uint8)
            dilated = cv2.dilate(img, kernel_dilate, iterations=1)
            enhanced_versions.append(dilated)
            
            logger.info(f"Geradas {len(enhanced_versions)} versões melhoradas")
            return enhanced_versions
            
        except Exception as e:
            logger.error(f"Erro na melhoria de texto: {e}")
            return [img]

    def _correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """Detecta bordas e alinha a imagem corrigindo perspectiva com melhoria para screenshots"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Para screenshots, usar detecção de bordas mais suave
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blur, 30, 100)  # Threshold mais baixo para screenshots
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    # Verificar se é um retângulo válido (não muito pequeno)
                    area = cv2.contourArea(approx)
                    img_area = gray.shape[0] * gray.shape[1]
                    
                    if area > img_area * 0.1:  # Pelo menos 10% da imagem
                        pts = approx.reshape(4, 2)
                        
                        # Ordenar pontos: topo-esquerda, topo-direita, baixo-direita, baixo-esquerda
                        rect = np.zeros((4, 2), dtype="float32")
                        s = pts.sum(axis=1)
                        rect[0] = pts[np.argmin(s)]
                        rect[2] = pts[np.argmax(s)]
                        
                        diff = np.diff(pts, axis=1)
                        rect[1] = pts[np.argmin(diff)]
                        rect[3] = pts[np.argmax(diff)]
                        
                        # Calcular dimensões do retângulo corrigido
                        (tl, tr, br, bl) = rect
                        widthA = np.linalg.norm(br - bl)
                        widthB = np.linalg.norm(tr - tl)
                        maxWidth = max(int(widthA), int(widthB))
                        
                        heightA = np.linalg.norm(tr - br)
                        heightB = np.linalg.norm(tl - bl)
                        maxHeight = max(int(heightA), int(heightB))
                        
                        # Pontos de destino
                        dst = np.array([
                            [0, 0],
                            [maxWidth - 1, 0],
                            [maxWidth - 1, maxHeight - 1],
                            [0, maxHeight - 1]
                        ], dtype="float32")
                        
                        # Aplicar transformação de perspectiva
                        M = cv2.getPerspectiveTransform(rect, dst)
                        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
                        
                        logger.info("Perspectiva corrigida com sucesso")
                        return warped
            
            logger.info("Nenhuma correção de perspectiva necessária")
            return img
            
        except Exception as e:
            logger.warning(f"Erro na correção de perspectiva: {e}")
            return img

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Pré-processa a imagem para melhorar OCR (agora com correção de perspectiva e contraste)"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            # 1) corrigir perspectiva
            img = self._correct_perspective(img)
            # 2) converter para tons de cinza
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 3) aumentar contraste e brilho
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            # 4) denoise + CLAHE + threshold
            denoised = cv2.medianBlur(gray, 3)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            binary = cv2.adaptiveThreshold(enhanced, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            logger.info("Pré-processamento avançado concluído")
            return binary
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.zeros((100,100), dtype=np.uint8)

    def preprocess_image_advanced(self, image_path: str) -> List[np.ndarray]:
        """Múltiplas versões de pré-processamento melhoradas para comprovantes"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            logger.info(f"Imagem carregada: {img.shape}")
            
            # 1. Redimensionar se muito grande (otimização)
            height, width = img.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Imagem redimensionada para: {new_width}x{new_height}")
            
            # 2. Corrigir perspectiva
            img_corrected = self._correct_perspective(img)
            
            # 3. Converter para tons de cinza
            gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
            
            # 4. Aplicar melhorias específicas para texto
            enhanced_versions = self._enhance_text_quality(gray)
            
            # 5. Adicionar versões com diferentes abordagens
            processed_versions = []
            
            for i, enhanced in enumerate(enhanced_versions):
                # Aplicar diferentes configurações de pré-processamento
                if i == 0:  # Versão com upscaling
                    processed_versions.append(enhanced)
                elif i == 1:  # Versão com contraste
                    # Aplicar denoising adicional
                    denoised = cv2.medianBlur(enhanced, 3)
                    processed_versions.append(denoised)
                elif i == 2:  # Versão binarizada
                    processed_versions.append(enhanced)
                elif i == 3:  # Versão com morfologia
                    processed_versions.append(enhanced)
                else:  # Versão com dilatação
                    # Aplicar threshold adaptativo
                    adaptive = cv2.adaptiveThreshold(enhanced, 255, 
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 11, 2)
                    processed_versions.append(adaptive)
            
            # 6. Adicionar versão original melhorada
            contrast_improved = cv2.convertScaleAbs(gray, alpha=1.3, beta=15)
            processed_versions.append(contrast_improved)
            
            logger.info(f"Geradas {len(processed_versions)} versões processadas")
            return processed_versions
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento avançado: {e}")
            return [self.preprocess_image(image_path)]

    def extract_text_multiple_configs(self, image_path: str) -> List[str]:
        """Extrai texto com múltiplas configurações otimizadas para comprovantes"""
        texts = []
        
        try:
            processed_images = self.preprocess_image_advanced(image_path)
            
            # Configurações específicas para comprovantes PIX
            configs = [
                '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÇÉÊÍÓÔÕÚÜàáâãçéêíóôõúü.,-:/$*() ',
                '--psm 4 --oem 3',
                '--psm 3 --oem 3',
                '--psm 7 --oem 3',
                '--psm 8 --oem 3',
                '--psm 11 --oem 3',
                '--psm 12 --oem 3',  # Texto esparso
                '--psm 6 --oem 1',   # Engine original
                '--psm 6 --oem 2'    # Engine antiga + nova
            ]
            
            for i, img in enumerate(processed_images):
                for j, config in enumerate(configs):
                    try:
                        text = pytesseract.image_to_string(img, lang='por', config=config)
                        if text and len(text.strip()) > 15:  # Texto mais substancial
                            texts.append(text)
                            logger.info(f"Config {j} na imagem {i}: {len(text)} caracteres - qualidade: {self._assess_text_quality(text)}")
                    except Exception as e:
                        logger.warning(f"Erro na config {j}, imagem {i}: {e}")
            
            # Ordenar por qualidade de texto
            texts = sorted(texts, key=self._assess_text_quality, reverse=True)
            return texts[:10]  # Retornar apenas os 10 melhores
            
        except Exception as e:
            logger.error(f"Erro na extração múltipla: {e}")
            return []

    def _assess_text_quality(self, text: str) -> float:
        """Avalia a qualidade do texto extraído baseado em padrões de comprovantes"""
        score = 0
        text_lower = text.lower()
        
        # Pontuação por palavras-chave importantes
        keywords = {
            'comprovante': 10, 'pix': 10, 'transferência': 8, 'valor': 8,
            'destinatário': 6, 'remetente': 6, 'data': 6, 'horário': 4,
            'transação': 8, 'banco': 6, 'agência': 4, 'conta': 4
        }
        
        for keyword, points in keywords.items():
            if keyword in text_lower:
                score += points
        
        # Pontuação por padrões estruturais
        if re.search(r'r\$\s*\d+[.,]\d{2}', text_lower):
            score += 15  # Valor monetário
        if re.search(r'\d{2}/\d{2}/\d{4}', text):
            score += 10  # Data
        if re.search(r'E\d{10,20}[a-zA-Z0-9]+', text):
            score += 12  # ID transação
        if re.search(r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}', text):
            score += 8   # CPF mascarado
        
        # Penalizar por caracteres especiais excessivos
        special_chars = len(re.findall(r'[^\w\s\.,\-:/$()áêçõã]', text))
        score -= special_chars * 0.5
        
        # Bonificação por tamanho apropriado
        if 100 <= len(text) <= 1500:
            score += 5
        
        return score

    def extract_with_templates(self, texts: List[str], bank: str) -> Dict:
        """Extrai dados usando templates específicos do banco com melhorias"""
        data = {}
        
        if bank not in self.templates:
            return self.extract_structured_data_generic(texts)
        
        template = self.templates[bank]
        combined_text = ' '.join(texts).lower()
        
        logger.info(f"Usando template do {bank} para extração")
        
        # Extrair valor com padrões específicos
        if 'valor_patterns' in template:
            for pattern in template['valor_patterns']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    valor = matches[0]
                    # Normalizar formato
                    if ',' in valor and '.' in valor:
                        valor = valor.replace('.', '').replace(',', '.')
                        valor = f"R$ {float(valor):.2f}".replace('.', ',')
                    elif ',' in valor:
                        valor = f"R$ {valor}"
                    else:
                        valor = f"R$ {valor},00"
                    
                    data['valor'] = valor
                    logger.info(f"Valor extraído: {valor}")
                    break
        
        # Extrair data com padrões melhorados
        if 'data_patterns' in template:
            for pattern in template['data_patterns']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    match = matches[0]
                    if len(match) == 3 and isinstance(match[1], str):
                        # Data com mês em texto
                        dia, mes_texto, ano = match
                        meses = {
                            'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
                            'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
                            'set': '09', 'out': '10', 'nov': '11', 'dez': '12'
                        }
                        mes = meses.get(mes_texto.lower().replace('.', ''), '01')
                        data['data'] = f"{dia.zfill(2)}/{mes}/{ano}"
                    elif len(match) == 3:
                        # Data numérica
                        dia, mes, ano = match
                        data['data'] = f"{dia.zfill(2)}/{mes.zfill(2)}/{ano}"
                    elif len(match) == 2:
                        # Formato especial (ex: dd/mm/yyyy extraído como (dd, yyyy))
                        data['data'] = f"{match[0]}/{match[1]}"
                    
                    logger.info(f"Data extraída: {data.get('data')}")
                    break
        
        # Extrair nomes com padrões melhorados
        if 'nome_patterns' in template:
            nomes_encontrados = []
            for pattern in template['nome_patterns']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    nome = match.strip().title()
                    # Limpar nomes muito curtos ou inválidos
                    if len(nome) > 3 and not any(char.isdigit() for char in nome):
                        nomes_encontrados.append(nome)
            
            # Atribuir nomes encontrados
            if nomes_encontrados:
                if 'destinatario' not in data:
                    data['destinatario'] = nomes_encontrados[0]
                    logger.info(f"Destinatário extraído: {nomes_encontrados[0]}")
                if len(nomes_encontrados) > 1 and 'remetente' not in data:
                    data['remetente'] = nomes_encontrados[1]
                    logger.info(f"Remetente extraído: {nomes_encontrados[1]}")
        
        # Extrair CPFs
        if 'cpf_patterns' in template:
            cpfs = []
            for pattern in template['cpf_patterns']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                cpfs.extend(matches)
            if cpfs:
                data['cpfs'] = list(set(cpfs))
                logger.info(f"CPFs extraídos: {cpfs}")
        
        # Extrair ID transação
        if 'id_patterns' in template:
            for pattern in template['id_patterns']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    data['id_transacao'] = matches[0]
                    logger.info(f"ID transação extraído: {matches[0]}")
                    break
        
        return data

    def extract_structured_data_generic(self, texts: List[str]) -> Dict:
        """Extração genérica melhorada quando não há template específico"""
        data = {}
        combined_text = ' '.join(texts)
        
        # Padrões genéricos melhorados
        patterns = {
            'valor': r'r\$\s*(\d+(?:[.,]\d{3})*[.,]\d{2})',
            'data': r'(\d{2})/(\d{2})/(\d{4})',
            'cpf': r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}',
            'id_transacao': r'E\d{11,20}[a-zA-Z0-9]+',
        }
        
        for field, pattern in patterns.items():
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                if field == 'data' and len(matches[0]) == 3:
                    data[field] = f"{matches[0][0]}/{matches[0][1]}/{matches[0][2]}"
                elif field == 'cpf':
                    data['cpfs'] = list(set(matches))
                else:
                    data[field] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return data

    def extract_names_advanced(self, text: str) -> Dict[str, str]:
        """Extrai nomes de pessoas do texto"""
        names = {}
        
        # Padrões para extrair nomes
        name_patterns = [
            r'nome[:\s]*([A-ZÁÊÇÕÃ][a-záêçõã\s]+)',
            r'destinatário[:\s]*([A-ZÁÊÇÕÃ][a-záêçõã\s]+)',
            r'remetente[:\s]*([A-ZÁÊÇÕÃ][a-záêçõã\s]+)',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_name = ' '.join(match.split())
                if len(clean_name) > 3:
                    if 'destinatario' not in names:
                        names['destinatario'] = clean_name
                    else:
                        names['remetente'] = clean_name
        
        return names

    def extract_text_ocr(self, image_path: str) -> Tuple[str, str]:
        """Extrai texto usando múltiplas estratégias"""
        try:
            # Estratégia 1: Múltiplas configurações
            texts = self.extract_text_multiple_configs(image_path)
            
            if texts:
                # Combinar o melhor resultado
                best_text = max(texts, key=len)
                logger.info(f"Melhor texto extraído: {len(best_text)} caracteres")
                return best_text, ""
            else:
                # Fallback para método original
                logger.warning("Usando método fallback")
                return self._fallback_ocr(image_path), ""
            
        except Exception as e:
            logger.error(f"Erro na extração de texto: {e}")
            return "", ""

    def _fallback_ocr(self, image_path: str) -> str:
        """Método fallback para OCR"""
        try:
            processed_img = self.preprocess_image(image_path)
            text = pytesseract.image_to_string(processed_img, lang='por', config='--psm 6')
            logger.info(f"Tesseract extraiu {len(text)} caracteres")
            return text
        except Exception as e:
            logger.error(f"Erro no fallback OCR: {e}")
            return ""

    def extract_simple_fallback(self, image_path: str) -> Dict:
        """Método simples como fallback baseado no Projeto.py"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}
            
            # Pré-processamento simples
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Múltiplas configurações do método simples
            configs = ['--psm 6', '--psm 11', '--psm 4', '']
            
            best_text = ""
            best_length = 0
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(gray, lang='por', config=config)
                    if len(text) > best_length:
                        best_text = text
                        best_length = len(text)
                except:
                    continue
            
            if not best_text:
                return {}
            
            # Extração com padrões simples
            dados = {}
            
            # Valor
            valores = re.findall(r'r\$\s*(\d+[.,]\d{2})', best_text, re.IGNORECASE)
            if valores:
                dados['valor'] = f"R$ {valores[0]}"
            
            # Data
            datas = re.findall(r'(\d{2})[\/\-](\d{2})[\/\-](\d{4})', best_text)
            if datas:
                dados['data'] = f"{datas[0][0]}/{datas[0][1]}/{datas[0][2]}"
            
            # CPF
            cpfs = re.findall(r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}', best_text)
            if cpfs:
                dados['cpfs'] = cpfs
            
            # ID transação
            ids = re.findall(r'E\d{10,20}[a-zA-Z0-9]+', best_text, re.IGNORECASE)
            if ids:
                dados['id_transacao'] = ids[0]
            
            logger.info(f"Fallback simples extraiu: {list(dados.keys())}")
            return dados
            
        except Exception as e:
            logger.error(f"Erro no fallback simples: {e}")
            return {}

    def _semantic_postprocess(self, text: str) -> Dict:
        """Pós-processamento semântico melhorado para extrair campos via regex"""
        res = {}
        text_lower = text.lower()
        
        # Data e hora com múltiplos formatos (melhorado)
        date_patterns = [
            r'(\d{2}[\/\-]\d{2}[\/\-]\d{4}\s*[-–]\s*\d{2}:\d{2}:\d{2})',
            r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',
            r'(\d{1,2})\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\.?\s+(\d{4})',
            r'(\d{2})\s+(mai|jun|jul)\.?\s+(\d{4})',  # Específico para casos encontrados
            r'terça,?\s*(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{2})/(\d{2})/(\d{4})\s*-\s*\d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in date_patterns:
            m = re.search(pattern, text_lower)
            if m:
                if len(m.groups()) == 3 and isinstance(m.group(2), str):
                    # Formato com mês por extenso
                    dia, mes_txt, ano = m.groups()
                    meses = {'jan':'01','fev':'02','mar':'03','abr':'04','mai':'05','jun':'06',
                            'jul':'07','ago':'08','set':'09','out':'10','nov':'11','dez':'12'}
                    mes = meses.get(mes_txt.replace('.',''), '01')
                    res['data'] = f"{dia.zfill(2)}/{mes}/{ano}"
                elif len(m.groups()) >= 3:
                    # Formato dd/mm/yyyy
                    res['data'] = f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"
                else:
                    res['data_hora'] = m.group(1)
                break
        
        # Valor (melhorado para diferentes formatos)
        value_patterns = [
            r'r\$\s*(\d+(?:[.,]\d{3})*[.,]\d{2})',
            r'valor[\s:]*r\$\s*(\d+(?:[.,]\d{3})*[.,]\d{2})',
            r'(\d+,\d{2})',  # Apenas números com vírgula
        ]
        
        for pattern in value_patterns:
            m = re.search(pattern, text_lower)
            if m:
                valor = m.group(1)
                # Normalizar formato
                if ',' in valor and '.' in valor:
                    valor = valor.replace('.', '').replace(',', '.')
                    valor = f"R$ {float(valor):.2f}".replace('.', ',')
                elif ',' in valor:
                    valor = f"R$ {valor}"
                else:
                    valor = f"R$ {valor},00"
                res['valor'] = valor
                break
        
        # Destinatário e Remetente (melhorado)
        name_patterns = [
            r'destino[\s\S]*?nome[\s:]*([A-ZÁ-Ú][A-Za-zÀ-ÿ\s]+?)(?=\n|\s{3,}|instituição|agência|cpf|$)',
            r'nome[\s:]*([A-ZÁ-Ú][A-Za-zÀ-ÿ\s]+?)(?=\n|\s{3,}|instituição|agência|cpf|$)',
            r'para[\s:]*([A-ZÁ-Ú][A-Za-zÀ-ÿ\s]+?)(?=\n|\s{3,}|instituição|agência|cpf|$)',
            r'de[\s:]*([A-ZÁ-Ú][A-Za-zÀ-ÿ\s]+?)(?=\n|\s{3,}|instituição|agência|cpf|$)'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                nome = match.strip().title()
                if len(nome) > 3 and len(nome) < 50 and not any(c.isdigit() for c in nome):
                    if 'destinatario' not in res:
                        res['destinatario'] = nome
                    elif 'remetente' not in res and nome != res.get('destinatario'):
                        res['remetente'] = nome
        
        # ID transação (melhorado com mais padrões)
        id_patterns = [
            r'ID\s+da\s+transa(?:ç|c)[aã]o[\s:]*([A-Z0-9]+)',
            r'E\d{11,20}[a-zA-Z0-9]+',
            r'E\d{17}[a-zA-Z0-9]+',
            r'id[\s:]*([A-Z0-9]{15,30})',
            r'transação[\s:]*([A-Z0-9]{15,30})'
        ]
        
        for pattern in id_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                res['id_transacao'] = m.group(1) if m.groups() else m.group(0)
                break
        
        # CPF (padrões melhorados)
        cpf_patterns = [
            r'\*{3}\.?\d{3}\.?\d{3}-?\*{2}',
            r'cpf[\s:]*\*{3}\.?\d{3}\.?\d{3}-?\*{2}',
            r'\*{3}\.\d{3}\.\d{3}-\*{2}'
        ]
        
        cpfs = []
        for pattern in cpf_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            cpfs.extend(matches)
        
        if cpfs:
            res['cpfs'] = list(set(cpfs))
        
        return res

    def merge_extraction_results(self, advanced_data: Dict, simple_data: Dict) -> Dict:
        """Combina os resultados dos dois métodos, priorizando o melhor de cada"""
        merged = advanced_data.copy()
        
        # Se o método simples encontrou valor e o avançado não
        if 'valor' not in merged and 'valor' in simple_data:
            merged['valor'] = simple_data['valor']
            logger.info("Usando valor do método simples")
        
        # Se o método simples encontrou data e o avançado não
        if 'data' not in merged and 'data' in simple_data:
            merged['data'] = simple_data['data']
            logger.info("Usando data do método simples")
        
        # Combinar CPFs
        cpfs_advanced = merged.get('cpfs', [])
        cpfs_simple = simple_data.get('cpfs', [])
        all_cpfs = list(set(cpfs_advanced + cpfs_simple))
        if all_cpfs:
            merged['cpfs'] = all_cpfs
            logger.info(f"CPFs combinados: {all_cpfs}")
        
        # Se o método simples encontrou ID e o avançado não
        if 'id_transacao' not in merged and 'id_transacao' in simple_data:
            merged['id_transacao'] = simple_data['id_transacao']
            logger.info("Usando ID transação do método simples")
        
        return merged

    def _calculate_extraction_success(self, data: Dict) -> float:
        """Calcula score de sucesso da extração baseado nos dados extraídos"""
        score = 0.0
        weights = {
            'valor': 0.3,
            'data': 0.2,
            'destinatario': 0.15,
            'remetente': 0.15,
            'cpfs': 0.1,
            'id_transacao': 0.1
        }
        
        for field, weight in weights.items():
            if field in data and data[field]:
                score += weight
        
        return score

    def process_comprovante(self, image_path: str) -> Dict:
        """Processa um comprovante real extraindo dados reais (não fictícios)"""
        try:
            logger.info(f"🔍 PROCESSANDO COMPROVANTE REAL: {image_path}")
            
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")
            
            # Etapa 1: Extrair texto com OCR
            logger.info("📝 Extraindo texto com OCR...")
            tesseract_text, easyocr_text = self.extract_text_ocr(image_path)
            
            if not tesseract_text.strip():
                logger.warning("⚠️ Nenhum texto extraído via OCR")
                return {
                    'arquivo_origem': Path(image_path).name,
                    'status': 'error',
                    'erro': 'Nenhum texto extraído do comprovante'
                }
            
            logger.info(f"✅ Texto extraído: {len(tesseract_text)} caracteres")
            
            # Etapa 2: Identificar banco
            logger.info("🏦 Identificando banco...")
            bank = self.identify_bank(tesseract_text)
            logger.info(f"✅ Banco identificado: {bank}")
            
            # Etapa 3: Extrair dados estruturados
            logger.info("🔬 Extraindo dados estruturados...")
            structured_data = self.extract_with_templates([tesseract_text], bank)
            
            # Etapa 4: Extrair nomes
            names = self.extract_names_advanced(tesseract_text)
            
            # Validar se extraiu dados essenciais
            success_score = self._calculate_extraction_success(structured_data)
            
            logger.info(f"📊 Score de sucesso: {success_score:.1%}")
            logger.info(f"✅ Campos extraídos: {list(structured_data.keys())}")
            
            # Resultado final com dados REAIS extraídos
            result = {
                'arquivo_origem': Path(image_path).name,
                'banco_identificado': bank,
                'texto_extraido': {
                    'tesseract': tesseract_text,
                    'easyocr': easyocr_text,
                    'caracteres_extraidos': len(tesseract_text)
                },
                'dados_estruturados': structured_data,
                'nomes': names,
                'score_extracao': success_score,
                'status': 'success' if success_score > 0.3 else 'partial',
                'timestamp': str(datetime.now())
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar {image_path}: {str(e)}")
            return {
                'arquivo_origem': Path(image_path).name,
                'status': 'error',
                'erro': str(e),
                'timestamp': str(datetime.now())
            }
