from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class PIXYOLODetector:
    def __init__(self, model_path: str = None):
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Usar modelo pré-treinado como base
            self.model = YOLO('yolov8n.pt')
        
        self.regions_of_interest = [
            'valor', 'data', 'nome_destinatario', 'nome_remetente', 
            'instituicao', 'id_transacao', 'qr_code'
        ]
    
    def detect_regions(self, image_path: str) -> Dict[str, List[Tuple]]:
        """Detecta regiões de interesse na imagem"""
        results = self.model(image_path)
        
        detected_regions = {}
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Coordenadas da caixa delimitadora
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Mapear para região de interesse (isso seria customizado)
                    region_name = self._map_class_to_region(class_id)
                    
                    if region_name not in detected_regions:
                        detected_regions[region_name] = []
                    
                    detected_regions[region_name].append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence)
                    })
        
        return detected_regions
    
    def _map_class_to_region(self, class_id: int) -> str:
        """Mapeia ID da classe para região de interesse"""
        # Isso seria customizado baseado no treinamento específico
        mapping = {
            0: 'valor',
            1: 'data',
            2: 'nome',
            3: 'instituicao',
            # ... outros mapeamentos
        }
        return mapping.get(class_id, 'unknown')
    
    def extract_roi_text(self, image_path: str, roi_detector) -> Dict:
        """Extrai texto das regiões de interesse detectadas"""
        regions = self.detect_regions(image_path)
        extracted_data = {}
        
        img = cv2.imread(image_path)
        
        for region_name, detections in regions.items():
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                roi = img[y1:y2, x1:x2]
                
                # Salvar ROI temporariamente
                roi_path = f"temp_roi_{region_name}.png"
                cv2.imwrite(roi_path, roi)
                
                # Extrair texto da ROI
                try:
                    text = roi_detector.extract_text_ocr(roi_path)[0]  # Usar Tesseract
                    if region_name not in extracted_data:
                        extracted_data[region_name] = []
                    extracted_data[region_name].append(text.strip())
                except:
                    pass
                finally:
                    # Limpar arquivo temporário
                    Path(roi_path).unlink(missing_ok=True)
        
        return extracted_data
