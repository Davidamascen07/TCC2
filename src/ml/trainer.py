import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple

class PIXMLTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models_path = Path('data/models')
        self.models_path.mkdir(exist_ok=True)
        
    def load_annotations(self, annotations_path: str) -> List[Dict]:
        """Carrega as anotações do arquivo JSON"""
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['anotacoes']
    
    def prepare_training_data(self, annotations: List[Dict]) -> Tuple[List[str], List[str]]:
        """Prepara dados de treinamento baseado nas anotações"""
        texts = []
        labels = []
        
        for annotation in annotations:
            # Criar texto combinado com todas as informações
            combined_text = f"""
            {annotation.get('valor', '')}
            {annotation.get('data', '')}
            {annotation.get('horario', '')}
            {annotation.get('destinatario', {}).get('nome', '')}
            {annotation.get('destinatario', {}).get('instituicao', '')}
            {annotation.get('remetente', {}).get('nome', '')}
            {annotation.get('remetente', {}).get('instituicao', '')}
            {annotation.get('tipo', '')}
            """.strip()
            
            # Identificar banco pela instituição
            instituicao_remetente = annotation.get('remetente', {}).get('instituicao', '').lower()
            instituicao_destinatario = annotation.get('destinatario', {}).get('instituicao', '').lower()
            
            bank_label = self._identify_bank_from_institution(
                instituicao_remetente, instituicao_destinatario
            )
            
            texts.append(combined_text)
            labels.append(bank_label)
        
        return texts, labels
    
    def _identify_bank_from_institution(self, inst_remetente: str, inst_destinatario: str) -> str:
        """Identifica o banco baseado nas instituições"""
        institutions = f"{inst_remetente} {inst_destinatario}"
        
        if 'nubank' in institutions or 'nu pagamentos' in institutions:
            return 'nubank'
        elif 'inter' in institutions:
            return 'inter'
        elif 'itau' in institutions:
            return 'itau'
        elif 'will' in institutions:
            return 'will'
        elif 'picpay' in institutions:
            return 'picpay'
        elif 'btg' in institutions:
            return 'btg'
        elif 'caixa' in institutions or 'econômica' in institutions:
            return 'caixa'
        elif 'brasil' in institutions or 'bco do brasil' in institutions:
            return 'bb'
        else:
            return 'unknown'
    
    def train_bank_classifier(self, annotations_path: str):
        """Treina classificador para identificação de bancos"""
        annotations = self.load_annotations(annotations_path)
        texts, labels = self.prepare_training_data(annotations)
        
        # Vetorização
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinamento
        self.classifier.fit(X_train, y_train)
        
        # Avaliação
        y_pred = self.classifier.predict(X_test)
        print("Relatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        # Salvar modelos
        joblib.dump(self.vectorizer, self.models_path / 'vectorizer.pkl')
        joblib.dump(self.classifier, self.models_path / 'bank_classifier.pkl')
        
        print(f"Modelos salvos em: {self.models_path}")
    
    def load_models(self):
        """Carrega modelos treinados"""
        try:
            self.vectorizer = joblib.load(self.models_path / 'vectorizer.pkl')
            self.classifier = joblib.load(self.models_path / 'bank_classifier.pkl')
            return True
        except FileNotFoundError:
            return False
    
    def predict_bank(self, text: str) -> str:
        """Prediz o banco baseado no texto"""
        if not hasattr(self.vectorizer, 'vocabulary_'):
            if not self.load_models():
                return 'unknown'
        
        X = self.vectorizer.transform([text])
        prediction = self.classifier.predict(X)[0]
        return prediction
