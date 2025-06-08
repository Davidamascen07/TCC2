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
from collections import Counter
import warnings

class PIXMLTrainer:
    def __init__(self, min_samples_per_class=3):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models_path = Path('data/models')
        self.models_path.mkdir(exist_ok=True)
        self.min_samples_per_class = min_samples_per_class
        
    def load_annotations(self, annotations_path: str) -> List[Dict]:
        """Carrega as anotações do arquivo JSON"""
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['anotacoes']
    
    def prepare_training_data(self, annotations: List[Dict]) -> Tuple[List[str], List[str]]:
        """Prepara dados de treinamento baseado nas anotações com agrupamento inteligente"""
        texts = []
        labels = []
        
        # Primeira passagem: coletar todos os labels para análise
        temp_labels = []
        temp_texts = []
        
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
            
            temp_texts.append(combined_text)
            temp_labels.append(bank_label)
        
        # Analisar distribuição de classes
        label_counts = Counter(temp_labels)
        print(f"📊 Distribuição de classes antes do agrupamento:")
        for label, count in sorted(label_counts.items()):
            print(f"   {label}: {count} exemplos")
        
        # Agrupar classes com poucos exemplos
        major_classes = {label for label, count in label_counts.items() 
                        if count >= self.min_samples_per_class}
        
        print(f"\n🎯 Classes principais (>= {self.min_samples_per_class} exemplos): {sorted(major_classes)}")
        
        # Segunda passagem: aplicar agrupamento
        for text, label in zip(temp_texts, temp_labels):
            if label in major_classes:
                final_label = label
            else:
                final_label = 'outros_bancos'  # Agrupa bancos com poucos exemplos
            
            texts.append(text)
            labels.append(final_label)
        
        # Mostrar distribuição final
        final_label_counts = Counter(labels)
        print(f"\n📊 Distribuição final de classes:")
        for label, count in sorted(final_label_counts.items()):
            print(f"   {label}: {count} exemplos")
        
        return texts, labels
    
    def _identify_bank_from_institution(self, inst_remetente: str, inst_destinatario: str) -> str:
        """Identifica o banco baseado nas instituições"""
        institutions = f"{inst_remetente} {inst_destinatario}".lower()
        
        # Mapeamento mais abrangente baseado nas anotações reais
        if any(term in institutions for term in ['nubank', 'nu pagamentos']):
            return 'nubank'
        elif any(term in institutions for term in ['inter', 'banco inter']):
            return 'inter'
        elif any(term in institutions for term in ['itau', 'unibanco']):
            return 'itau'
        elif 'will' in institutions:
            return 'will'
        elif 'picpay' in institutions:
            return 'picpay'
        elif any(term in institutions for term in ['btg', 'pactual']):
            return 'btg'
        elif any(term in institutions for term in ['caixa', 'econômica']):
            return 'caixa'
        elif any(term in institutions for term in ['brasil', 'bco do brasil']):
            return 'bb'
        elif any(term in institutions for term in ['pagbank', 'pagseguro']):
            return 'pagbank'
        elif any(term in institutions for term in ['cloudwalk', 'mercado pago']):
            return 'fintech'
        else:
            return 'unknown'
    
    def train_bank_classifier(self, annotations_path: str):
        """Treina classificador para identificação de bancos com validação inteligente"""
        print("🚀 Iniciando treinamento do classificador de bancos...")
        
        try:
            annotations = self.load_annotations(annotations_path)
            print(f"📁 Carregadas {len(annotations)} anotações")
            
            texts, labels = self.prepare_training_data(annotations)
            
            # Verificar se há dados suficientes
            if len(set(labels)) < 2:
                print("⚠️  Aviso: Apenas uma classe encontrada. Criando modelo básico...")
                self._create_basic_model(texts, labels)
                return
            
            # Vetorização
            print("🔄 Vetorizando textos...")
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Verificar se podemos fazer stratify
            label_counts = Counter(labels)
            min_count = min(label_counts.values())
            
            if min_count >= 2:
                # Pode usar stratify
                print("✅ Usando divisão estratificada")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                use_validation = True
            else:
                # Não pode usar stratify
                print("⚠️  Usando divisão simples (sem estratificação)")
                warnings.filterwarnings('ignore')
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                use_validation = len(X_test) > 0 and len(set(y_test)) > 1
            
            # Treinamento
            print("🧠 Treinando modelo...")
            self.classifier.fit(X_train, y_train)
            
            # Avaliação (se possível)
            if use_validation and len(X_test) > 0:
                y_pred = self.classifier.predict(X_test)
                print("\n📊 Relatório de Classificação:")
                print(classification_report(y_test, y_pred, zero_division=0))
                
                # Calcular acurácia
                accuracy = np.mean(y_pred == y_test)
                print(f"🎯 Acurácia: {accuracy:.2%}")
            else:
                print("ℹ️  Validação não disponível (dados insuficientes)")
            
            # Salvar modelos
            print("💾 Salvando modelos...")
            joblib.dump(self.vectorizer, self.models_path / 'vectorizer.pkl')
            joblib.dump(self.classifier, self.models_path / 'bank_classifier.pkl')
            
            # Salvar metadados
            metadata = {
                'classes': list(set(labels)),
                'total_samples': len(texts),
                'class_distribution': dict(Counter(labels)),
                'min_samples_per_class': self.min_samples_per_class
            }
            
            with open(self.models_path / 'model_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Modelos salvos em: {self.models_path}")
            print(f"📋 Classes treinadas: {sorted(set(labels))}")
            
        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            # Criar modelo básico em caso de erro
            self._create_basic_model(['texto_exemplo'], ['nubank'])
    
    def _create_basic_model(self, texts: List[str], labels: List[str]):
        """Cria um modelo básico quando há problemas com os dados"""
        print("🔧 Criando modelo básico...")
        
        # Adicionar exemplos sintéticos para ter pelo menos 2 classes
        synthetic_texts = [
            "nubank nu pagamentos transferência",
            "inter banco inter pix",
            "itau unibanco transferência",
            "outros banco transferência"
        ]
        synthetic_labels = ['nubank', 'inter', 'itau', 'outros_bancos']
        
        all_texts = texts + synthetic_texts
        all_labels = labels + synthetic_labels
        
        # Vetorizar e treinar
        X = self.vectorizer.fit_transform(all_texts)
        self.classifier.fit(X, all_labels)
        
        # Salvar
        joblib.dump(self.vectorizer, self.models_path / 'vectorizer.pkl')
        joblib.dump(self.classifier, self.models_path / 'bank_classifier.pkl')
        
        print("✅ Modelo básico criado com dados sintéticos")
    
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
        try:
            if not hasattr(self.vectorizer, 'vocabulary_'):
                if not self.load_models():
                    return 'unknown'
            
            X = self.vectorizer.transform([text])
            prediction = self.classifier.predict(X)[0]
            
            # Se predisse 'outros_bancos', tentar identificação manual
            if prediction == 'outros_bancos':
                manual_prediction = self._manual_bank_identification(text)
                return manual_prediction if manual_prediction != 'unknown' else prediction
            
            return prediction
            
        except Exception as e:
            print(f"⚠️  Erro na predição: {e}")
            return self._manual_bank_identification(text)
    
    def _manual_bank_identification(self, text: str) -> str:
        """Identificação manual baseada em palavras-chave como fallback"""
        text_lower = text.lower()
        
        # Padrões específicos baseados nas anotações
        patterns = {
            'nubank': ['nubank', 'nu pagamentos'],
            'inter': ['inter', 'banco inter'],
            'itau': ['itau', 'unibanco'],
            'will': ['will bank', 'will'],
            'picpay': ['picpay'],
            'btg': ['btg', 'pactual'],
            'caixa': ['caixa', 'econômica federal'],
            'bb': ['brasil', 'bco do brasil'],
            'pagbank': ['pagbank', 'pagseguro'],
            'c6': ['c6 bank', 'banco c6']
        }
        
        for bank, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return bank
        
        return 'unknown'
    
    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo treinado"""
        try:
            metadata_path = self.models_path / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {'status': 'Modelo não encontrado'}
        except Exception as e:
            return {'status': f'Erro: {e}'}
