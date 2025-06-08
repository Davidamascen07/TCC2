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
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.min_samples_per_class = min_samples_per_class
        
    def load_annotations(self, annotations_path: str) -> List[Dict]:
        """Carrega as anotações do arquivo JSON com dados REAIS"""
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('anotacoes', [])
        except FileNotFoundError:
            print(f"⚠️ Arquivo de anotações não encontrado: {annotations_path}")
            return []
        except Exception as e:
            print(f"⚠️ Erro ao carregar anotações: {e}")
            return []
    
    def prepare_training_data_real(self, annotations: List[Dict]) -> Tuple[List[str], List[str]]:
        """Prepara dados de treinamento REAIS baseado nas anotações existentes"""
        texts = []
        labels = []
        
        print(f"📚 Processando {len(annotations)} anotações reais...")
        
        for i, annotation in enumerate(annotations):
            # Extrair texto real das anotações
            combined_features = self._extract_real_features(annotation)
            
            # Identificar banco real pela instituição
            bank_label = self._identify_real_bank(annotation)
            
            texts.append(combined_features)
            labels.append(bank_label)
            
            if i % 20 == 0:
                print(f"   Processadas {i}/{len(annotations)} anotações...")
        
        # Analisar distribuição real
        label_counts = Counter(labels)
        print(f"\n📊 Distribuição REAL de bancos:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {label}: {count} comprovantes")
        
        # Agrupar classes com poucos exemplos (dados reais)
        major_classes = {label for label, count in label_counts.items() 
                        if count >= self.min_samples_per_class}
        
        print(f"\n🎯 Classes principais (>= {self.min_samples_per_class} exemplos): {sorted(major_classes)}")
        
        # Aplicar agrupamento inteligente para dados reais
        final_texts = []
        final_labels = []
        
        for text, label in zip(texts, labels):
            if label in major_classes:
                final_label = label
            else:
                final_label = 'outros_bancos'
            
            final_texts.append(text)
            final_labels.append(final_label)
        
        # Mostrar distribuição final
        final_counts = Counter(final_labels)
        print(f"\n📊 Distribuição final (pós-agrupamento):")
        for label, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {label}: {count} exemplos")
        
        return final_texts, final_labels

    def _extract_real_features(self, annotation: Dict) -> str:
        """Extrai características reais dos comprovantes para treinamento"""
        features = []
        
        # Dados do destinatário (reais)
        if 'destinatario' in annotation:
            dest = annotation['destinatario']
            if 'nome' in dest:
                features.append(f"destinatario_nome {dest['nome']}")
            if 'instituicao' in dest:
                features.append(f"destinatario_banco {dest['instituicao']}")
            if 'cpf' in dest:
                features.append(f"destinatario_cpf {dest['cpf']}")
        
        # Dados do remetente (reais)
        if 'remetente' in annotation:
            rem = annotation['remetente']
            if 'nome' in rem:
                features.append(f"remetente_nome {rem['nome']}")
            if 'instituicao' in rem:
                features.append(f"remetente_banco {rem['instituicao']}")
            if 'cpf' in rem:
                features.append(f"remetente_cpf {rem['cpf']}")
        
        # Dados da transação (reais)
        if 'valor' in annotation:
            features.append(f"valor {annotation['valor']}")
        if 'data' in annotation:
            features.append(f"data {annotation['data']}")
        if 'horario' in annotation:
            features.append(f"horario {annotation['horario']}")
        if 'tipo' in annotation:
            features.append(f"tipo {annotation['tipo']}")
        if 'id_transacao' in annotation:
            features.append(f"id_transacao {annotation['id_transacao']}")
        
        # Dados técnicos (reais)
        if 'arquivo_origem' in annotation:
            # Extrair informações do nome do arquivo
            filename = annotation['arquivo_origem'].lower()
            if 'nubank' in filename:
                features.append("arquivo_nubank")
            elif 'inter' in filename:
                features.append("arquivo_inter")
            elif 'itau' in filename:
                features.append("arquivo_itau")
        
        return ' '.join(features)

    def _identify_real_bank(self, annotation: Dict) -> str:
        """Identifica o banco REAL baseado nas instituições das anotações"""
        # Verificar remetente e destinatário
        institutions = []
        
        if 'remetente' in annotation and 'instituicao' in annotation['remetente']:
            institutions.append(annotation['remetente']['instituicao'].lower())
        
        if 'destinatario' in annotation and 'instituicao' in annotation['destinatario']:
            institutions.append(annotation['destinatario']['instituicao'].lower())
        
        # Combinar todas as instituições
        all_institutions = ' '.join(institutions)
        
        # Mapeamento baseado nos dados REAIS das anotações
        if any(term in all_institutions for term in ['nubank', 'nu pagamentos']):
            return 'nubank'
        elif any(term in all_institutions for term in ['inter', 'banco inter']):
            return 'inter'
        elif any(term in all_institutions for term in ['itau', 'unibanco']):
            return 'itau'
        elif any(term in all_institutions for term in ['btg', 'pactual']):
            return 'btg'
        elif 'will' in all_institutions:
            return 'will'
        elif 'picpay' in all_institutions:
            return 'picpay'
        elif any(term in all_institutions for term in ['caixa', 'econômica']):
            return 'caixa'
        elif any(term in all_institutions for term in ['brasil', 'bco do brasil']):
            return 'bb'
        elif any(term in all_institutions for term in ['pagbank', 'pagseguro']):
            return 'pagbank'
        elif any(term in all_institutions for term in ['cloudwalk', 'mercado pago', 'conpay']):
            return 'fintech'
        else:
            return 'unknown'
    
    def train_bank_classifier(self, annotations_path: str = None):
        """Treina classificador usando dados REAIS das anotações"""
        print("🚀 Iniciando treinamento com dados REAIS...")
        
        try:
            # Usar caminho padrão se não especificado
            if annotations_path is None:
                annotations_path = 'data/annotations/comprovantes_anotados.json'
            
            annotations = self.load_annotations(annotations_path)
            
            if not annotations:
                print("⚠️ Nenhuma anotação encontrada. Criando modelo básico...")
                self._create_basic_model(['texto_exemplo'], ['nubank'])
                return

            print(f"📁 Carregadas {len(annotations)} anotações reais")
            
            texts, labels = self.prepare_training_data_real(annotations)
            
            # Verificar se há dados suficientes
            if len(set(labels)) < 2:
                print("⚠️ Apenas uma classe encontrada. Adicionando dados sintéticos...")
                texts, labels = self._augment_with_synthetic_data(texts, labels)
            
            # Vetorização
            print("🔄 Vetorizando textos reais...")
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Divisão estratificada se possível
            label_counts = Counter(labels)
            min_count = min(label_counts.values())
            
            if min_count >= 2:
                print("✅ Usando divisão estratificada")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                use_validation = True
            else:
                print("⚠️ Usando divisão simples")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                use_validation = len(X_test) > 0 and len(set(y_test)) > 1
            
            # Treinamento
            print("🧠 Treinando modelo com dados reais...")
            self.classifier.fit(X_train, y_train)
            
            # Avaliação
            if use_validation and len(X_test) > 0:
                y_pred = self.classifier.predict(X_test)
                print("\n📊 Relatório de Classificação (DADOS REAIS):")
                print(classification_report(y_test, y_pred, zero_division=0))
                
                accuracy = np.mean(y_pred == y_test)
                print(f"🎯 Acurácia com dados reais: {accuracy:.2%}")
            
            # Salvar modelos
            print("💾 Salvando modelos treinados com dados reais...")
            joblib.dump(self.vectorizer, self.models_path / 'vectorizer.pkl')
            joblib.dump(self.classifier, self.models_path / 'bank_classifier.pkl')
            
            # Salvar metadados
            metadata = {
                'classes': list(set(labels)),
                'total_samples': len(texts),
                'class_distribution': dict(Counter(labels)),
                'min_samples_per_class': self.min_samples_per_class,
                'data_source': 'real_annotations',
                'training_date': str(pd.Timestamp.now())
            }
            
            with open(self.models_path / 'model_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Modelos treinados com DADOS REAIS salvos em: {self.models_path}")
            print(f"📋 Classes treinadas: {sorted(set(labels))}")
            
        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            self._create_basic_model(['texto_exemplo'], ['nubank'])

    def _augment_with_synthetic_data(self, texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """Aumenta dados reais com exemplos sintéticos se necessário"""
        print("🔧 Aumentando dados reais com exemplos sintéticos...")
        
        # Exemplos sintéticos baseados nos padrões reais
        synthetic_examples = [
            ("nubank nu pagamentos transferencia pix", "nubank"),
            ("inter banco inter pix enviado", "inter"),
            ("itau unibanco transferencia", "itau"),
            ("btg pactual banco", "btg"),
            ("will bank", "will"),
            ("picpay transferencia", "picpay"),
            ("outros banco qualquer transferencia", "outros_bancos")
        ]
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        for text, label in synthetic_examples:
            augmented_texts.append(text)
            augmented_labels.append(label)
        
        print(f"📈 Dados aumentados: {len(texts)} reais + {len(synthetic_examples)} sintéticos")
        return augmented_texts, augmented_labels

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
            print(f"⚠️ Erro na predição: {e}")
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
