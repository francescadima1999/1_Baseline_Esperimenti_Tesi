# -*- coding: utf-8 -*-

import spacy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import warnings
from collections import Counter, defaultdict
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# Import per file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Carica modello spaCy
try:
    nlp = spacy.load("en_core_web_md")
    print("Modello spaCy caricato correttamente")
except OSError:
    print("ERRORE: Modello spaCy non trovato.")
    exit()

class UltraOptimizedClassifier:
    
    def __init__(self, tuple_file_path=None):
        self.tuple_df = None
        self.campaigns = []
        self.tuple_col = None
        self.campaign_col = None
        self.id_col = None
        
        # Memorizza tutto il testo per ogni campagna
        self.campaign_texts = {}
        self.campaign_tuples = {}
        self.campaign_keywords = {}
        self.campaign_entities = {}
        
        # Pattern speciali per contenuti specifici
        self.twitter_patterns = [
            r'RT\s+@\w+',
            r'^@\w+',
            r'#\w+',
            r'\bRT\b'
        ]
        
        self.election_keywords = [
            'clinton', 'hillary', 'trump', 'election', 'vote', 'voting', 'ballot', 
            'campaign', 'president', 'presidential', 'democrat', 'republican',
            'electoral', 'polls', 'primary', 'candidate'
        ]
        
        self.climate_keywords = [
            'climate', 'global warming', 'temperature', 'weather', 'solar', 'sunspot',
            'atmosphere', 'cooling', 'warming', 'ice age', 'nasa', 'scientist'
        ]
        
        self.doctor_keywords = [
            'doctor', 'holistic', 'death', 'murder', 'killed', 'medical', 'cancer',
            'treatment', 'alternative medicine', 'suspicious death'
        ]
        
        # TF-IDF vectorizer ottimizzato
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 4),  # Aumentato per catturare frasi
            min_df=1,
            max_df=0.95,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Solo parole alfabetiche
        )
        
        self.tfidf_matrix = None
        self.campaign_tfidf_profiles = {}
        
        # Parametri 
        self.min_confidence_threshold = 0.12  
        self.tfidf_weight = 0.5               # Peso principale
        self.embedding_weight = 0.25          # Peso secondario  
        self.keyword_weight = 0.15            # Peso keywords
        self.special_bonus_weight = 0.1       #bonus speciali
        
        if tuple_file_path:
            self.load_opencti_tuples(tuple_file_path)
    
    def _get_file_path(self, prompt_message):
        """Ottiene il percorso del file"""
        if HAS_TKINTER:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title=prompt_message,
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )
            root.destroy()
            return file_path
        else:
            return input(f"{prompt_message}\nInserisci il percorso completo del file: ")
    
    def _is_twitter_content(self, text):
        """Verifica se il contenuto è Twitter"""
        text_str = str(text).strip()
        twitter_indicators = 0
        
        for pattern in self.twitter_patterns:
            if re.search(pattern, text_str, re.IGNORECASE):
                twitter_indicators += 1
        
        return twitter_indicators >= 1
    
    def _extract_enhanced_keywords(self, tupla):
        """Estrae keywords migliorato con più strategie"""
        if pd.isna(tupla):
            return []
            
        tupla_str = str(tupla)
        
        # 1. Rimuovi parentesi esplicative
        cleaned = re.sub(r'\([^)]*\)', '', tupla_str)
        
        # 2. Estrai parti separate da trattini
        parts = re.split(r'[-–—]', cleaned)
        
        keywords = set()
        
        for part in parts:
            part = part.strip()
            if len(part) > 1:
                # 3. Processamento NLP
                doc = nlp(part)
                
                # Entità nominate
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                        keywords.add(ent.text.lower())
                        # Aggiungi anche singole parole dell'entità
                        for word in ent.text.split():
                            if len(word) > 2:
                                keywords.add(word.lower())
                
                # Token importanti
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and 
                        token.is_alpha and 
                        len(token.text) > 2 and
                        not token.is_stop):
                        keywords.add(token.lemma_.lower())
                        keywords.add(token.text.lower())
                
                # 4. Estrai frasi complete significative
                if len(part.split()) > 1 and len(part.split()) < 6:
                    keywords.add(part.lower().strip())
        
        # 5. Aggiungi varianti comuni
        enhanced_keywords = set(keywords)
        for keyword in list(keywords):
            # Plurali/singolari
            if keyword.endswith('s') and len(keyword) > 3:
                enhanced_keywords.add(keyword[:-1])
            elif not keyword.endswith('s'):
                enhanced_keywords.add(keyword + 's')
        
        return list(enhanced_keywords)
    
    def _calculate_special_content_bonus(self, text, campaign):
        """Calcola bonus per tipi di contenuto speciali"""
        text_lower = text.lower()
        bonus = 0.0
        
        # 1. Bonus Twitter per campagne russe
        if self._is_twitter_content(text) and 'russian troll' in campaign.lower():
            bonus += 0.4
        
        # 2. Bonus keywords tematici
        if 'election' in campaign.lower() or 'hyperpartisan' in campaign.lower():
            election_matches = sum(1 for kw in self.election_keywords if kw in text_lower)
            bonus += min(election_matches * 0.05, 0.3)
        
        if 'climate' in campaign.lower():
            climate_matches = sum(1 for kw in self.climate_keywords if kw in text_lower)
            bonus += min(climate_matches * 0.06, 0.35)
        
        if 'doctor' in campaign.lower():
            doctor_matches = sum(1 for kw in self.doctor_keywords if kw in text_lower)
            bonus += min(doctor_matches * 0.07, 0.4)
        
        # 3. Bonus per nomi di persone specifiche
        person_patterns = [
            r'\bhillary\b', r'\bclinton\b', r'\btrump\b', r'\bobama\b',
            r'\bbush\b', r'\bkasich\b'
        ]
        
        person_matches = sum(1 for pattern in person_patterns 
                           if re.search(pattern, text_lower))
        if person_matches > 0 and ('election' in campaign.lower() or 'hyperpartisan' in campaign.lower()):
            bonus += min(person_matches * 0.08, 0.25)
        
        return min(bonus, 0.6)  # Cap totale
    
    def load_opencti_tuples(self, file_path=None):
        """Carica e processa le tuple"""
        if not file_path:
            file_path = self._get_file_path("Seleziona il file delle tuple OpenCTI")
        
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")
        
        try:
            if file_path.endswith('.xlsx'):
                self.tuple_df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                self.tuple_df = pd.read_csv(file_path)
            else:
                raise ValueError("Formato file non supportato")
        except Exception as e:
            raise Exception(f"Errore nel caricamento: {e}")
        
        print(f"Caricate {len(self.tuple_df)} tuple")
        print(f"Colonne: {self.tuple_df.columns.tolist()}")
        
        self._identify_columns()
        self._build_ultra_optimized_profiles()
        
        return self
    
    def _identify_columns(self):
        """Identifica le colonne del dataset"""
        cols = [col.lower() for col in self.tuple_df.columns]
        
        # Trova colonna tuple
        tuple_candidates = ['tupla', 'tuple', 'semantic_tuple']
        self.tuple_col = None
        for candidate in tuple_candidates:
            for i, col in enumerate(cols):
                if candidate in col:
                    self.tuple_col = self.tuple_df.columns[i]
                    break
            if self.tuple_col:
                break
        
        if not self.tuple_col:
            print("Colonne disponibili:", self.tuple_df.columns.tolist())
            col_input = input("Inserisci il nome della colonna tuple: ")
            if col_input in self.tuple_df.columns:
                self.tuple_col = col_input
            else:
                raise ValueError(f"Colonna '{col_input}' non trovata")
        
        # Trova colonna campagna
        campaign_candidates = ['campagna', 'campaign']
        self.campaign_col = None
        for candidate in campaign_candidates:
            for i, col in enumerate(cols):
                if candidate in col:
                    self.campaign_col = self.tuple_df.columns[i]
                    break
            if self.campaign_col:
                break
        
        if not self.campaign_col:
            print("Colonne disponibili:", self.tuple_df.columns.tolist())
            col_input = input("Inserisci il nome della colonna campagna: ")
            if col_input in self.tuple_df.columns:
                self.campaign_col = col_input
            else:
                raise ValueError(f"Colonna '{col_input}' non trovata")
        
        print(f"Colonna tuple: {self.tuple_col}")
        print(f"Colonna campagna: {self.campaign_col}")
    
    def _build_ultra_optimized_profiles(self):
        """Costruisce profili per ogni campagna"""
        print("Costruendo profili per ogni campagna...")
        
        self.campaigns = self.tuple_df[self.campaign_col].unique()
        self.campaigns = [c for c in self.campaigns if pd.notna(c)]
        
        # Per ogni campagna, costruisco profili multipli
        all_campaign_texts = []
        
        for campaign in self.campaigns:
            campaign_tuples = self.tuple_df[
                self.tuple_df[self.campaign_col] == campaign
            ][self.tuple_col].tolist()
            
            # 1. Testo pulito per TF-IDF
            clean_tuples = []
            all_keywords = []
            all_entities = []
            
            for tupla in campaign_tuples:
                if pd.notna(tupla):
                    tupla_str = str(tupla).strip()
                    
                    # Pulisci per TF-IDF
                    clean_tuple = re.sub(r'\([^)]*\)', '', tupla_str)
                    clean_tuple = re.sub(r'[-–—]', ' ', clean_tuple)
                    clean_tuple = re.sub(r'\s+', ' ', clean_tuple).strip()
                    clean_tuples.append(clean_tuple)
                    
                    # Estrai keywords avanzate
                    keywords = self._extract_enhanced_keywords(tupla_str)
                    all_keywords.extend(keywords)
                    
                    # Estrai entità specifiche
                    doc = nlp(clean_tuple)
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                            all_entities.append(ent.text.lower())
            
            # 2. Crea testi multipli per la campagna
            
            # Testo base (tuple pulite)
            base_text = ' '.join(clean_tuples)
            
            # Testo keywords (ripeti keywords importanti)
            unique_keywords = list(set(all_keywords))
            keyword_text = ' '.join(unique_keywords * 2)  # Ripeti 2 volte per peso
            
            # Testo entità (ripeti entità importanti)
            unique_entities = list(set(all_entities))
            entity_text = ' '.join(unique_entities * 3)  # Ripeti 3 volte per peso
            
            # Combina tutto
            combined_text = f"{base_text} {keyword_text} {entity_text}"
            
            # Memorizza
            self.campaign_texts[campaign] = combined_text
            self.campaign_tuples[campaign] = campaign_tuples
            self.campaign_keywords[campaign] = unique_keywords
            self.campaign_entities[campaign] = unique_entities
            all_campaign_texts.append(combined_text)
        
        # Costruisci matrice TF-IDF ottimizzata
        print("Costruendo matrice TF-IDF ultra-ottimizzata...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_campaign_texts)
        
        # Memorizza vettori TF-IDF per ogni campagna
        for i, campaign in enumerate(self.campaigns):
            self.campaign_tfidf_profiles[campaign] = self.tfidf_matrix[i]
        
        print(f"Profili ultra-ottimizzati creati per {len(self.campaigns)} campagne:")
        for campaign in self.campaigns:
            tuple_count = len(self.campaign_tuples[campaign])
            keyword_count = len(self.campaign_keywords[campaign])
            entity_count = len(self.campaign_entities[campaign])
            print(f"  {campaign}: {tuple_count} tuple, {keyword_count} keywords, {entity_count} entità")
    
    def _calculate_enhanced_tfidf_similarity(self, text):
        """Calcola similarità TF-IDF migliorata"""
        # Preprocessa il testo per matching migliore
        processed_text = text.lower()
        
        # Espandi abbreviazioni comuni
        expansions = {
            'pac': 'political action committee',
            'gop': 'republican party',
            'dem': 'democrat democratic',
            'potus': 'president united states',
            'scotus': 'supreme court'
        }
        
        for abbr, expansion in expansions.items():
            processed_text = re.sub(r'\b' + abbr + r'\b', expansion, processed_text)
        
        # Trasforma in vettore TF-IDF
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        
        # Calcola similarità con ogni campagna
        similarities = {}
        for campaign in self.campaigns:
            campaign_tfidf = self.campaign_tfidf_profiles[campaign]
            similarity = cosine_similarity(text_tfidf, campaign_tfidf)[0][0]
            similarities[campaign] = similarity
        
        return similarities
    
    def _calculate_enhanced_keyword_bonus(self, text):
        """Calcola bonus keywordsultra-migliorato"""
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w{2,}\b', text_lower))
        
        bonuses = {}
        for campaign in self.campaigns:
            campaign_keywords = set(self.campaign_keywords[campaign])
            campaign_entities = set(self.campaign_entities[campaign])
            
            # 1. Match esatti keywords
            exact_keyword_matches = len(text_words & campaign_keywords)
            
            # 2. Match esatti entità
            exact_entity_matches = len(text_words & campaign_entities)
            
            # 3. Match substring per keywords importanti
            substring_matches = 0
            for keyword in campaign_keywords:
                if len(keyword) > 3 and keyword in text_lower:
                    substring_matches += 1
            
            # 4. Match substring per entità
            entity_substring_matches = 0
            for entity in campaign_entities:
                if len(entity) > 2 and entity in text_lower:
                    entity_substring_matches += 2  # Peso doppio per entità
            
            # Calcola bonus pesato
            total_keywords = max(len(campaign_keywords), 1)
            total_entities = max(len(campaign_entities), 1)
            
            keyword_score = (exact_keyword_matches + substring_matches) / total_keywords
            entity_score = (exact_entity_matches + entity_substring_matches) / total_entities
            
            # Combina con pesi
            final_bonus = min((keyword_score * 0.6) + (entity_score * 0.4), 1.0)
            bonuses[campaign] = final_bonus
        
        return bonuses
    
    def classify_text(self, text):
        """Classifica un testo"""
        if not self.campaigns:
            raise ValueError("Nessuna campagna caricata")
        
        # Calcola tutti i tipi di similarità
        tfidf_similarities = self._calculate_enhanced_tfidf_similarity(text)
        
        # Embedding similarities (come backup)
        embedding_similarities = {}
        text_doc = nlp(text)
        text_vector = text_doc.vector
        
        if text_vector.any():
            for campaign in self.campaigns:
                campaign_text = self.campaign_texts[campaign][:500000]  # Limita per performance
                campaign_doc = nlp(campaign_text)
                
                if campaign_doc.vector.any():
                    similarity = cosine_similarity([text_vector], [campaign_doc.vector])[0][0]
                else:
                    similarity = 0.0
                
                embedding_similarities[campaign] = similarity
        else:
            embedding_similarities = {campaign: 0.0 for campaign in self.campaigns}
        
    
        keyword_bonuses = self._calculate_enhanced_keyword_bonus(text)
        
        # Special content bonuses
        special_bonuses = {}
        for campaign in self.campaigns:
            special_bonuses[campaign] = self._calculate_special_content_bonus(text, campaign)
        
        # Combina tutti i punteggi
        final_scores = {}
        for campaign in self.campaigns:
            tfidf_score = tfidf_similarities[campaign]
            embedding_score = embedding_similarities[campaign]
            keyword_bonus = keyword_bonuses[campaign]
            special_bonus = special_bonuses[campaign]
            
            # Score finale ultra-pesato
            final_score = (
                tfidf_score * self.tfidf_weight +
                embedding_score * self.embedding_weight +
                keyword_bonus * self.keyword_weight +
                special_bonus * self.special_bonus_weight
            )
            
            final_scores[campaign] = {
                'final_score': final_score,
                'tfidf_score': tfidf_score,
                'embedding_score': embedding_score,
                'keyword_bonus': keyword_bonus,
                'special_bonus': special_bonus
            }
        
        # Trova migliore classificazione
        best_campaign = max(final_scores.keys(), key=lambda x: final_scores[x]['final_score'])
        best_score = final_scores[best_campaign]['final_score']
        
        # Calcola confidence gap
        sorted_campaigns = sorted(final_scores.items(), 
                                key=lambda x: x[1]['final_score'], 
                                reverse=True)
        
        confidence_gap = 0
        if len(sorted_campaigns) > 1:
            confidence_gap = (sorted_campaigns[0][1]['final_score'] - 
                            sorted_campaigns[1][1]['final_score'])
        
        # Determina confidence level
        confidence_level = self._determine_confidence_level(best_score, confidence_gap)
        
        # Filtra classificazioni con confidence troppo bassa
        if best_score < self.min_confidence_threshold:
            return {
                'predicted_campaign': 'CONFIDENCE_TOO_LOW',
                'confidence': best_score,
                'confidence_level': 'Molto Bassa',
                'confidence_gap': confidence_gap,
                'classification_method': 'Ultra_Filtered',
                'details': final_scores[best_campaign],
                'all_scores': final_scores
            }
        
        return {
            'predicted_campaign': best_campaign,
            'confidence': best_score,
            'confidence_level': confidence_level,
            'confidence_gap': confidence_gap,
            'classification_method': 'Ultra_Optimized_Hybrid',
            'details': final_scores[best_campaign],
            'all_scores': final_scores,
            'sorted_predictions': sorted_campaigns[:3]
        }
    
    def _determine_confidence_level(self, confidence, confidence_gap):
        """Determina livello di confidenza ultra-calibrato"""
        if confidence < 0.15:
            return "Molto Bassa"
        elif confidence < 0.3:
            return "Bassa"
        elif confidence < 0.45:
            return "Media"
        elif confidence < 0.6 or confidence_gap < 0.08:
            return "Media-Alta"
        elif confidence < 0.75 or confidence_gap < 0.15:
            return "Alta"
        else:
            return "Molto Alta"
    
    def _identify_test_columns(self, test_df):
        """Identifica colonne nel dataset di test"""
        cols = [col.lower() for col in test_df.columns]
        
        # Trova colonne principali
        text_col = None
        text_candidates = ['text', 'testo', 'content', 'contenuto', 'body']
        for candidate in text_candidates:
            for i, col in enumerate(cols):
                if candidate in col:
                    text_col = test_df.columns[i]
                    break
            if text_col:
                break
        
        campaign_col = None
        campaign_candidates = ['campaign', 'campagna', 'category', 'label']
        for candidate in campaign_candidates:
            for i, col in enumerate(cols):
                if candidate in col:
                    campaign_col = test_df.columns[i]
                    break
            if campaign_col:
                break
        
        return {
            'text': text_col,
            'campaign': campaign_col,
            'id': test_df.columns[0] if len(test_df.columns) > 0 else None
        }
    
    def classify_dataset(self, test_file_path=None):
        print("\n" + "="*70)
        print("CLASSIFICAZIONE")
        print("="*70)
        
        # Carica dataset
        if not test_file_path:
            test_file_path = self._get_file_path("Seleziona il file di test")
        
        if not test_file_path or not os.path.exists(test_file_path):
            raise FileNotFoundError(f"File non trovato: {test_file_path}")
        
        try:
            if test_file_path.endswith('.xlsx'):
                test_df = pd.read_excel(test_file_path)
            elif test_file_path.endswith('.csv'):
                test_df = pd.read_csv(test_file_path)
            else:
                raise ValueError("Formato non supportato")
        except Exception as e:
            raise Exception(f"Errore caricamento: {e}")
        
        print(f"Dataset caricato: {len(test_df)} righe")
        
        # Identifica colonne
        columns = self._identify_test_columns(test_df)
        
        if not columns['text']:
            raise ValueError("Colonna TEXT non trovata")
        
        # Prepara risultati
        results_df = test_df.copy()
        results_df['PREDICTED_CAMPAIGN'] = ''
        results_df['PREDICTION_CONFIDENCE'] = 0.0
        results_df['CONFIDENCE_LEVEL'] = ''
        results_df['CONFIDENCE_GAP'] = 0.0
        results_df['CLASSIFICATION_METHOD'] = ''
        results_df['TFIDF_SCORE'] = 0.0
        results_df['EMBEDDING_SCORE'] = 0.0
        results_df['KEYWORD_BONUS'] = 0.0
        results_df['SPECIAL_BONUS'] = 0.0
        results_df['SECOND_BEST_CAMPAIGN'] = ''
        results_df['SECOND_BEST_SCORE'] = 0.0
        results_df['IS_TWITTER'] = False
        
        if columns['campaign']:
            results_df['CAMPAIGN_PREDICTION_CORRECT'] = False
            results_df['CAMPAIGN_MATCH'] = ''
        
        print(f"Inizio classificazione....attendi")
        
        successful = 0
        failed = 0
        correct = 0
        low_confidence = 0
        
        # Processa ogni riga
        for idx, row in test_df.iterrows():
            try:
                text = str(row[columns['text']]).strip()
                
                if len(text) < 10:
                    results_df.at[idx, 'PREDICTED_CAMPAIGN'] = 'TESTO_TROPPO_CORTO'
                    failed += 1
                    continue
                
                # Classifica
                result = self.classify_text(text)
                
                # Popola risultati
                results_df.at[idx, 'PREDICTED_CAMPAIGN'] = result['predicted_campaign']
                results_df.at[idx, 'PREDICTION_CONFIDENCE'] = result['confidence']
                results_df.at[idx, 'CONFIDENCE_LEVEL'] = result['confidence_level']
                results_df.at[idx, 'CONFIDENCE_GAP'] = result['confidence_gap']
                results_df.at[idx, 'CLASSIFICATION_METHOD'] = result['classification_method']
                results_df.at[idx, 'IS_TWITTER'] = self._is_twitter_content(text)
                
                # Dettagli ultra-ottimizzati
                if 'details' in result and result['details']:
                    details = result['details']
                    results_df.at[idx, 'TFIDF_SCORE'] = details.get('tfidf_score', 0.0)
                    results_df.at[idx, 'EMBEDDING_SCORE'] = details.get('embedding_score', 0.0)
                    results_df.at[idx, 'KEYWORD_BONUS'] = details.get('keyword_bonus', 0.0)
                    results_df.at[idx, 'SPECIAL_BONUS'] = details.get('special_bonus', 0.0)
                
                # Seconda migliore
                if 'sorted_predictions' in result and len(result['sorted_predictions']) > 1:
                    results_df.at[idx, 'SECOND_BEST_CAMPAIGN'] = result['sorted_predictions'][1][0]
                    results_df.at[idx, 'SECOND_BEST_SCORE'] = result['sorted_predictions'][1][1]['final_score']
                
                # Conta statistiche
                if result['predicted_campaign'] == 'CONFIDENCE_TOO_LOW':
                    low_confidence += 1
                
                # Verifica accuratezza
                if columns['campaign']:
                    true_campaign = str(row[columns['campaign']]).strip()
                    predicted = result['predicted_campaign']
                    
                    if predicted not in ['CONFIDENCE_TOO_LOW', 'NO_CAMPAIGNS_FOUND', 'TESTO_TROPPO_CORTO']:
                        is_correct = (true_campaign.lower() == predicted.lower())
                        results_df.at[idx, 'CAMPAIGN_PREDICTION_CORRECT'] = is_correct
                        results_df.at[idx, 'CAMPAIGN_MATCH'] = 'MATCH' if is_correct else 'NO_MATCH'
                        
                        if is_correct:
                            correct += 1
                    else:
                        results_df.at[idx, 'CAMPAIGN_PREDICTION_CORRECT'] = False
                        results_df.at[idx, 'CAMPAIGN_MATCH'] = 'FILTERED'
                
                successful += 1
                
                if (idx + 1) % 25 == 0:
                    print(f"Processate {idx + 1}/{len(test_df)} righe...")
                    
            except Exception as e:
                print(f"Errore riga {idx+1}: {e}")
                results_df.at[idx, 'PREDICTED_CAMPAIGN'] = 'ERRORE_PROCESSING'
                failed += 1
        
        # Statistiche finali ultra-dettagliate
        print(f"\n" + "="*50)
        print("RISULTATI CLASSIFICAZIONE :")
        print("="*50)
        print(f"Righe processate: {successful}")
        print(f"Errori: {failed}")
        print(f"Confidence troppo bassa: {low_confidence}")
        
        if columns['campaign'] and successful > 0:
            valid_classifications = successful - low_confidence
            if valid_classifications > 0:
                accuracy = (correct / valid_classifications) * 100
                print(f"Accuratezza (classificazioni valide): {accuracy:.2f}% ({correct}/{valid_classifications})")
            
            total_accuracy = (correct / successful) * 100
            print(f"Accuratezza totale: {total_accuracy:.2f}% ({correct}/{successful})")
        
        # Analisi dettagliata delle performance
        twitter_count = results_df['IS_TWITTER'].sum()
        twitter_correct = 0
        if columns['campaign']:
            twitter_df = results_df[results_df['IS_TWITTER'] == True]
            if len(twitter_df) > 0:
                twitter_correct = twitter_df['CAMPAIGN_PREDICTION_CORRECT'].sum()
                twitter_accuracy = (twitter_correct / len(twitter_df)) * 100
                print(f"Accuratezza contenuti Twitter: {twitter_accuracy:.2f}% ({twitter_correct}/{len(twitter_df)})")
        
        # Distribuzione per livello di confidenza
        confidence_dist = results_df['CONFIDENCE_LEVEL'].value_counts()
        print(f"\nDistribuzione livelli di confidenza:")
        for level, count in confidence_dist.items():
            print(f"  {level}: {count}")
        
        # Distribuzione predizioni
        dist = results_df['PREDICTED_CAMPAIGN'].value_counts()
        print(f"\nDistribuzione predizioni:")
        for campaign, count in dist.items():
            percentage = (count / len(results_df)) * 100
            print(f"  {campaign}: {count} ({percentage:.1f}%)")
        
        # Analisi score components medi
        valid_results = results_df[results_df['PREDICTION_CONFIDENCE'] > 0]
        if len(valid_results) > 0:
            print(f"\nScore medi (classificazioni valide):")
            print(f"  TF-IDF: {valid_results['TFIDF_SCORE'].mean():.3f}")
            print(f"  Embedding: {valid_results['EMBEDDING_SCORE'].mean():.3f}")
            print(f"  Keyword Bonus: {valid_results['KEYWORD_BONUS'].mean():.3f}")
            print(f"  Special Bonus: {valid_results['SPECIAL_BONUS'].mean():.3f}")
        
        # Salva risultati
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"risultati_ultra_ottimizzati_{timestamp}.xlsx"
        
        try:
            results_df.to_excel(output_file, index=False)
            print(f"\nRisultati salvati: {output_file}")
        except Exception as e:
            print(f"Errore salvataggio Excel: {e}")
            csv_file = f"risultati_ultra_ottimizzati_{timestamp}.csv"
            results_df.to_csv(csv_file, index=False)
            print(f"Salvato come CSV: {csv_file}")
        
        return results_df
    
    def analyze_classification_errors(self, results_df):
        """Analizza gli errori di classificazione per insights"""
        if 'CAMPAIGN_PREDICTION_CORRECT' not in results_df.columns:
            print("Nessuna colonna di verità disponibile per l'analisi errori")
            return
        
        print("\n" + "="*50)
        print("ANALISI ERRORI DI CLASSIFICAZIONE")
        print("="*50)
        
        # Errori per campagna vera
        errors_by_true_campaign = results_df[
            results_df['CAMPAIGN_PREDICTION_CORRECT'] == False
        ].groupby('CAMPAIGN').size().sort_values(ascending=False)
        
        print("Errori per campagna vera:")
        for campaign, count in errors_by_true_campaign.items():
            print(f"  {campaign}: {count} errori")
        
        # Errori per campagna predetta
        errors_by_predicted = results_df[
            results_df['CAMPAIGN_PREDICTION_CORRECT'] == False
        ].groupby('PREDICTED_CAMPAIGN').size().sort_values(ascending=False)
        
        print("\nErrori per campagna predetta:")
        for campaign, count in errors_by_predicted.items():
            print(f"  {campaign}: {count} errori")
        
        # Matrice di confusione semplificata (top 5 campagne)
        top_campaigns = results_df['CAMPAIGN'].value_counts().head(5).index
        confusion_data = []
        
        print(f"\nMatrice confusione (top 5 campagne):")
        print("Vera vs Predetta")
        
        for true_camp in top_campaigns:
            true_subset = results_df[results_df['CAMPAIGN'] == true_camp]
            pred_counts = true_subset['PREDICTED_CAMPAIGN'].value_counts()
            
            print(f"\n{true_camp}:")
            for pred_camp, count in pred_counts.head(3).items():
                accuracy = "✓" if true_camp.lower() == pred_camp.lower() else "✗"
                print(f"  → {pred_camp}: {count} {accuracy}")
    
    def generate_optimization_report(self, results_df):
        """Genera report di ottimizzazione dettagliato"""
        report = []
        report.append("="*80)
        report.append("REPORT CLASSIFICATORE")
        report.append("="*80)
        
        # Statistiche generali
        total = len(results_df)
        valid_predictions = len(results_df[
            ~results_df['PREDICTED_CAMPAIGN'].isin(['CONFIDENCE_TOO_LOW', 'ERRORE_PROCESSING', 'TESTO_TROPPO_CORTO'])
        ])
        
        report.append(f"\nSTATISTICHE GENERALI:")
        report.append(f"  Dataset totale: {total} esempi")
        report.append(f"  Predizioni valide: {valid_predictions} ({(valid_predictions/total)*100:.1f}%)")
        report.append(f"  Predizioni filtrate: {total - valid_predictions} ({((total-valid_predictions)/total)*100:.1f}%)")
        
        # Performance per metodo
        method_performance = results_df.groupby('CLASSIFICATION_METHOD').agg({
            'CAMPAIGN_PREDICTION_CORRECT': 'sum',
            'PREDICTION_CONFIDENCE': 'mean'
        })
        
        report.append(f"\nPERFORMANCE PER METODO:")
        for method, stats in method_performance.iterrows():
            method_total = len(results_df[results_df['CLASSIFICATION_METHOD'] == method])
            accuracy = (stats['CAMPAIGN_PREDICTION_CORRECT'] / method_total) * 100
            report.append(f"  {method}: {accuracy:.1f}% accuracy, {stats['PREDICTION_CONFIDENCE']:.3f} avg confidence")
        
        # Analisi componenti score
        valid_df = results_df[results_df['PREDICTION_CONFIDENCE'] > 0]
        if len(valid_df) > 0:
            report.append(f"\nCOMPONENTI SCORE (media ± std):")
            report.append(f"  TF-IDF: {valid_df['TFIDF_SCORE'].mean():.3f} ± {valid_df['TFIDF_SCORE'].std():.3f}")
            report.append(f"  Embedding: {valid_df['EMBEDDING_SCORE'].mean():.3f} ± {valid_df['EMBEDDING_SCORE'].std():.3f}")
            report.append(f"  Keyword Bonus: {valid_df['KEYWORD_BONUS'].mean():.3f} ± {valid_df['KEYWORD_BONUS'].std():.3f}")
            report.append(f"  Special Bonus: {valid_df['SPECIAL_BONUS'].mean():.3f} ± {valid_df['SPECIAL_BONUS'].std():.3f}")
        
        # Top performing campaigns
        if 'CAMPAIGN_PREDICTION_CORRECT' in results_df.columns:
            campaign_accuracy = results_df.groupby('CAMPAIGN').agg({
                'CAMPAIGN_PREDICTION_CORRECT': ['sum', 'count']
            }).round(3)
            
            campaign_accuracy.columns = ['correct', 'total']
            campaign_accuracy['accuracy'] = (campaign_accuracy['correct'] / campaign_accuracy['total']) * 100
            campaign_accuracy = campaign_accuracy.sort_values('accuracy', ascending=False)
            
            report.append(f"\nACCURATEZZA PER CAMPAGNA (top 10):")
            for campaign, stats in campaign_accuracy.head(10).iterrows():
                report.append(f"  {campaign}: {stats['accuracy']:.1f}% ({stats['correct']:.0f}/{stats['total']:.0f})")
        
        # Suggerimenti di ottimizzazione
        report.append(f"\nSUGGERIMENTI OTTIMIZZAZIONE:")
        
        low_confidence_count = len(results_df[results_df['PREDICTED_CAMPAIGN'] == 'CONFIDENCE_TOO_LOW'])
        if low_confidence_count > total * 0.2:
            report.append(f"  - Considera di abbassare la soglia minima (attuale: {self.min_confidence_threshold})")
        
        avg_tfidf = valid_df['TFIDF_SCORE'].mean() if len(valid_df) > 0 else 0
        if avg_tfidf < 0.1:
            report.append(f"  - Score TF-IDF bassi: considera di aumentare ngram_range o aggiungere sinonimi")
        
        twitter_accuracy = 0
        twitter_df = results_df[results_df['IS_TWITTER'] == True]
        if len(twitter_df) > 0 and 'CAMPAIGN_PREDICTION_CORRECT' in results_df.columns:
            twitter_accuracy = twitter_df['CAMPAIGN_PREDICTION_CORRECT'].mean() * 100
            if twitter_accuracy < 70:
                report.append(f"  - Accuratezza Twitter bassa ({twitter_accuracy:.1f}%): migliora pattern Twitter")
        
        return "\n".join(report)

def main():
    print("CLASSIFICATORE. Dimmi cosa vuoi classificare!")
    print("="*60)
    
    try:
        classifier = UltraOptimizedClassifier()
        
        print("\n1. Caricamento tuple e costruzione profili campagne...")
        classifier.load_opencti_tuples()
        
        print("\n2. Classificazione dataset...")
        results_df = classifier.classify_dataset()
        
        print("\n3. Analisi errori...")
        classifier.analyze_classification_errors(results_df)
        
        print("\n4. Generazione report ...")
        optimization_report = classifier.generate_optimization_report(results_df)
        print("\n" + optimization_report)
        
        # Salva report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(optimization_report)
        print(f"\nReport salvato: {report_file}")
        
        print("\nClassificazione completata!")
        
    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()