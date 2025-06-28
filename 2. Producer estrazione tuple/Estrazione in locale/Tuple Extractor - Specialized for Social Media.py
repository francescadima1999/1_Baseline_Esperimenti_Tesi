import os
import sys
import torch
import pandas as pd
import re
import time
import signal
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# =============================================================================
# TIMEOUT HANDLER
# =============================================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Model execution timeout")

# =============================================================================
# CONFIGURAZIONE INIZIALE
# =============================================================================

def setup_environment():
    """Configura l'ambiente e verifica le dipendenze"""
    print(f"Python version: {sys.version}")
    
    # Verifica se CUDA è disponibile
    if torch.cuda.is_available():
        print(f"CUDA disponibile: {torch.cuda.get_device_name()}")
    else:
        print("CUDA non disponibile - utilizzerà CPU")
    
    return torch.cuda.is_available()

def download_model(model_name="MaziyarPanahi/Llama-3-8B-Instruct-v0.10-GGUF"):
    """Scarica il modello da Hugging Face"""
    
    print("Scegli la quantizzazione del modello:")
    print("1. Q4_K_M (4-bit) - Minima memoria, velocissima")
    print("2. Q5_K_M (5-bit) - Compromesso tra efficienza e qualità") 
    print("3. Q8_0 (8-bit) - Massima accuratezza, più pesante")
    
    choice = input("Inserisci la tua scelta (1-3, default 1 per CPU): ").strip()
    
    if choice == "2":
        model_file = "Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf"
    elif choice == "3":
        model_file = "Llama-3-8B-Instruct-v0.10.Q8_0.gguf"
    else:
        model_file = "Llama-3-8B-Instruct-v0.10.Q4_K_M.gguf"
    
    print(f"Scaricando modello: {model_file}")
    
    # Crea directory models se non esiste
    os.makedirs('models', exist_ok=True)
    
    model_path = hf_hub_download(
        model_name,
        filename=model_file,
        local_dir='models/',
        local_dir_use_symlinks=False
    )
    
    print(f"Modello scaricato in: {model_path}")
    return model_path

# =============================================================================
# CARICAMENTO DATI
# =============================================================================

def load_data_from_excel(file_path):
    """Carica i dati dal file Excel con gestione migliorata dei NaN"""
    try:
        df = pd.read_excel(file_path)
        print(f"File caricato: {file_path}")
        print(f"Righe: {len(df)}, Colonne: {len(df.columns)}")
        
        # Mostra informazioni dettagliate sulle colonne
        print("\nInformazioni sulle colonne:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"  {col}: {non_null_count}/{len(df)} valori non nulli")
        
        print("\nPrime 3 righe:")
        print(df.head(3))
        
        # Gestisci i valori NaN nella colonna TEXT
        if 'TEXT' in df.columns:
            # Conta quanti testi sono validi
            valid_texts = df['TEXT'].notna().sum()
            print(f"\nTesti validi nella colonna TEXT: {valid_texts}/{len(df)}")
            
            # Mostra esempi di testi se disponibili
            valid_text_sample = df[df['TEXT'].notna()]['TEXT'].head(2)
            if not valid_text_sample.empty:
                print("\nEsempi di testi trovati:")
                for idx, text in valid_text_sample.items():
                    print(f"  Riga {idx}: {str(text)[:100]}...")
        else:
            print("\nATTENZIONE: Colonna 'TEXT' non trovata!")
            print("Colonne disponibili:", list(df.columns))
        
        return df
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return None

# =============================================================================
# PREPROCESSING TWEET MIGLIORATO
# =============================================================================

def preprocess_tweet_advanced(text):
    """Preprocessa il tweet mantenendo il contesto semantico"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    original_text = text
    
    # Sostituzioni semanticamente consapevoli
    # Gestisci RT preservando il contenuto originale
    text = re.sub(r'RT @(\w+):\s*', r'[User \1 shared]: ', text)
    
    # Gestisci le URL mantenendo il contesto
    url_pattern = r'https?://\S+'
    if re.search(url_pattern, text):
        text = re.sub(url_pattern, '[LINK]', text)
    
    # Gestisci hashtag mantenendo il significato
    text = re.sub(r'#(\w+)', r'topic:\1', text)
    
    # Gestisci le menzioni preservando l'informazione
    text = re.sub(r'@(\w+)', r'user:\1', text)
    
    # Pulisci caratteri di controllo ma mantieni punteggiatura importante
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# =============================================================================
# FUNZIONI DI ESTRAZIONE MIGLIORATE
# =============================================================================

def initialize_model(model_path):
    """Inizializza il modello LLM con parametri ottimizzati per CPU"""
    try:
        print("Inizializzazione modello con parametri ottimizzati...")
        
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Ridotto per CPU
            n_threads=4,  # Limita i thread
            n_batch=128,  # Batch più piccolo
            n_gpu_layers=0,  # Forza CPU
            verbose=False,
            use_mmap=True,  # Usa memory mapping
            use_mlock=False,  # Non bloccare memoria
            f16_kv=True,  # Usa float16 per key-value cache
        )
        
        print("Modello inizializzato con successo")
        return llm
    except Exception as e:
        print(f"Errore nell'inizializzazione del modello: {e}")
        return None

def model_run_with_timeout(llm, user_input, timeout=90):
    """Esegue il modello con timeout aumentato per processing più complesso"""
    
    def run_model():
        try:
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert at extracting semantic relationships from social media content. Focus on meaningful, factual relationships."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.2,  # Leggermente più alta per creatività controllata
                top_p=0.85,       # Più focalizzato
                max_tokens=400,   # Aumentato per risposte più ricche
                stop=["##ANALYSIS_END##", "END_EXTRACTION"],
                stream=False
            )
            return output['choices'][0]['message']['content']
        except Exception as e:
            print(f"Errore interno modello: {e}")
            return f"ERRORE_MODELLO: {str(e)}"
    
    # Prova l'esecuzione con timeout (solo su Unix-like systems)
    try:
        if hasattr(signal, 'SIGALRM'):
            # Sistema Unix-like
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                result = run_model()
                signal.alarm(0)  # Cancella l'allarme
                return result
            except TimeoutException:
                print(f"TIMEOUT: Il modello ha impiegato più di {timeout} secondi")
                return "TIMEOUT_ERROR"
        else:
            # Sistema Windows - esegui senza timeout
            print("Sistema Windows: esecuzione senza timeout")
            return run_model()
            
    except Exception as e:
        print(f"Errore nell'esecuzione con timeout: {e}")
        return f"ERRORE_TIMEOUT: {str(e)}"

def create_advanced_extraction_prompt(text):
    """Crea un prompt semanticamente più sofisticato"""
    preprocessed_text = preprocess_tweet_advanced(text)
    
    if not preprocessed_text or len(preprocessed_text.strip()) < 5:
        return None
    
    # Limita la lunghezza mantenendo frasi complete
    words = preprocessed_text.split()
    if len(words) > 60:
        # Prova a tagliare a fine frase
        truncated = ' '.join(words[:60])
        last_punct = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_punct > len(truncated) * 0.7:  # Se c'è punteggiatura nella parte finale
            preprocessed_text = truncated[:last_punct + 1]
        else:
            preprocessed_text = truncated + "..."
    
    return f'''Extract meaningful semantic relationships from this social media post. Focus on concrete facts and actions, not meta-commentary about the post itself.

CONTENT: {preprocessed_text}

INSTRUCTIONS:
1. Identify WHO does WHAT to/with WHOM/WHAT
2. Focus on real-world entities and actions, not social media mechanics
3. Use specific names when available, avoid generic terms like "user" or "tweet author"
4. Each relationship should be factual and meaningful
5. Format: Subject - Action - Object
6. Avoid relationships about posting, tweeting, or sharing unless that's the main content

GOOD EXAMPLES:
- "Trump - won - election"
- "Hillary - sold - uranium"
- "Obama - interrupted - vacation"
- "voting machines - switched - votes"

AVOID:
- "User - posted - tweet"
- "Author - mentioned - topic"
- "Tweet - contains - hashtag"

Extract only the most important relationships:

##RELATIONSHIPS##
'''

def extract_semantic_tuples(text):
    """Estrae tuple con focus semantico migliorato"""
    if "ERRORE" in text or "TIMEOUT" in text:
        return []
    
    # Cerca la sezione delle relazioni
    relationships_section = ""
    if "##RELATIONSHIPS##" in text:
        relationships_section = text.split("##RELATIONSHIPS##")[1]
    else:
        relationships_section = text
    
    # Pattern più flessibile per catturare relazioni semantiche
    patterns = [
        r'(?:^|\n)\s*([^-\n]+?)\s+-\s+([^-\n]+?)\s+-\s+([^\n]+?)(?:\n|$)',  # Standard format
        r'(?:^|\n)\s*-\s*([^-\n]+?)\s+-\s+([^-\n]+?)\s+-\s+([^\n]+?)(?:\n|$)',  # With bullet
        r'(?:^|\n)\s*\d+\.\s*([^-\n]+?)\s+-\s+([^-\n]+?)\s+-\s+([^\n]+?)(?:\n|$)',  # Numbered
    ]
    
    tuples = []
    for pattern in patterns:
        matches = re.findall(pattern, relationships_section, re.MULTILINE)
        for match in matches:
            subject = match[0].strip()
            verb = match[1].strip()
            obj = match[2].strip()
            
            # Validazione semantica
            if is_valid_semantic_tuple(subject, verb, obj):
                tuples.append(f"{subject} - {verb} - {obj}")
    
    return tuples

def is_valid_semantic_tuple(subject, verb, obj):
    """Valida la qualità semantica di una tupla"""
    # Lunghezza minima
    if len(subject) < 2 or len(verb) < 2 or len(obj) < 2:
        return False
    
    # Evita tuple troppo generiche o meta-linguistiche
    generic_subjects = ['user', 'author', 'tweet', 'post', 'content', 'text', 'message']
    generic_verbs = ['posted', 'tweeted', 'wrote', 'said', 'mentioned', 'contains']
    
    subject_lower = subject.lower()
    verb_lower = verb.lower()
    
    # Evita soggetti troppo generici
    if any(generic in subject_lower for generic in generic_subjects):
        # Permetti solo se il verbo non è generico
        if any(generic_verb in verb_lower for generic_verb in generic_verbs):
            return False
    
    # Evita verbi di collegamento vuoti
    linking_verbs = ['is', 'are', 'was', 'were', 'be', 'been']
    if verb_lower.strip() in linking_verbs and len(obj.split()) < 2:
        return False
    
    # Controlla che non sia solo punteggiatura
    if not re.search(r'[a-zA-Z]', subject) or not re.search(r'[a-zA-Z]', verb) or not re.search(r'[a-zA-Z]', obj):
        return False
    
    return True

def clean_semantic_tuples(tuple_list):
    """Pulisce le tuple con focus sulla qualità semantica"""
    cleaned_tuples = []
    seen_semantics = set()
    
    for tupla in tuple_list:
        # Pulizia base
        tupla = re.sub(r'^[^\w]+|[^\w]+$', '', tupla)
        tupla = re.sub(r'\s+', ' ', tupla.strip())
        
        if tupla.count(' - ') != 2:
            continue
        
        parts = tupla.split(' - ')
        if len(parts) != 3:
            continue
            
        subject, verb, obj = parts
        
        # Normalizza per rilevare duplicati semantici
        semantic_key = f"{subject.lower().strip()} {verb.lower().strip()} {obj.lower().strip()}"
        
        if semantic_key not in seen_semantics:
            seen_semantics.add(semantic_key)
            
            # Capitalizza correttamente
            subject = capitalize_entity(subject.strip())
            verb = verb.strip().lower()
            obj = capitalize_entity(obj.strip())
            
            cleaned_tupla = f"{subject} - {verb} - {obj}"
            cleaned_tuples.append(cleaned_tupla)
    
    return cleaned_tuples

def capitalize_entity(text):
    """Capitalizza entità in modo intelligente"""
    # Nomi propri comuni
    proper_nouns = ['trump', 'clinton', 'hillary', 'obama', 'texas', 'milwaukee', 'america', 'usa']
    
    words = text.split()
    result = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in proper_nouns or (len(word) > 1 and word.isupper()):
            result.append(word.capitalize())
        elif word_lower in ['url', 'link']:
            result.append(word.upper())
        else:
            result.append(word)
    
    return ' '.join(result)

# =============================================================================
# FUNZIONE PRINCIPALE DI ESTRAZIONE MIGLIORATA
# =============================================================================

def extract_tuples_from_tweets_advanced(df, llm, output_filename="TwitterTuplesExtracted_Advanced.xlsx"):
    """Estrae tuple da tutti i tweet con processing semantico migliorato"""
    
    all_rows = []
    total_texts = len(df)
    
    print(f"Inizio estrazione semantica avanzata da {total_texts} tweet...")
    
    if 'TEXT' not in df.columns:
        print("ERRORE: Colonna 'TEXT' non trovata nel DataFrame!")
        return pd.DataFrame()
    
    for i, row in df.iterrows():
        print(f"\n--- Processando tweet {i+1}/{total_texts} ---")
        
        # Gestione sicura dei valori
        text = row.get('TEXT', '')
        if pd.isna(text) or text is None:
            text = ""
        else:
            text = str(text)
            
        article_id = row.get('ID', i)
        campaign = str(row.get('CAMPAIGN', 'N/A'))
        
        print(f"Tweet ID: {article_id}")
        print(f"Testo lungo: {len(text)} caratteri")
        
        if len(text) > 0:
            print(f"Anteprima: {text[:100]}...")
        
        if not text or len(text.strip()) < 15:  # Soglia più alta per qualità
            all_rows.append({
                'TUPLA': 'TESTO_TROPPO_CORTO',
                'ID ARTICOLO': article_id,
                'CAMPAGNA': campaign,
                'STATO': 'testo_insufficiente',
                'QUALITA': 'bassa'
            })
            print("→ Testo insufficiente")
            continue
        
        # Crea prompt avanzato
        user_input = create_advanced_extraction_prompt(text)
        
        if user_input is None:
            all_rows.append({
                'TUPLA': 'PREPROCESSING_FALLITO',
                'ID ARTICOLO': article_id,
                'CAMPAGNA': campaign,
                'STATO': 'preprocessing_error',
                'QUALITA': 'errore'
            })
            print("→ Preprocessing fallito")
            continue
        
        try:
            print("→ Invio al modello per analisi semantica (timeout 90s)...")
            start_time = time.time()
            
            response = model_run_with_timeout(llm, user_input, timeout=90)
            
            end_time = time.time()
            print(f"→ Risposta ricevuta in {end_time - start_time:.1f}s")
            print(f"→ Lunghezza risposta: {len(response)}")
            
            # Gestisci errori del modello
            if "ERRORE" in response or "TIMEOUT" in response:
                all_rows.append({
                    'TUPLA': response,
                    'ID ARTICOLO': article_id,
                    'CAMPAGNA': campaign,
                    'STATO': 'errore_modello',
                    'QUALITA': 'errore'
                })
                print(f"→ Errore modello: {response}")
                continue
            
            # Mostra un po' della risposta per debug
            print(f"→ Anteprima risposta: {response[:150]}...")
            
            # Estrai tuple con algoritmo migliorato
            tuple_list = extract_semantic_tuples(response)
            tuple_list = clean_semantic_tuples(tuple_list)
            
            print(f"→ Tuple semantiche estratte: {len(tuple_list)}")
            
            if tuple_list:
                for tupla in tuple_list:
                    # Valuta la qualità della tupla
                    parts = tupla.split(' - ')
                    if len(parts) == 3:
                        quality = evaluate_tuple_quality(parts[0], parts[1], parts[2])
                    else:
                        quality = 'bassa'
                    
                    all_rows.append({
                        'TUPLA': tupla,
                        'ID ARTICOLO': article_id,
                        'CAMPAGNA': campaign,
                        'STATO': 'successo',
                        'QUALITA': quality
                    })
                    print(f"  • {tupla} [{quality}]")
            else:
                all_rows.append({
                    'TUPLA': 'NESSUNA_TUPLA_SEMANTICA_TROVATA',
                    'ID ARTICOLO': article_id,
                    'CAMPAGNA': campaign,
                    'STATO': 'nessuna_tupla',
                    'QUALITA': 'vuota'
                })
                print("→ Nessuna tupla semantica estratta")
                
        except Exception as e:
            print(f"→ ERRORE durante elaborazione: {e}")
            all_rows.append({
                'TUPLA': f'ERRORE_GENERALE: {str(e)}',
                'ID ARTICOLO': article_id,
                'CAMPAGNA': campaign,
                'STATO': 'errore_exception',
                'QUALITA': 'errore'
            })
        
        # Pausa tra i tweet
        time.sleep(1.5)
    
    # Crea DataFrame risultati
    df_tuple = pd.DataFrame(all_rows)
    
    # Statistiche dettagliate
    if 'STATO' in df_tuple.columns:
        print("\n" + "="*60)
        print("STATISTICHE ESTRAZIONE SEMANTICA AVANZATA:")
        print("="*60)
        stato_stats = df_tuple['STATO'].value_counts()
        for stato, count in stato_stats.items():
            print(f"{stato}: {count} casi")
        
        if 'QUALITA' in df_tuple.columns:
            print("\nDistribuzione qualità tuple:")
            qualita_stats = df_tuple['QUALITA'].value_counts()
            for qualita, count in qualita_stats.items():
                print(f"  {qualita}: {count} tuple")
    
    # Salva risultati
    try:
        df_tuple.to_excel(output_filename, index=False)
        print(f"\nRisultati salvati in: {output_filename}")
    except Exception as e:
        print(f"Errore nel salvare Excel: {e}")
        # Salva come CSV in caso di errore
        csv_filename = output_filename.replace('.xlsx', '.csv')
        df_tuple.to_csv(csv_filename, index=False)
        print(f"Salvato come CSV: {csv_filename}")
    
    return df_tuple

def evaluate_tuple_quality(subject, verb, obj):
    """Valuta la qualità semantica di una tupla"""
    score = 0
    
    # Lunghezza appropriata
    if 2 <= len(subject.split()) <= 4:
        score += 1
    if 1 <= len(verb.split()) <= 2:
        score += 1
    if 2 <= len(obj.split()) <= 5:
        score += 1
    
    # Specificità (evita termini troppo generici)
    if subject.lower() not in ['user', 'author', 'person', 'someone']:
        score += 1
    if verb.lower() not in ['is', 'are', 'was', 'were']:
        score += 1
    
    # Presenza di entità riconoscibili
    entities = ['trump', 'clinton', 'hillary', 'obama', 'texas', 'america']
    text_lower = f"{subject} {obj}".lower()
    if any(entity in text_lower for entity in entities):
        score += 1
    
    if score >= 5:
        return 'alta'
    elif score >= 3:
        return 'media'
    else:
        return 'bassa'

# =============================================================================
# MAIN MIGLIORATO
# =============================================================================

def main():
    """Funzione principale migliorata"""
    
    print("=== TWITTER SEMANTIC TUPLE EXTRACTOR (VERSIONE AVANZATA) ===")
    
    # 1. Setup ambiente
    print("\n=== SETUP AMBIENTE ===")
    cuda_available = setup_environment()
    
    # 2. Verifica modello
    print("\n=== VERIFICA MODELLO ===")
    model_path = "models/Llama-3-8B-Instruct-v0.10.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        # Prova altri file
        possible_files = [
            "models/Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf",
            "models/Llama-3-8B-Instruct-v0.10.Q8_0.gguf"
        ]
        
        found = False
        for path in possible_files:
            if os.path.exists(path):
                model_path = path
                found = True
                break
        
        if not found:
            print("Nessun modello trovato, avvio download...")
            model_path = download_model()
    
    print(f"Usando modello: {model_path}")
    
    # 3. Carica dati
    print("\n=== CARICAMENTO DATI ===")
    excel_file = input("Inserisci il path del file Excel: ").strip()
    if not excel_file:
        excel_file = "data.xlsx"
    
    df = load_data_from_excel(excel_file)
    if df is None:
        return
    
    # Verifica testi validi
    if 'TEXT' not in df.columns or df['TEXT'].notna().sum() == 0:
        print("ERRORE: Nessun testo valido trovato!")
        return
    
    # 4. Inizializza modello
    print("\n=== INIZIALIZZAZIONE MODELLO ===")
    llm = initialize_model(model_path)
    if llm is None:
        return
    
    print("NOTA: Usando algoritmi di estrazione semantica avanzati")
    print("L'elaborazione sarà più lenta ma produrrà tuple di qualità superiore")
    
    # 5. Estrazione avanzata
    print("\n=== ESTRAZIONE TUPLE SEMANTICHE ===")
    output_file = f"TwitterTuples_Semantic_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx"
    
    df_results = extract_tuples_from_tweets_advanced(df, llm, output_file)
    
    print(f"\n=== COMPLETATO ===")
    print(f"Tweet elaborati: {len(df)}")
    print(f"Risultati totali: {len(df_results)}")
    print(f"File salvato: {output_file}")
    
    # Statistiche finali di qualità
    if 'QUALITA' in df_results.columns:
        alta_qualita = len(df_results[df_results['QUALITA'] == 'alta'])
        media_qualita = len(df_results[df_results['QUALITA'] == 'media'])
        print(f"Tuple di alta qualità: {alta_qualita}")
        print(f"Tuple di media qualità: {media_qualita}")

if __name__ == "__main__":
    main()