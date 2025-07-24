# -*- coding: utf-8 -*-
import torch
import huggingface_hub
import os
import time
import pandas as pd
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import tkinter as tk
from tkinter import filedialog

def setup_model():
    """Scarica e configura il modello LLM"""
    print("Configurazione del modello...")
    
    # Verifica versione Python
    import sys
    print(f"Python version: {sys.version}")
    
    # Verifica CUDA
    if torch.cuda.is_available():
        print(f"CUDA disponibile: {torch.cuda.get_device_name()}")
        print(f"Versione CUDA: {torch.version.cuda}")
    else:
        print("CUDA non disponibile, utilizzo CPU")
    
    model_name = "MaziyarPanahi/Llama-3-8B-Instruct-v0.10-GGUF"
    
    # Scegli il livello di quantizzazione (puoi cambiare qui)
    # Quantizzazione 1 (più veloce, meno precisa)
    model_file = "Llama-3-8B-Instruct-v0.10.Q4_K_M.gguf"
    
    # Quantizzazione 2 (bilanciata)
    # model_file = "Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf"
    
    # Quantizzazione 3 (più lenta, più precisa)
    # model_file = "Llama-3-8B-Instruct-v0.10.Q8_0.gguf"
    
    # Crea cartella models se non exists
    os.makedirs('models', exist_ok=True)
    
    # Download del modello
    print(f"Scaricamento del modello {model_file}...")
    model_path = hf_hub_download(
        model_name,
        filename=model_file,
        local_dir='models/',
        local_dir_use_symlinks=False
    )
    
    print(f"Modello scaricato in: {model_path}")
    return model_path

def load_excel_file():
    """Carica il file Excel usando una finestra di dialogo"""
    print("Seleziona il file Excel da elaborare...")
    
    # Crea una finestra root nascosta
    root = tk.Tk()
    root.withdraw()
    
    # Apri dialog per selezionare file [train.xlsx]
    file_path = filedialog.askopenfilename(
        title="Seleziona il file Excel",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    
    if not file_path:
        print("Nessun file selezionato.")
        return None
    
    print(f"File selezionato: {file_path}")
    
    # Carica il DataFrame
    df = pd.read_excel(file_path)
    print(f"File caricato con successo. Righe: {len(df)}")
    print("Prime 3 righe:")
    print(df.head(3))
    
    return df

def count_tokens_approx(text):
    """Stima approssimativa del numero di token (circa 4 caratteri per token)"""
    return len(text) // 4

def model_run(llm, user_input):
    """Esegue il modello con l'input dell'utente"""
    stop_sequence = ["##END LIST##", "#END LIST#", "END LIST"]
    
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Extract subject-verb-object tuples from text."},
            {"role": "user", "content": user_input}
        ],
        temperature=0,
        top_p=0.95,
        stop=stop_sequence,
        max_tokens=200  # Limita i token di output
    )
    
    return output['choices'][0]['message']['content']

def extract_all_tuples(text):
    """Estrae tutte le tuple nel formato 'soggetto - verbo - oggetto'"""
    pattern = r'([^\n]+? - [^\n]+? - [^\n]+?)\n'
    return re.findall(pattern, text)

def clean_tuples(tuples):
    """Pulisce e filtra le tuple estratte"""
    import re
    cleaned_set = set()
    final_tuples = []
    
    for t in tuples:
        # Rimuove spazi multipli
        t = re.sub(r'\s+', ' ', t.strip())
        
        # Verifica che la tupla sia nel formato corretto: 2 trattini
        if t.count(' - ') != 2:
            continue
            
        # Salta tuple con verbi generici o soggetti ambigui
        if any(word in t.lower() for word in ['is -', 'are -', 'was -', 'it -', 'there -']):
            continue
            
        # Normalizzazione
        t_lower = t.lower()
        
        # Rimuove tuple duplicate
        if t_lower not in cleaned_set:
            cleaned_set.add(t_lower)
            final_tuples.append(t)
    
    return final_tuples

def create_full_prompt(text):
    """Crea il prompt completo originale (ora possibile con 1024 token)"""
    prompt = f'''Instructions:
1. Focus only on the main factual actions or events.
2. Each tuple must contain:
   - A clear subject (the entity performing the action),
   - A precise verb (the action or relation),
   - A meaningful object (the entity affected or involved).
3. Include factual statements even if they appear within direct quotes, **as long as the speaker is identifiable or can be inferred from context**.
   - In such cases, use the speaker as the subject of the tuple.
   - For example: if the sentence is 'John said, "We support the initiative."', extract: John - said - we support the initiative.
4. Omit speculative, hypothetical, or unclear relationships.
5. Use only the following format, one per line:
   Subject - Verb - Object
6. Do not include explanations or paraphrasing.
7. Write '##END LIST##' after the last tuple.

Example:
Text: 'Senator Smith stated, "We must protect the environment."'
Output:
Senator Smith - stated - we must protect the environment

Example:
Text: 'John gave a book to Mary.'
Output:
John - gave - a book to Mary
##END LIST##

Now, extract tuples from this text: {text}'''
    return prompt

def process_articles(df, llm):
    """Processa tutti gli articoli ed estrae le tuple"""
    print("Inizio elaborazione degli articoli...")
    
    # Estrai le colonne necessarie
    colonna_id = df['ID']
    colonna_testo = df['TEXT']
    colonna_campagna = df['CAMPAIGN']
    
    print(f"Numero di articoli da elaborare: {len(colonna_testo)}")
    
    start_time = time.time()
    all_rows = []
    
    for i in range(len(colonna_testo)):
        print(f"Elaborazione articolo {i+1}/{len(colonna_testo)}")
        
        # Pulisci cache GPU se disponibile
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Riduci drasticamente il testo per rimanere nel context window
        testo_originale = str(colonna_testo[i])
        
        # Inizia con 50 parole e aumenta gradualmente se necessario
        max_words = 50
        
        while max_words <= 150:  # Massimo 150 parole
            testo_ridotto = ' '.join(testo_originale.split()[:max_words])
            
            # Crea prompt completo (ora possibile con 1024 token - 2048)
            user_input = create_full_prompt(testo_ridotto)
            
            # Controlla se il prompt è troppo lungo
            estimated_tokens = count_tokens_approx(user_input)
            print(f"Token stimati per articolo {i+1}: {estimated_tokens}")
            
            if estimated_tokens < 800:  # Lascia margine per la risposta (1024-800=224 token)
                break
            
            max_words -= 10  # Riduci ulteriormente se necessario
            
            if max_words < 20:
                print(f"Testo troppo lungo per articolo {i+1}, saltato")
                break
        
        if max_words < 20:
            continue
        
        try:
            response = model_run(llm, user_input)
            print(f"Risposta articolo {i+1}:")
            print(response)
            print("-" * 60)
            
            # Estrai tutte le tuple
            tuple_list = extract_all_tuples(response + '\n')  # Aggiungi \n per il pattern
            
            # Se non trova tuple con il pattern, prova a estrarre manualmente
            if not tuple_list:
                lines = response.split('\n')
                for line in lines:
                    if ' - ' in line and line.count(' - ') >= 2:
                        # Prendi solo i primi 3 elementi separati da ' - '
                        parts = line.split(' - ')
                        if len(parts) >= 3:
                            tuple_str = ' - '.join(parts[:3])
                            tuple_list.append(tuple_str)
            
            # Applica la pulizia delle tuple
            cleaned_tuples = clean_tuples(tuple_list)
            
            print(f"Tuple estratte prima della pulizia: {len(tuple_list)}")
            print(f"Tuple valide dopo la pulizia: {len(cleaned_tuples)}")
            
            # Crea una riga per ogni tupla pulita
            for tupla in cleaned_tuples:
                all_rows.append({
                    'TUPLA': tupla,
                    'ID ARTICOLO': colonna_id[i],
                    'CAMPAGNA': colonna_campagna[i]
                })
            
            if not cleaned_tuples:
                print(f"Nessuna tupla valida trovata nell'articolo {i+1}")
                # Aggiungi almeno una riga vuota per tracciare l'articolo processato
                all_rows.append({
                    'TUPLA': 'NO_TUPLES_FOUND',
                    'ID ARTICOLO': colonna_id[i],
                    'CAMPAGNA': colonna_campagna[i]
                })
                
        except Exception as e:
            print(f"Errore nell'elaborazione dell'articolo {i+1}: {e}")
            # Aggiungi riga di errore per tracciare
            all_rows.append({
                'TUPLA': f'ERROR: {str(e)}',
                'ID ARTICOLO': colonna_id[i],
                'CAMPAGNA': colonna_campagna[i]
            })
            continue
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Costruzione del DataFrame
    df_tuple = pd.DataFrame(all_rows)
    
    # Salvataggio su Excel
    output_filename = 'TuplesExtracted_PC_Version_Fixed.xlsx'
    df_tuple.to_excel(output_filename, index=False)
    
    # Statistiche finali
    valid_tuples = len(df_tuple[~df_tuple['TUPLA'].str.contains('ERROR|NO_TUPLES', na=False)])
    
    print(f"\nElaborazione completata!")
    print(f"Tempo di esecuzione: {runtime:.2f} secondi")
    print(f"Righe totali processate: {len(df_tuple)}")
    print(f"Tuple valide estratte: {valid_tuples}")
    print(f"File salvato come: {output_filename}")
    print("\nPrime 10 righe del risultato:")
    print(df_tuple.head(10))
    
    return df_tuple

def main():
    """Funzione principale"""
    print("=== ESTRAZIONE TUPLE DA ARTICOLI - VERSIONE CORRETTA ===")
    print("Versione PC - RISOLTO: Context window exceeded error")
    print("=" * 60)
    
    try:
        # 1. Setup del modello
        model_path = setup_model()
        
        # 2. Caricamento del file Excel
        df = load_excel_file()
        if df is None:
            return
        
        # 3. Inizializzazione del modello LLM con context window di 1024 token
        print("Caricamento del modello LLM...")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window di 1024 token -amplia a 2048
            verbose=False  # Riduci output verboso
        )
        print("Modello caricato con successo!")
        
        # 4. Elaborazione degli articoli
        df_tuple = process_articles(df, llm)
        
        print("\n" + "=" * 60)
        print("ELABORAZIONE COMPLETATA CON SUCCESSO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()