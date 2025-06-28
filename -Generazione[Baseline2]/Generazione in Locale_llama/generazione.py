# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import random
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Installazione delle librerie necessarie
# Esegui i seguenti comandi nel terminale se non hai già installato le librerie:
# pip uninstall numpy pandas -y
# pip install numpy==1.26.4 pandas==2.1.4
# pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122 --force-reinstall
# pip install torch==2.3.0 torchvision torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# pip install huggingface_hub openpyxl

# Definisci il percorso del file del modello
model_file = "Llama-3-8B-Instruct-v0.10.Q4_K_M.gguf"  # Assicurati che il file sia presente nella directory 'models/'
model_path = hf_hub_download(
    "MaziyarPanahi/Llama-3-8B-Instruct-v0.10-GGUF",
    filename=model_file,
    local_dir='models/'
)

# Caricamento del modello
llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window di 1024 token o 2048
            verbose=False  # Riduci output verboso
        )
# Funzione per caricare i dati
def load_data(file_path):
    return pd.read_excel(file_path)

data_path = 'test.xlsx'  # Assicurati che il file sia nella stessa directory
data = load_data(data_path)

# Conta quante volte ogni campagna compare nel file
campaign_counts = data['CAMPAIGN'].value_counts()

# Filtra solo le campagne che hanno almeno  2 articoli e meno di 150:le campagne del test set partono da 2 articoli e arrivano max a 10
filtered_campaigns = campaign_counts[(campaign_counts > 1) & (campaign_counts < 2000)].index

# Scegli una campagna casuale tra quelle disponibili nel file o selezionane una a mano
if len(filtered_campaigns) > 0:
    campaign = random.choice(filtered_campaigns)
else:
    print("Nessuna campagna con più di 1 occorrenze trovata:le campagne del test set partono da 2 articoli")

# Scegli una campagna specifica se necessario
# campaign = 'Viral Fake Election News'  # Esempio di campagna
print(f"Campagna scelta: {campaign}")

campaign_news = data[data['CAMPAIGN'] == campaign]
print(f"Numero di notizie nella campagna '{campaign}': {campaign_news.shape[0]}")

# Funzione per contare il numero di parole in un esempio
def count_words(example):
    return len(example.split())

# Funzione per formattare un articolo
def format_example(row):
    title = row['TITLE']
    if pd.isna(title) or str(title).strip().lower() == 'nan' or title == '':
        # Se il titolo è vuoto, usa l'handle Twitter o un estratto del testo come titolo
        if row['TYPE'] == 'TWITTER':
            # Prendi i primi 50 caratteri del testo come titolo
            text_excerpt = row['TEXT'][:50] + ("..." if len(row['TEXT']) > 50 else "")
            title = f"Tweet by {row['SOURCE']}: {text_excerpt}"
        else:
            title = "No Title"
    text = row['TEXT']
    if pd.isna(text) or str(text).strip().lower() == 'nan':
        text = ""
    return f"ARTICLE {row['ID']}\nTITLE: {title}\nTEXT: {text}\n"



# Funzione per creare le liste example e comparison
def create_example_and_comparison(news_df, max_words):
    example_list = []
    comparison_articles = []
    current_word_count = 0

    news_list = news_df.index.tolist()
    random.shuffle(news_list)  # Mischia gli articoli

    for idx in news_list:
        article = news_df.loc[idx]
        formatted_example = format_example(article)
        word_count = count_words(formatted_example)

        # Verifica se l'articolo può essere aggiunto alla example list
        if current_word_count + word_count <= max_words:
            example_list.append(formatted_example)
            current_word_count += word_count
        else:
            # Se non c'è più spazio, aggiungi l'articolo nella comparison list
            comparison_articles.append(formatted_example)

    return example_list, comparison_articles

# Crea le due liste
example_list, comparison_articles = create_example_and_comparison(campaign_news, 2000)

# Verifica del risultato
print(f"Numero di articoli nella example_list: {len(example_list)}")
print(f"Numero di articoli nella comparison_articles: {len(comparison_articles)}")

# Funzione per aggiungere una nuova riga ad un file Excel
def append_to_excel(df, file_name):
    try:
        # Controllo se il file esiste già
        with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Se il file esiste, appendo i dati senza riscrivere l'intestazione
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    except FileNotFoundError:
        # Se il file non esiste, lo creo e aggiungo i dati con l'intestazione
        df.to_excel(file_name, index=False)

# Pulizia della cache GPU
torch.cuda.empty_cache()

messages = [
    {"role": "system", "content": "You are a bot that writes fake news articles with TITLE and TEXT."},
    {"role": "user", "content": f"Write ONLY TWO realistic fake news articles similar content of the examples I will give you.\nAnswer ONLY filling these fields:\n-TITLE: '...'\n-TEXT: '...' and EOF at the end of response.\n\nHere are the examples to follow:\n"
     + "\n".join(example_list)}
]

# Stampa del prompt inviato al modello
print("Prompt inviato al modello:")
for message in messages:
    print(f"Ruolo: {message['role']}, Contenuto: {message['content']}")

output = llm.create_chat_completion(
    messages=messages,
    temperature=0.7,
    top_k=50,
    top_p=0.85,
    max_tokens=1024  # più output
)


# Funzione per rimuovere tag indesiderati come ù
# Funzione per rimuovere tag indesiderati come </s>, [INST], ecc.
def clean_response(response):
    # Usa una regex per rimuovere tutti i tag come </s>, <s>, [INST], ecc.
    cleaned_response = re.sub(r"</?s>|</?inst>|[<>]|(?:/system|/inst|/sys)+|\[\]/sys(?:!?\[\]/sys)*", "", response, flags=re.IGNORECASE)
    return cleaned_response.strip()

# Risposta del modello
content = output['choices'][0]['message']['content']
cleaned_content = clean_response(content)
print(f"_________________________________________________________________________________________________________________________________\n{cleaned_content}")

#------------------------Salvataggio articolo generato----------------------------
file_name = 'Generated_fake_news.xlsx'

# Controlla se il file Excel esiste
if os.path.exists(file_name):
    # Leggi il file Excel
    existing_df = pd.read_excel(file_name)

    if 'Id' in existing_df.columns and not existing_df.empty:
        # Trova l'ultimo ID nel file Excel
        last_id = existing_df['Id'].max()
        id_counter = last_id + 1
    else:
        id_counter = 1  # Se non ci sono articoli, inizia da 1
else:
    id_counter = 1  # Se il file non esiste, inizia da 1

generated_news = {
    'Id': id_counter,
    'Context': ["\n".join([f"Ruolo: {message['role']}, Contenuto: {message['content']}" for message in messages])],
    'Generated': [cleaned_content],
    'Campaign': [campaign]
}
id_counter += 1

# Creazione data frame e salvataggio in file excel
new_df = pd.DataFrame(generated_news)

# Appendi l'articolo generato a un excel esistente o salvalo in un nuovo file da solo
append_to_excel(new_df, file_name)

print(f"Dati salvati nel file {file_name}")

# Crea un DataFrame con due colonne: 'Example Articles' e 'Comparison Articles'
df = pd.DataFrame({
    'Example Articles': pd.Series(example_list),
    'Comparison Articles': pd.Series(comparison_articles)
})

# Salva il DataFrame in un file Excel
df.to_excel('Articles_Lists.xlsx', index=False)

print("File 'Articles_Lists.xlsx' creato con successo.")
