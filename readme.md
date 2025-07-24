## Struttura del Repository

```
├── -Alternativa DeepSeek estrazionetuple/    # Implementazione alternativa con DeepSeek
├── -Generazione[Baseline2]/                  # Seconda versione del baseline
├── 1. Train e Test set/                      # Dataset per training e testing
├── 2. Producer estrazione tuple/             # Modulo per l'estrazione di tuple
├── 3. Rappresentazione in OPENCTI/producer/  # Integrazione con OpenCTI
├── 4.Consumer Sincronizzazione/              # Sistema di sincronizzazione
├── 5 Consumer riceve tuple e fake news e valuta/ # Valutazione di fake news
└── FAKEBERT /              # Esperimenti con FAKEBERT
```

## Prerequisiti

Prima di iniziare, assicurati di avere installato:

- Python 3.8 o superiore
- pip (gestore pacchetti Python)
- Git
Poi installa le dipendenze

pip install torch transformers pandas numpy scikit-learn matplotlib seaborn spacy
python -m spacy download en_core_web_md

## Utilizzo

### 1. Train e Test Set

Prima di tutto, devi preparare i dataset di train e test:

```bash
cd "1. Train e Test set"

# Dividi il dataset originale in train e test
python split.py

# Genera grafici e statistiche del dataset
python statistichedataset.py
```

**File richiesti:**
- `FakeCTI Dataset.xlsx` - Il dataset originale

**Output generati:**
- `train.xlsx` - Set di training (90% per campagne ≤101 articoli, 90 articoli per campagne >101)
- `test.xlsx` - Set di test (10% per campagne ≤101 articoli, 10 articoli per campagne >101)
- Grafici PNG: `top_campaigns_scaled.png`, `threat_actors_bar.png`, `type_distribution.png`, `sources_distribution.png`
- `fakecti_summary.csv` - Tabella riassuntiva delle statistiche

### 2. Producer Estrazione Tuple

Questo step estrae le tuple (soggetto-verbo-oggetto) dai testi usando il modello LLaMA. Sono disponibili diverse versioni:

**Opzione A: Esecuzione in Google Colab (Consigliata)**
```bash
cd "2. Producer estrazione tuple"
# Carica e esegui il notebook Jupyter:
jupyter notebook Estrazione_tuple_tesi_.ipynb
```

**Opzione B: Esecuzione Locale**
```bash
cd "2. Producer estrazione tuple"
python "Estrazione_tuple_in_locale.py"
```

**Opzione C: Versione Specializzata per Social Media**
```bash
cd "2. Producer estrazione tuple"
python "Tuple Extractor - Specialized for Social Media.py"
```

**Configurazione del Modello:**
- **Q4_K_M (4-bit)**: Veloce, minor precisione, minima memoria
- **Q5_K_M (5-bit)**: Bilanciato tra velocità e qualità
- **Q8_0 (8-bit)**: Massima precisione, più lento e pesante

**Input richiesti:**
- File Excel con dati train (train.xlsx) 
- Il modello LLaMA viene scaricato automaticamente da Hugging Face

**Output generati:**
- `TuplesExtracted_*.xlsx` - File Excel con le tuple estratte nel formato:
  - TUPLA: "soggetto - verbo - oggetto"
  - ID ARTICOLO: identificativo dell'articolo originale
  - CAMPAGNA: campagna di appartenenza
  - STATO/QUALITA: informazioni sulla qualità dell'estrazione

#### Implementazione Alternativa con DeepSeek

**Opzione D: Estrazione con DeepSeek (Alternativa)**
```bash
cd "-Alternativa DeepSeek estrazionetuple"
# Carica e esegui il notebook Jupyter:
jupyter notebook DeepSeek_completo.ipynb
```
Questa implementazione alternativa utilizza il modello DeepSeek per l'estrazione delle tuple, offrendo:
- Diverso approccio nell'elaborazione del linguaggio naturale
- Possibilità di confronto delle performance tra modelli


### 3. Rappresentazione in OpenCTI/Producer

Questo step carica le tuple estratte in una piattaforma OpenCTI per la rappresentazione semantica e l'analisi della disinformazione.

**Prerequisiti:**
- Docker e Docker Compose installati
- File `.env` con le variabili d'ambiente (vedi sezione Configurazione)

**Avvio dell'ambiente OpenCTI:**
```bash
cd "3. Rappresentazione in OPENCTI/producer"

# Avvia tutti i servizi con Docker Compose
docker-compose up -d

```

**Accesso all'interfaccia OpenCTI:**
- URL: http://localhost:8081
- Email: valore da `OPENCTI_ADMIN_EMAIL` nel file .env
- Password: valore da `OPENCTI_ADMIN_PASSWORD` nel file .env

**Caricamento delle tuple:**
```bash
# Assicurati che il file "tuple estratte.xlsx" sia nella directory
python producer_import.py
```

**Servizi inclusi nell'ambiente Docker:**
- **OpenCTI Platform**: Interfaccia principale (porta 8081)
- **Redis**: Cache e sessioni
- **Elasticsearch**: Indicizzazione e ricerca
- **MinIO**: Storage oggetti
- **RabbitMQ**: Sistema di messaggistica (porta 15674 per management)
- **Worker**: Processamento background

**Struttura dati creata in OpenCTI:**
- **Identity**: Soggetti e oggetti delle tuple
- **Attack-Pattern**: Verbi/azioni delle tuple
- **Indicator**: Rappresentazione completa della tripla semantica
- **Campaign**: Raggruppamento per campagna di disinformazione
- **Relationships**: Collegamenti semantici tra entità

**File richiesti:**
- `docker-compose.yml` - Configurazione dei servizi
- `producer_import.py` - Script di importazione delle tuple
- `tuple estratte.xlsx` - File Excel con le tuple da caricare

### 5. Consumer Riceve Tuple e Fake News e Valuta

Questo step implementa un sistema di classificazione avanzato che valuta i contenuti di disinformazione utilizzando le tuple sincronizzate dall'istanza Consumer.

**Prerequisiti:**
- spaCy con modello inglese: `python -m spacy download en_core_web_md`
- Scikit-learn, pandas, numpy
- File tuple sincronizzate dal punto 4
- Dataset di test da classificare

**Installazione dipendenze:**
```bash
cd "5 Consumer riceve tuple e fake news e valuta"
pip install spacy pandas scikit-learn numpy
python -m spacy download en_core_web_md
```

**Esecuzione della classificazione:**
```bash
python Classificazione_Consumer.py
```

**Processo di classificazione:**

1. **Caricamento tuple**: Carica le tuple sincronizzate dal Consumer
2. **Costruzione profili**: Crea profili avanzati per ogni campagna utilizzando:
   - **TF-IDF**: Analisi frequenza termini
   - **Word Embeddings**: Rappresentazioni semantiche con spaCy
   - **Keywords enhanced**: Estrazione entità e termini chiave
   - **Pattern speciali**: Riconoscimento contenuti Twitter, elezioni, clima

3. **Classificazione multi-criterio**:
   - **TF-IDF similarity** (peso 50%): Similarità vettoriale
   - **Embedding similarity** (peso 25%): Similarità semantica
   - **Keyword bonus** (peso 15%): Corrispondenza termini chiave
   - **Special content bonus** (peso 10%): Pattern specifici

**Output generati:**
- `risultati_ultra_ottimizzati_[timestamp].xlsx` - Risultati dettagliati con:
  - PREDICTED_CAMPAIGN: Campagna predetta
  - PREDICTION_CONFIDENCE: Livello di confidenza (0-1)
  - Score dettagliati per ogni componente


### 6. Generazione Contenuti (Baseline2)
Genera artificialmente fake news utilizzando LLaMA per creare un test set sintetico per analisi predittiva.

cd "-Generazione[Baseline2]"
# Opzione A: Google Colab (consigliata)
 generazione_partendo_da_articoli.ipynb

# Opzione B: Locale
cd Generazione in Locale_llama
python generazione.py
**Processo:**
Selezione campagna dal dataset originale
Creazione prompt con esempi della campagna scelta
Generazione nuovo articolo fake con stile simile
Monitoraggio risorse (CPU, RAM, GPU) durante generazione
Salvataggio in Generated_fake_news.xlsx

**Output:**: Articoli generati per valutare robustezza del sistema di classificazione
 
### 6. Generazione Contenuti (Baseline2)
Genera artificialmente fake news utilizzando LLaMA per creare un test set sintetico per analisi predittiva.

cd "-Generazione[Baseline2]"
# Opzione A: Google Colab (consigliata)
 generazione_partendo_da_articoli.ipynb

# Opzione B: Locale
cd Generazione in Locale_llama
python generazione.py
**Processo:**
Selezione campagna dal dataset originale
Creazione prompt con esempi della campagna scelta
Generazione nuovo articolo fake con stile simile
Monitoraggio risorse (CPU, RAM, GPU) durante generazione
Salvataggio in Generated_fake_news.xlsx

**Output:**: Articoli generati per valutare robustezza del sistema di classificazione
 ### 6. Generazione Contenuti (Baseline2)
Genera artificialmente fake news utilizzando LLaMA per creare un test set sintetico per analisi predittiva.

cd "-Generazione[Baseline2]"
# Opzione A: Google Colab (consigliata)
 generazione_partendo_da_articoli.ipynb

# Opzione B: Locale
cd Generazione in Locale_llama
python generazione.py
**Processo:**
Selezione campagna dal dataset originale
Creazione prompt con esempi della campagna scelta
Generazione nuovo articolo fake con stile simile
Monitoraggio risorse (CPU, RAM, GPU) durante generazione
Salvataggio in Generated_fake_news.xlsx

**Output:**: Articoli generati per valutare robustezza del sistema di classificazione
 ### 6. Generazione Contenuti (Baseline2)
Genera artificialmente fake news utilizzando LLaMA per creare un test set sintetico per analisi predittiva.

cd "-Generazione[Baseline2]"
# Opzione A: Google Colab (consigliata)
 generazione_partendo_da_articoli.ipynb

# Opzione B: Locale
cd Generazione in Locale_llama
python generazione.py
**Processo:**
Selezione campagna dal dataset originale
Creazione prompt con esempi della campagna scelta
Generazione nuovo articolo fake con stile simile
Monitoraggio risorse (CPU, RAM, GPU) durante generazione
Salvataggio in Generated_fake_news.xlsx

**Output:**: Articoli generati per valutare robustezza del sistema di classificazione
 a### 6. Generazione Contenuti (Baseline2)
Genera artificialmente fake news utilizzando LLaMA per creare un test set sintetico per analisi predittiva.

cd "-Generazione[Baseline2]"
# Opzione A: Google Colab (consigliata)
 generazione_partendo_da_articoli.ipynb

# Opzione B: Locale
cd Generazione in Locale_llama
python generazione.py
**Processo:**
Selezione campagna dal dataset originale
Creazione prompt con esempi della campagna scelta
Generazione nuovo articolo fake con stile simile
Monitoraggio risorse (CPU, RAM, GPU) durante generazione
Salvataggio in Generated_fake_news.xlsx

**Output:**: Articoli generati per valutare robustezza del sistema di classificazione
 
 #### Implementazione Alternativa con DeepSeek
```bash
cd "-Alternativa DeepSeek estrazionetuple"
# Carica e esegui il notebook per la generazione con DeepSeek:
jupyter notebook DeepSeek_completo.ipynb
```

**Nota**: Il notebook `DeepSeek_completo.ipynb` include sia l'estrazione delle tuple che la generazione di contenuti utilizzando il modello DeepSeek, fornendo un'implementazione completa alternativa al workflow principale con LLaMA.

**Processo comune ad entrambe le implementazioni:**
- Selezione campagna dal dataset originale
- Creazione prompt con esempi della campagna scelta
- Generazione nuovo articolo fake con stile simile
- Monitoraggio risorse (CPU, RAM, GPU) durante generazione
- Salvataggio in Generated_fake_news.xlsx

**Output**: Articoli generati per valutare robustezza del sistema di classificazione, confrontando le performance tra LLaMA e DeepSeek

### 7. FAKEBERT Esperimento (Baseline Stato dell'Arte)

Implementa il sistema BERT classico per il rilevamento di fake news come confronto con l'approccio innovativo proposto nella tesi.

```bash
cd "FAKEBERT"
# Esecuzione in Google Colab (consigliata per GPU)
jupyter notebook fakebert_definitiva.ipynb
```

**Caratteristiche del sistema FAKEBERT:**
- Utilizza il modello BERT pre-addestrato (`bert-base-uncased`)
- Approccio supervisionato classico: addestramento su dataset etichettato "vero/falso"
- Analizza ogni notizia in modo isolato senza considerare campagne di disinformazione

**Limiti dell'approccio BERT identificati:**
- Non riconosce campagne di disinformazione organizzate
- Analisi superficiale basata su pattern testuali
- Mancanza di conoscenza esperta e contesto semantico
- Fallisce su contenuti diversi dal training set (es. notizie FakeCTI)