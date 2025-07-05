import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Imposta lo stile dei grafici
plt.style.use('default')
sns.set_palette("viridis")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

try:
    # Carica il dataset
    df = pd.read_excel('FakeCTI Dataset.xlsx')
    print("Dataset caricato con successo!")
    print(f"Dimensioni: {df.shape}")
    print(f"Colonne: {list(df.columns)}")
    
    # Statistiche base
    print("\n=== STATISTICHE FAKECTI DATASET ===")
    print(f"Totale articoli: {len(df):,}")
    
    # Verifica colonne disponibili (nomi corretti in maiuscolo)
    if 'CAMPAIGN' in df.columns:
        print(f"Campagne uniche: {df['CAMPAIGN'].nunique()}")
        
        # Grafico unico: Top 8 campagne con Russian Troll in scala ridotta
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prendi le top 8 campagne
        top_campaigns = df['CAMPAIGN'].value_counts().head(8)
        
        # Modifica il valore di Russian Troll per la visualizzazione
        display_values = []
        actual_values = []
        labels = []
        
        for campaign, value in top_campaigns.items():
            if 'Russian troll' in campaign:
                display_values.append(1000)  # Valore per visualizzazione
                actual_values.append(value)   # Valore reale
                labels.append(f"{campaign} ({value:,})")
            else:
                display_values.append(value)
                actual_values.append(value)
                labels.append(campaign)
        
        # Colori gradient per le barre
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_campaigns)))
        
        bars = ax.barh(range(len(top_campaigns)), display_values, color=colors)
        ax.set_yticks(range(len(top_campaigns)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Numero di Articoli', fontsize=12, fontweight='bold')
        ax.set_title('Top 8 Campagne Principali', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Aggiungi valori sulle barre
        for i, (bar, actual_val, display_val) in enumerate(zip(bars, actual_values, display_values)):
            if actual_val != display_val:  # Russian Troll
                ax.text(display_val + display_val*0.02, i, f'{actual_val:,}', va='center', fontweight='bold', color='black')
            else:
                ax.text(display_val + display_val*0.02, i, f'{display_val:,}', va='center', fontweight='bold', color='black')
        
        # Bordi
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
        
        plt.tight_layout()
        plt.savefig('top_campaigns_scaled.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    if 'THREAT ACTOR' in df.columns:
        print(f"Attori minacciosi: {df['THREAT ACTOR'].nunique()}")
        
        # Grafico 2: Attori minacciosi con grafico a barre orizzontali
        fig, ax = plt.subplots(figsize=(14, 10))
        threat_counts = df['THREAT ACTOR'].value_counts()
        
        # Colori colorati per le barre
        colors = plt.cm.Set1(np.linspace(0, 1, len(threat_counts)))
        
        bars = ax.barh(range(len(threat_counts)), threat_counts.values, color=colors)
        ax.set_yticks(range(len(threat_counts)))
        ax.set_yticklabels(threat_counts.index, fontsize=11)
        ax.set_xlabel('Numero di Articoli', fontsize=12, fontweight='bold')
        ax.set_title('Distribuzione per Attore Minaccioso', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Aggiungi valori sulle barre
        for i, (bar, value) in enumerate(zip(bars, threat_counts.values)):
            ax.text(value + value*0.02, i, f'{value:,}', va='center', fontweight='bold', color='black')
        
        # Bordi e stile
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
        
        plt.tight_layout()
        plt.savefig('threat_actors_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    # Nota: Non c'è una colonna 'Medium' nel tuo dataset, ma c'è 'TYPE'
    if 'TYPE' in df.columns:
        print(f"Tipi utilizzati: {df['TYPE'].nunique()}")
        
        # Grafico 3: Distribuzione per tipo con colori personalizzati
        fig, ax = plt.subplots(figsize=(12, 8))
        type_counts = df['TYPE'].value_counts()
        
        # Colori personalizzati per evitare il nero
        colors = []
        for tipo in type_counts.index:
            if 'TWITTER' in tipo.upper():
                colors.append('#1DA1F2')  # Blu Twitter ufficiale
            elif 'WEB' in tipo.upper():
                colors.append('#FF6B6B')  # Rosso corallo
            elif 'FACEBOOK' in tipo.upper():
                colors.append('#4267B2')  # Blu Facebook
            else:
                colors.append('#4ECDC4')  # Verde acqua
        
        bars = ax.bar(type_counts.index, type_counts.values, color=colors)
        ax.set_xlabel('Tipo di Contenuto', fontsize=12, fontweight='bold')
        ax.set_ylabel('Numero di Articoli', fontsize=12, fontweight='bold')
        ax.set_title('Distribuzione per Tipo di Contenuto', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Aggiungi valori sopra le barre
        for bar, value in zip(bars, type_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                   f'{value:,}', ha='center', va='bottom', fontweight='bold', color='black')
        
        # Bordi e stile
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('type_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    if 'SOURCE' in df.columns:
        print(f"Fonti diverse: {df['SOURCE'].nunique()}")
        
        # Grafico 4: Top fonti con stile moderno
        fig, ax = plt.subplots(figsize=(14, 10))
        top_sources = df['SOURCE'].value_counts().head(15)
        
        # Colori gradient
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(top_sources)))
        
        bars = ax.barh(range(len(top_sources)), top_sources.values, color=colors)
        ax.set_yticks(range(len(top_sources)))
        ax.set_yticklabels(top_sources.index, fontsize=10)
        ax.set_xlabel('Numero di Articoli', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Fonti per Numero di Articoli', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Aggiungi valori sulle barre
        for i, (bar, value) in enumerate(zip(bars, top_sources.values)):
            ax.text(value + value*0.02, i, f'{value:,}', va='center', fontweight='bold', color='white')
        
        # Bordi e stile
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(0.5)
        
        plt.tight_layout()
        plt.savefig('sources_distribution.png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.show()
    
    # Tabella riassuntiva
    print("\n=== TABELLA RIASSUNTIVA ===")
    summary_data = {
        'Metrica': ['Totale Articoli', 'Campagne', 'Attori Minacciosi', 'Fonti', 'Tipi'],
        'Valore': [
            len(df),
            df['CAMPAIGN'].nunique() if 'CAMPAIGN' in df.columns else 'N/A',
            df['THREAT ACTOR'].nunique() if 'THREAT ACTOR' in df.columns else 'N/A',
            df['SOURCE'].nunique() if 'SOURCE' in df.columns else 'N/A',
            df['TYPE'].nunique() if 'TYPE' in df.columns else 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Salva la tabella
    summary_df.to_csv('fakecti_summary.csv', index=False)
    print("\nTabella salvata come 'fakecti_summary.csv'")
    
    # Mostra un campione dei dati per verifica
    print("\n=== CAMPIONE DEI DATI ===")
    print(df[['TITLE', 'SOURCE', 'CAMPAIGN', 'THREAT ACTOR', 'TYPE']].head())
    
except FileNotFoundError:
    print("ERRORE: File 'FakeCTI Dataset.xlsx' non trovato!")
    print("Assicurati che il file sia nella stessa cartella dello script.")
except Exception as e:
    print(f"ERRORE: {e}")
    print("Debug: Colonne disponibili nel dataset:")
    try:
        df = pd.read_excel('FakeCTI Dataset.xlsx')
        print(list(df.columns))
    except:
        pass