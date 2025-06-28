import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Legge il dataset dal file Excel "dataset.xlsx"
df = pd.read_excel("FakeCTI Dataset.xlsx")

# Estrae tutti i nomi unici delle campagne presenti nel dataset
campaigns = df['CAMPAIGN'].unique().tolist()

# Conta il numero di articoli per ogni campagna
campaign_counts = df['CAMPAIGN'].value_counts()
print("Campagne e numero di articoli:")
for campaign, count in campaign_counts.items():
    print(f"- {campaign}: {count} articoli")

# Seleziona solo le campagne con >10 articoli
eligible_campaigns = campaign_counts[campaign_counts > 10]
print(f"\nCampagne con >10 articoli: {len(eligible_campaigns)}")
for campaign, count in eligible_campaigns.items():
    print(f"- {campaign}: {count} articoli")

# Crea dataframe vuoti per i set di train e test
train_df = pd.DataFrame(columns=df.columns)
test_df = pd.DataFrame(columns=df.columns)

# Per ogni campagna eligible, applica la strategia
for campaign in eligible_campaigns.index:
    # Filtra gli articoli relativi a questa campagna
    campaign_articles = df[df['CAMPAIGN'] == campaign]
    
    # Mescola gli articoli in modo casuale
    campaign_articles = campaign_articles.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_articles = len(campaign_articles)
    print(f"\nCampagna '{campaign}': {total_articles} articoli totali")
    
    if total_articles > 101:
        # Caso >101: seleziona solo 100 istanze (90 train + 10 test)
        selected_articles = campaign_articles.iloc[:100]
        train_size = 90
        test_size = 10
        print(f"  -> Campagna >101 articoli: limitata a 100 (90 train + 10 test)")
        
    else:
        # Caso 11-101: usa tutti gli articoli con divisione 90%-10%
        selected_articles = campaign_articles
        train_size = int(total_articles * 0.9)
        test_size = total_articles - train_size
        print(f"  -> Campagna 11-101 articoli: divisione 90%-10% ({train_size} train + {test_size} test)")
    
    # Divide in train e test
    train_articles = selected_articles.iloc[:train_size]
    test_articles = selected_articles.iloc[train_size:train_size + test_size]
    
    # Aggiunge agli dataframe finali
    train_df = pd.concat([train_df, train_articles], ignore_index=True)
    test_df = pd.concat([test_df, test_articles], ignore_index=True)

# Salva i dataframe train e test in file Excel separati
train_df.to_excel("train.xlsx", index=False)
test_df.to_excel("test.xlsx", index=False)

total_final = len(train_df) + len(test_df)
print(f"\n{'='*50}")
print(f"RISULTATI FINALI:")
print(f"{'='*50}")
print(f"Set di train finale: {len(train_df)} articoli salvati in train.xlsx")
print(f"Set di test finale: {len(test_df)} articoli salvati in test.xlsx")
print(f"Totale articoli utilizzati: {total_final}")

if total_final > 0:
    print(f"Percentuale finale - Train: {(len(train_df)/total_final)*100:.1f}%, Test: {(len(test_df)/total_final)*100:.1f}%")

print(f"\n{'='*50}")
print("DISTRIBUZIONE FINALE PER CAMPAGNA:")
print(f"{'='*50}")

print("\nTrain set:")
train_campaign_counts = train_df['CAMPAIGN'].value_counts().sort_values(ascending=False)
for campaign, count in train_campaign_counts.items():
    original_count = campaign_counts[campaign]
    if original_count > 101:
        print(f"- {campaign}: {count} articoli (da {original_count} → limitato a 100, 90 per train)")
    else:
        print(f"- {campaign}: {count} articoli (da {original_count}, 90% per train)")

print("\nTest set:")
test_campaign_counts = test_df['CAMPAIGN'].value_counts().sort_values(ascending=False)
for campaign, count in test_campaign_counts.items():
    original_count = campaign_counts[campaign]
    if original_count > 101:
        print(f"- {campaign}: {count} articoli (da {original_count} → limitato a 100, 10 per test)")
    else:
        print(f"- {campaign}: {count} articoli (da {original_count}, 10% per test)")

# Verifica che le regole siano state applicate correttamente
print(f"\n{'='*50}")
print("VERIFICA REGOLE:")
print(f"{'='*50}")
for campaign in eligible_campaigns.index:
    original_count = campaign_counts[campaign]
    train_count = train_campaign_counts.get(campaign, 0)
    test_count = test_campaign_counts.get(campaign, 0)
    total_used = train_count + test_count
    
    if original_count > 101:
        expected_total = 100
        expected_train = 90
        expected_test = 10
    else:
        expected_total = original_count
        expected_train = int(original_count * 0.9)
        expected_test = original_count - expected_train
    
    status = "✓" if (total_used == expected_total and train_count == expected_train and test_count == expected_test) else "✗"
    print(f"{status} {campaign}: {train_count}+{test_count}={total_used} (atteso: {expected_train}+{expected_test}={expected_total})")