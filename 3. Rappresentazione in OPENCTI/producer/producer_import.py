#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pycti import OpenCTIApiClient
import time
import sys
import json
import hashlib
import uuid
import logging
import re

# Disabilita i log di debug dell'API
logging.getLogger('api').setLevel(logging.WARNING)
logging.getLogger('pycti').setLevel(logging.WARNING)

# Configurazione del client
client = OpenCTIApiClient("http://localhost:8081", "e4eaaaf2-d142-11e1-b3e4-080027620c1d")

# Inizializzazione dei valori di default globali
print("Inizializzazione dei valori di default...")
DEFAULT_CREATED_BY = None
DEFAULT_MARKING = None

# Path del file Excel
file_path = "tuple estratte.xlsx"

# Caricamento del file Excel
try:
    print(f"Caricamento del file: {file_path}")
    df = pd.read_excel(file_path)
    print(f"File caricato con successo. {len(df)} righe trovate.")
except Exception as e:
    print(f"Errore durante il caricamento del file: {e}")
    sys.exit(1)

def clean_text(text):
    """
    Pulisce il testo rimuovendo parti tra parentesi e caratteri problematici
    """
    if not text:
        return ""
    
    # Rimuovi tutto il contenuto tra parentesi (incluse le parentesi)
    cleaned = re.sub(r'\s*\([^)]*\)', '', text)
    
    # Rimuovi spazi multipli e caratteri speciali problematici
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Limita la lunghezza per evitare problemi
    if len(cleaned) > 200:
        cleaned = cleaned[:200] + "..."
    
    return cleaned

def safe_name(text):
    """
    Crea un nome sicuro per OpenCTI rimuovendo caratteri problematici
    """
    if not text:
        return "unnamed"
    
    # Prima pulisci il testo
    clean = clean_text(text)
    
    # Se il testo pulito è vuoto, usa il testo originale limitato
    if not clean:
        clean = text[:100] if text else "unnamed"
    
    # Rimuovi caratteri che possono causare problemi
    safe = re.sub(r'[<>:"/\\|?*]', '', clean)
    safe = re.sub(r'\s+', ' ', safe).strip()
    
    return safe if safe else "unnamed"

def get_default_marking():
    """Recupera l'ID del TLP:GREEN marking o crea un fallback se non trovato"""
    try:
        tlp_green_id = "marking-definition--613f2e26-407d-48c7-9eca-b8e91df99dc9"
        markings = client.marking_definition.list()
        
        for marking in markings:
            if marking.get('id') == tlp_green_id:
                return [tlp_green_id]
                
        for marking in markings:
            if "TLP:" in marking.get('definition_type', ''):
                return [marking.get('id')]
                
        if markings and len(markings) > 0:
            return [markings[0].get('id')]
            
        return [tlp_green_id]
    except Exception as e:
        print(f"Errore nel recupero dei marking definitions: {e}")
        return ["marking-definition--613f2e26-407d-48c7-9eca-b8e91df99dc9"]

def get_default_identity():
    """Recupera l'ID dell'identità di sistema di default o crea un fallback"""
    try:
        filters = {
            "mode": "and",
            "filters": [{"key": "name", "values": ["System"]}],
            "filterGroups": []
        }
        system_identities = client.identity.list(filters=filters)
        
        if system_identities and len(system_identities) > 0:
            return system_identities[0].get('id')
            
        identities = client.identity.list()
        if identities and len(identities) > 0:
            return identities[0].get('id')
            
        system_identity = client.identity.create(
            name="System",
            type="System",
            description="Default system identity"
        )
        return system_identity.get('id')
    except Exception as e:
        print(f"Errore nel recupero dell'identità di sistema: {e}")
        return "identity--d37acc64-4a6f-4dc2-879a-a4c138d0a27f"

def get_or_create_label(label_name, color="#FF6B6B"):
    """Trova o crea una label"""
    try:
        filters = {
            "mode": "and",
            "filters": [{"key": "value", "values": [label_name]}],
            "filterGroups": []
        }
        existing_labels = client.label.list(filters=filters)
        
        if existing_labels and len(existing_labels) > 0:
            return existing_labels[0]['id']
        else:
            new_label = client.label.create(value=label_name, color=color)
            return new_label['id']
    except Exception as e:
        print(f"Errore nella gestione della label '{label_name}': {e}")
        return None

def create_subject_entity(contenuto, tripla_completa, campagna):
    """
    Crea un'Identity per rappresentare il soggetto della tripla
    """
    try:
        # Nome pulito e sicuro
        entity_name = safe_name(contenuto)
        contenuto_pulito = clean_text(contenuto)
        
        # Controlla se l'entità esiste già usando il contenuto pulito
        filters = {
            "mode": "and",
            "filters": [
                {"key": "name", "values": [entity_name]},
                {"key": "x_triple_contenuto", "values": [contenuto_pulito]}
            ],
            "filterGroups": []
        }
        
        try:
            existing_entities = client.identity.list(filters=filters)
            for entity in existing_entities:
                if (entity.get('x_triple_ruolo') == 'soggetto' and 
                    entity.get('x_triple_contenuto') == contenuto_pulito):
                    print(f"Soggetto esistente trovato: {entity_name}")
                    return entity
        except:
            pass
        
        # Determina il tipo di identità
        identity_class = "Organization"
        if any(keyword in contenuto_pulito.lower() for keyword in ["person", "persona", "individ", "mr", "mrs", "dr"]):
            identity_class = "Individual"
        elif any(keyword in contenuto_pulito.lower() for keyword in ["country", "nation", "state", "governo"]):
            identity_class = "Organization"
        
        # Prepara le labels
        subject_labels = []
        for label_name in ["semantic-triple", "soggetto", "disinformation"]:
            label_id = get_or_create_label(label_name, "#FF6B6B")
            if label_id:
                subject_labels.append(label_id)
        
        # Crea la nuova Identity
        entity_data = {
            "name": entity_name,
            "type": identity_class,
            "description": f"Soggetto della tripla semantica: {clean_text(tripla_completa)}",
            "objectLabel": subject_labels,
            "x_triple_contenuto": contenuto_pulito,
            "x_triple_ruolo": "soggetto",
            "x_triple_completa": clean_text(tripla_completa),
            "x_triple_campagna": campagna,
            "createdBy": DEFAULT_CREATED_BY,
            "objectMarking": DEFAULT_MARKING,
            "x_opencti_score": 60
        }
        
        entity = client.identity.create(**entity_data)
        print(f"Soggetto creato: {entity_name}")
        return entity
        
    except Exception as e:
        print(f"Errore nella creazione del soggetto '{contenuto}': {e}")
        return None

def create_verb_entity(contenuto, tripla_completa, campagna):
    """
    Crea un Attack-Pattern per rappresentare il verbo/azione della tripla
    """
    try:
        # Nome pulito e sicuro
        entity_name = safe_name(contenuto)
        contenuto_pulito = clean_text(contenuto)
        
        # Controlla se l'entità esiste già
        filters = {
            "mode": "and",
            "filters": [
                {"key": "name", "values": [entity_name]},
                {"key": "x_triple_contenuto", "values": [contenuto_pulito]}
            ],
            "filterGroups": []
        }
        
        try:
            existing_entities = client.attack_pattern.list(filters=filters)
            for entity in existing_entities:
                if (entity.get('x_triple_ruolo') == 'verbo' and 
                    entity.get('x_triple_contenuto') == contenuto_pulito):
                    print(f"Verbo esistente trovato: {entity_name}")
                    return entity
        except:
            pass
        
        # Prepara le labels
        verb_labels = []
        for label_name in ["semantic-triple", "verbo", "disinformation", "action"]:
            label_id = get_or_create_label(label_name, "#4ECDC4")
            if label_id:
                verb_labels.append(label_id)
        
        # Crea il nuovo Attack-Pattern
        entity_data = {
            "name": entity_name,
            "description": f"Azione/verbo della tripla semantica: {clean_text(tripla_completa)}",
            "objectLabel": verb_labels,
            "x_mitre_platforms": ["disinformation"],
            "x_triple_contenuto": contenuto_pulito,
            "x_triple_ruolo": "verbo",
            "x_triple_completa": clean_text(tripla_completa),
            "x_triple_campagna": campagna,
            "createdBy": DEFAULT_CREATED_BY,
            "objectMarking": DEFAULT_MARKING,
            "x_opencti_score": 60
        }
        
        entity = client.attack_pattern.create(**entity_data)
        print(f"Verbo creato: {entity_name}")
        return entity
        
    except Exception as e:
        print(f"Errore nella creazione del verbo '{contenuto}': {e}")
        return None

def create_object_entity(contenuto, tripla_completa, campagna):
    """
    Crea un'Identity per rappresentare l'oggetto della tripla
    """
    try:
        # Nome pulito e sicuro
        entity_name = safe_name(contenuto)
        contenuto_pulito = clean_text(contenuto)
        
        # Controlla se l'entità esiste già
        filters = {
            "mode": "and",
            "filters": [
                {"key": "name", "values": [entity_name]},
                {"key": "x_triple_contenuto", "values": [contenuto_pulito]}
            ],
            "filterGroups": []
        }
        
        try:
            existing_entities = client.identity.list(filters=filters)
            for entity in existing_entities:
                if (entity.get('x_triple_ruolo') == 'oggetto' and 
                    entity.get('x_triple_contenuto') == contenuto_pulito):
                    print(f"Oggetto esistente trovato: {entity_name}")
                    return entity
        except:
            pass
        
        # Determina il tipo di identità
        identity_class = "Organization"
        if any(keyword in contenuto_pulito.lower() for keyword in ["person", "persona", "individ"]):
            identity_class = "Individual"
        
        # Prepara le labels
        object_labels = []
        for label_name in ["semantic-triple", "oggetto", "disinformation"]:
            label_id = get_or_create_label(label_name, "#45B7D1")
            if label_id:
                object_labels.append(label_id)
        
        # Crea la nuova Identity
        entity_data = {
            "name": entity_name,
            "type": identity_class,
            "description": f"Oggetto della tripla semantica: {clean_text(tripla_completa)}",
            "objectLabel": object_labels,
            "x_triple_contenuto": contenuto_pulito,
            "x_triple_ruolo": "oggetto",
            "x_triple_completa": clean_text(tripla_completa),
            "x_triple_campagna": campagna,
            "createdBy": DEFAULT_CREATED_BY,
            "objectMarking": DEFAULT_MARKING,
            "x_opencti_score": 60
        }
        
        entity = client.identity.create(**entity_data)
        print(f"Oggetto creato: {entity_name}")
        return entity
        
    except Exception as e:
        print(f"Errore nella creazione dell'oggetto '{contenuto}': {e}")
        return None

def create_main_indicator(tripla_completa, campagna):
    """
    Crea l'Indicator principale che rappresenta l'intera tripla semantica
    """
    try:
        # Pulisci la tripla
        tripla_pulita = clean_text(tripla_completa)
        indicator_name = safe_name(tripla_pulita)
        
        # Genera un pattern STIX personalizzato per la tripla
        pattern = f"[x-semantic-triple:content = '{tripla_pulita}']"
        
        # Controlla se l'indicatore esiste già
        filters = {
            "mode": "and",
            "filters": [{"key": "pattern", "values": [pattern]}],
            "filterGroups": []
        }
        existing_indicators = client.indicator.list(filters=filters)
        
        if existing_indicators and len(existing_indicators) > 0:
            print(f"Indicatore principale esistente trovato per: {indicator_name}")
            return existing_indicators[0]
        
        # Prepara le labels
        indicator_labels = []
        for label_name in ["disinformation", "semantic-triple", "narrative", "misinformation", "tripla-completa"]:
            label_id = get_or_create_label(label_name, "#9B59B6")
            if label_id:
                indicator_labels.append(label_id)
        
        # Crea il nuovo indicatore principale
        indicator_data = {
            "name": indicator_name,
            "pattern": pattern,
            "pattern_type": "stix",
            "x_opencti_main_observable_type": "Unknown",
            "objectLabel": indicator_labels,
            "description": f"Complete semantic triple from disinformation campaign '{campagna}': {tripla_pulita}",
            "confidence": 80,
            "createdBy": DEFAULT_CREATED_BY,
            "objectMarking": DEFAULT_MARKING,
            "x_opencti_score": 75,
            "x_opencti_detection": True,
            "valid_from": "2020-01-01T00:00:00.000Z",
            "x_triple_completa": tripla_pulita,
            "x_triple_campagna": campagna,
            "x_triple_type": "semantic-narrative"
        }
        
        indicator = client.indicator.create(**indicator_data)
        print(f"Indicatore principale creato: {indicator_name}")
        return indicator
        
    except Exception as e:
        print(f"Errore nella creazione dell'indicatore principale '{tripla_completa}': {e}")
        return None

def find_or_create_campaign(name):
    """Trova o crea una campagna di disinformazione"""
    if not name or name.strip() == "":
        print("Nome campagna vuoto")
        return None
    
    try:
        filters = {
            "mode": "and",
            "filters": [{"key": "name", "values": [name]}],
            "filterGroups": []
        }
        campaigns = client.campaign.list(filters=filters)
        
        if campaigns and len(campaigns) > 0:
            print(f"Campagna trovata: {name}")
            return campaigns[0]
    except Exception as e:
        print(f"Errore nella ricerca della campagna '{name}': {e}")
    
    try:
        campaign = client.campaign.create(
            name=name,
            description=f"Disinformation campaign containing semantic triples: {name}",
            createdBy=DEFAULT_CREATED_BY,
            objectMarking=DEFAULT_MARKING,
            x_opencti_score=70
        )
        print(f"Campagna creata: {name}")
        return campaign
    except Exception as e:
        print(f"Errore nella creazione della campagna '{name}': {e}")
        return None

def create_core_relationship(source_id, target_id, relationship_type, description):
    """Crea una relazione STIX Core tra due oggetti con retry logic"""
    try:
        # Controlla se la relazione esiste già
        filters = {
            "mode": "and",
            "filters": [
                {"key": "fromId", "values": [source_id]},
                {"key": "toId", "values": [target_id]},
                {"key": "relationship_type", "values": [relationship_type]}
            ],
            "filterGroups": []
        }
        
        existing_relationships = client.stix_core_relationship.list(filters=filters)
        if existing_relationships and len(existing_relationships) > 0:
            return existing_relationships[0]

        # Crea una nuova relazione core con retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                relationship = client.stix_core_relationship.create(
                    fromId=source_id,
                    toId=target_id,
                    relationship_type=relationship_type,
                    description=description[:500],  # Limita la descrizione
                    confidence=75,
                    createdBy=DEFAULT_CREATED_BY,
                    objectMarking=DEFAULT_MARKING,
                    x_opencti_score=65
                )
                return relationship
            except Exception as retry_error:
                if attempt < max_retries - 1:
                    print(f"Tentativo {attempt + 1} fallito, riprovo in 2 secondi...")
                    time.sleep(2)
                else:
                    print(f"Errore nella creazione della relazione dopo {max_retries} tentativi: {retry_error}")
                    return None
    except Exception as e:
        print(f"Errore nella creazione della relazione {relationship_type}: {e}")
        return None

def process_semantic_triples():
    """
    Elabora le triple dal file Excel creando il modello semantico
    """
    total = len(df)
    success = 0
    failed = 0

    print(f"\nInizio elaborazione di {total} triple semantiche...\n")

    for index, row in df.iterrows():
        try:
            tupla = str(row['TUPLA']).strip()
            campagna = str(row['CAMPAGNA']).strip()
            
            print(f"\n[{index+1}/{total}] Elaborazione tripla: '{tupla[:100]}...' (Campagna: '{campagna}')")
        except KeyError as e:
            print(f"Errore nelle colonne: {e}. Assicurati che siano: 'TUPLA', 'CAMPAGNA'")
            return

        try:
            # Pulisci la tupla prima di dividerla
            tupla_pulita = clean_text(tupla)
            
            # Dividi la tupla nei suoi componenti
            parts = tupla_pulita.split(" - ")
            if len(parts) != 3:
                print(f"Riga {index+1}: formato TUPLA non valido ('{tupla_pulita}'). Salto.")
                failed += 1
                continue
                
            soggetto, verbo, oggetto = [x.strip() for x in parts]
            print(f"  Soggetto: '{soggetto[:50]}...', Verbo: '{verbo}', Oggetto: '{oggetto[:50]}...'")
            
            # FASE 1: Crea le entità STIX
            print(f"  1. Creazione entità STIX...")
            
            entity_soggetto = create_subject_entity(soggetto, tupla, campagna)
            time.sleep(1)  # Pausa per evitare lock
            
            entity_verbo = create_verb_entity(verbo, tupla, campagna)
            time.sleep(1)  # Pausa per evitare lock
            
            entity_oggetto = create_object_entity(oggetto, tupla, campagna)
            time.sleep(1)  # Pausa per evitare lock
            
            if not all([entity_soggetto, entity_verbo, entity_oggetto]):
                print(f"Impossibile creare tutte le entità per la tripla")
                failed += 1
                continue
            
            # FASE 2: Crea l'indicatore principale
            print(f"  2. Creazione indicatore principale...")
            indicatore_principale = create_main_indicator(tupla, campagna)
            
            if not indicatore_principale:
                print(f"Impossibile creare l'indicatore principale")
                failed += 1
                continue
            
            # FASE 3: Collega l'indicatore principale alle entità
            print(f"  3. Creazione relazioni...")
            
            rel_soggetto = create_core_relationship(
                source_id=indicatore_principale['id'],
                target_id=entity_soggetto['id'],
                relationship_type="related-to",
                description=f"Tripla semantica contiene soggetto"
            )
            
            rel_verbo = create_core_relationship(
                source_id=indicatore_principale['id'],
                target_id=entity_verbo['id'],
                relationship_type="related-to",
                description=f"Tripla semantica contiene azione"
            )
            
            rel_oggetto = create_core_relationship(
                source_id=indicatore_principale['id'],
                target_id=entity_oggetto['id'],
                relationship_type="related-to",
                description=f"Tripla semantica contiene oggetto"
            )
            
            # FASE 4: Crea relazioni semantiche tra soggetto-verbo-oggetto
            rel_subj_verb = create_core_relationship(
                source_id=entity_soggetto['id'],
                target_id=entity_verbo['id'],
                relationship_type="related-to",
                description=f"Il soggetto esegue l'azione"
            )
            
            rel_verb_obj = create_core_relationship(
                source_id=entity_verbo['id'],
                target_id=entity_oggetto['id'],
                relationship_type="targets",
                description=f"L'azione è diretta verso l'oggetto"
            )
            
            relations_created = sum(1 for rel in [rel_soggetto, rel_verbo, rel_oggetto, rel_subj_verb, rel_verb_obj] if rel)
            print(f"  {relations_created}/5 relazioni create")
            
            # FASE 5: Collega l'indicatore alla campagna
            print(f"  5. Collegamento alla campagna...")
            campaign = find_or_create_campaign(campagna)
            
            if campaign:
                rel_campagna = create_core_relationship(
                    source_id=indicatore_principale['id'],
                    target_id=campaign['id'],
                    relationship_type="related-to",
                    description=f"Tripla semantica attribuita alla campagna"
                )
                
                if rel_campagna:
                    print(f"  Tripla collegata alla campagna: {campagna}")
            
            success += 1
            print(f"  Tripla elaborata con successo")

        except Exception as e:
            failed += 1
            print(f"Errore nell'elaborazione della tripla: {e}")

        # Pausa tra le elaborazioni per evitare sovraccarichi
        if (index + 1) % 2 == 0:
            print("  Pausa di 3 secondi...")
            time.sleep(3)

    print(f"\n" + "="*60)
    print(f"ELABORAZIONE COMPLETATA")
    print(f"Triple processate con successo: {success}")
    print(f"Triple fallite: {failed}")
    print("="*60)

# Punto di ingresso principale
if __name__ == "__main__":
    print("Inizializzazione OpenCTI...")
    
    try:
        DEFAULT_MARKING = get_default_marking()
        DEFAULT_CREATED_BY = get_default_identity()
        
        print(f"Usando marking: {DEFAULT_MARKING}")
        print(f"Usando createdBy identity: {DEFAULT_CREATED_BY}")
        
        process_semantic_triples()
    except Exception as e:
        print(f"Errore critico: {e}")
        sys.exit(1)