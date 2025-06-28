#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycti import OpenCTIApiClient
import time
import traceback

# Configurazione
PRODUCER_URL = "http://localhost:8081"
PRODUCER_TOKEN = "e4eaaaf2-d142-11e1-b3e4-080027620c1d"

CONSUMER_URL = "http://localhost:8080"
CONSUMER_TOKEN = "f5feb3a4-e152-21f2-c4f5-181138731d2e"

# Connessione alle due istanze
producer_client = OpenCTIApiClient(PRODUCER_URL, PRODUCER_TOKEN)
consumer_client = OpenCTIApiClient(CONSUMER_URL, CONSUMER_TOKEN)

def get_or_create_default_marking():
    """Recupera o crea marking definitions di default"""
    try:
        markings = consumer_client.marking_definition.list()
        if markings and len(markings) > 0:
            return [markings[0]['id']]
        return None
    except:
        return None

def get_or_create_default_identity():
    """Recupera o crea identity di default"""
    try:
        identities = consumer_client.identity.list()
        for identity in identities:
            if identity.get('name') == 'System':
                return identity['id']
        
        # Crea identity di sistema se non esiste
        system_identity = consumer_client.identity.create(
            name="System",
            type="Organization",
            description="Default system identity for synchronization"
        )
        return system_identity['id']
    except Exception as e:
        print(f"⚠ Errore nella gestione dell'identità di default: {e}")
        return None

def test_connection(client, name):
    """Testa la connessione a un'istanza OpenCTI"""
    try:
        # Prova a recuperare alcune informazioni di base
        labels = client.label.list(first=1)
        return True, f"Connesso a {name}"
    except Exception as e:
        return False, f"Errore connessione a {name}: {e}"

def sync_labels():
    """Sincronizza tutte le labels"""
    try:
        print("\n🏷️  Sincronizzazione Labels...")
        labels = producer_client.label.list(get_all=True)
        print(f"✓ {len(labels)} labels trovate nel Producer.")

        success = 0
        failed = 0

        for index, label in enumerate(labels):
            try:
                # Controlla se la label esiste già
                existing_labels = consumer_client.label.list(
                    filters={"mode": "and", "filters": [{"key": "value", "values": [label["value"]]}], "filterGroups": []}
                )
                
                if existing_labels and len(existing_labels) > 0:
                    print(f"  ↪ Label già esistente: {label['value']}")
                    continue

                consumer_client.label.create(
                    value=label["value"],
                    color=label.get("color", "#FF6B6B")
                )

                success += 1
                print(f"✓ [{index+1}] Label sincronizzata: {label['value']}")
                time.sleep(0.1)

            except Exception as e:
                failed += 1
                print(f"✗ [{index+1}] Errore label '{label['value']}': {e}")

        print(f"✅ Labels: {success} create, {failed} fallite.")
        return True

    except Exception as e:
        print(f"✗ Errore sincronizzazione labels: {e}")
        return False

def sync_campaigns():
    """Sincronizza tutte le campagne"""
    try:
        print("\n📋 Sincronizzazione Campagne...")
        campaigns = producer_client.campaign.list(get_all=True)
        print(f"✓ {len(campaigns)} campagne trovate nel Producer.")

        success = 0
        failed = 0
        campaign_mapping = {}

        default_marking = get_or_create_default_marking()
        default_identity = get_or_create_default_identity()

        for index, campaign in enumerate(campaigns):
            try:
                # Controlla se la campagna esiste già
                existing_campaigns = consumer_client.campaign.list(
                    filters={"mode": "and", "filters": [{"key": "name", "values": [campaign["name"]]}], "filterGroups": []}
                )
                
                if existing_campaigns and len(existing_campaigns) > 0:
                    campaign_mapping[campaign['id']] = existing_campaigns[0]['id']
                    print(f"  ↪ Campagna già esistente: {campaign['name']}")
                    continue

                new_campaign = consumer_client.campaign.create(
                    name=campaign["name"],
                    description=campaign.get("description", ""),
                    createdBy=default_identity,
                    objectMarking=default_marking
                )

                campaign_mapping[campaign['id']] = new_campaign['id']
                success += 1
                print(f"✓ [{index+1}] Campagna sincronizzata: {campaign['name']}")
                time.sleep(0.1)

            except Exception as e:
                failed += 1
                print(f"✗ [{index+1}] Errore campagna '{campaign['name']}': {e}")

        print(f"✅ Campagne: {success} create, {failed} fallite.")
        return campaign_mapping

    except Exception as e:
        print(f"✗ Errore sincronizzazione campagne: {e}")
        return {}

def sync_identities():
    """Sincronizza tutte le identità (soggetti e oggetti delle triple)"""
    try:
        print("\n👤 Sincronizzazione Identità...")
        identities = producer_client.identity.list(get_all=True)
        print(f"✓ {len(identities)} identità trovate nel Producer.")

        success = 0
        failed = 0
        identity_mapping = {}

        default_marking = get_or_create_default_marking()
        default_identity = get_or_create_default_identity()

        for index, identity in enumerate(identities):
            try:
                # Salta l'identità di sistema
                if identity.get('name') == 'System':
                    identity_mapping[identity['id']] = default_identity
                    continue

                # Controlla se l'identità esiste già (basandosi su nome e contenuto tripla)
                filters = {"mode": "and", "filters": [{"key": "name", "values": [identity["name"]]}], "filterGroups": []}
                existing_identities = consumer_client.identity.list(filters=filters)
                
                # Verifica match più specifico se ha attributi personalizzati
                found_match = False
                if existing_identities:
                    for existing in existing_identities:
                        if (existing.get('x_triple_contenuto') == identity.get('x_triple_contenuto') and
                            existing.get('x_triple_ruolo') == identity.get('x_triple_ruolo')):
                            identity_mapping[identity['id']] = existing['id']
                            found_match = True
                            print(f"  ↪ Identità già esistente: {identity['name']} ({identity.get('x_triple_ruolo', 'N/A')})")
                            break
                
                if found_match:
                    continue

                # Recupera le labels
                label_ids = []
                for label in identity.get("objectLabel", []):
                    try:
                        existing_labels = consumer_client.label.list(
                            filters={"mode": "and", "filters": [{"key": "value", "values": [label["value"]]}], "filterGroups": []}
                        )
                        if existing_labels and len(existing_labels) > 0:
                            label_ids.append(existing_labels[0]['id'])
                    except:
                        pass

                new_identity = consumer_client.identity.create(
                    name=identity["name"],
                    type=identity.get("type", "Organization"),
                    description=identity.get("description", ""),
                    objectLabel=label_ids,
                    x_triple_contenuto=identity.get("x_triple_contenuto"),
                    x_triple_ruolo=identity.get("x_triple_ruolo"),
                    x_triple_completa=identity.get("x_triple_completa"),
                    x_triple_campagna=identity.get("x_triple_campagna"),
                    createdBy=default_identity,
                    objectMarking=default_marking
                )

                identity_mapping[identity['id']] = new_identity['id']
                success += 1
                print(f"✓ [{index+1}] Identità sincronizzata: {identity['name']} ({identity.get('x_triple_ruolo', 'N/A')})")
                time.sleep(0.1)

            except Exception as e:
                failed += 1
                print(f"✗ [{index+1}] Errore identità '{identity['name']}': {e}")

        print(f"✅ Identità: {success} create, {failed} fallite.")
        return identity_mapping

    except Exception as e:
        print(f"✗ Errore sincronizzazione identità: {e}")
        return {}

def sync_narratives():
    """Sincronizza tutte le narrative (verbi delle triple)"""
    try:
        print("\n📖 Sincronizzazione Narrative...")
        narratives = producer_client.narrative.list(get_all=True)
        print(f"✓ {len(narratives)} narrative trovate nel Producer.")

        success = 0
        failed = 0
        narrative_mapping = {}

        default_marking = get_or_create_default_marking()
        default_identity = get_or_create_default_identity()

        for index, narrative in enumerate(narratives):
            try:
                # Controlla se la narrative esiste già
                filters = {"mode": "and", "filters": [{"key": "name", "values": [narrative["name"]]}], "filterGroups": []}
                existing_narratives = consumer_client.narrative.list(filters=filters)
                
                # Verifica match più specifico
                found_match = False
                if existing_narratives:
                    for existing in existing_narratives:
                        if (existing.get('x_triple_contenuto') == narrative.get('x_triple_contenuto') and
                            existing.get('x_triple_ruolo') == 'verbo'):
                            narrative_mapping[narrative['id']] = existing['id']
                            found_match = True
                            print(f"  ↪ Narrative già esistente: {narrative['name']}")
                            break
                
                if found_match:
                    continue

                # Recupera le labels
                label_ids = []
                for label in narrative.get("objectLabel", []):
                    try:
                        existing_labels = consumer_client.label.list(
                            filters={"mode": "and", "filters": [{"key": "value", "values": [label["value"]]}], "filterGroups": []}
                        )
                        if existing_labels and len(existing_labels) > 0:
                            label_ids.append(existing_labels[0]['id'])
                    except:
                        pass

                new_narrative = consumer_client.narrative.create(
                    name=narrative["name"],
                    description=narrative.get("description", ""),
                    objectLabel=label_ids,
                    x_triple_contenuto=narrative.get("x_triple_contenuto"),
                    x_triple_ruolo=narrative.get("x_triple_ruolo"),
                    x_triple_completa=narrative.get("x_triple_completa"),
                    x_triple_campagna=narrative.get("x_triple_campagna"),
                    createdBy=default_identity,
                    objectMarking=default_marking
                )

                narrative_mapping[narrative['id']] = new_narrative['id']
                success += 1
                print(f"✓ [{index+1}] Narrative sincronizzata: {narrative['name']}")
                time.sleep(0.1)

            except Exception as e:
                failed += 1
                print(f"✗ [{index+1}] Errore narrative '{narrative['name']}': {e}")

        print(f"✅ Narrative: {success} create, {failed} fallite.")
        return narrative_mapping

    except Exception as e:
        print(f"✗ Errore sincronizzazione narrative: {e}")
        return {}

def sync_indicators(identity_mapping, narrative_mapping, campaign_mapping):
    """Sincronizza tutti gli indicatori (triple complete)"""
    try:
        print("\n🎯 Sincronizzazione Indicatori...")
        indicators = producer_client.indicator.list(get_all=True)
        print(f"✓ {len(indicators)} indicatori trovati nel Producer.")

        success = 0
        failed = 0
        indicator_mapping = {}

        default_marking = get_or_create_default_marking()
        default_identity = get_or_create_default_identity()

        for index, indicator in enumerate(indicators):
            try:
                # Controlla se l'indicatore esiste già
                filters = {"mode": "and", "filters": [{"key": "pattern", "values": [indicator["pattern"]]}], "filterGroups": []}
                existing_indicators = consumer_client.indicator.list(filters=filters)
                
                if existing_indicators and len(existing_indicators) > 0:
                    indicator_mapping[indicator['id']] = existing_indicators[0]['id']
                    print(f"  ↪ Indicatore già esistente: {indicator['name']}")
                    continue

                # Recupera le labels
                label_ids = []
                for label in indicator.get("objectLabel", []):
                    try:
                        existing_labels = consumer_client.label.list(
                            filters={"mode": "and", "filters": [{"key": "value", "values": [label["value"]]}], "filterGroups": []}
                        )
                        if existing_labels and len(existing_labels) > 0:
                            label_ids.append(existing_labels[0]['id'])
                    except:
                        pass

                new_indicator = consumer_client.indicator.create(
                    name=indicator["name"],
                    description=indicator.get("description", ""),
                    pattern=indicator["pattern"],
                    pattern_type=indicator["pattern_type"],
                    x_opencti_main_observable_type=indicator.get("x_opencti_main_observable_type", "Unknown"),
                    objectLabel=label_ids,
                    confidence=indicator.get("confidence", 50),
                    valid_from=indicator.get("valid_from"),
                    x_triple_completa=indicator.get("x_triple_completa"),
                    x_triple_campagna=indicator.get("x_triple_campagna"),
                    x_triple_type=indicator.get("x_triple_type"),
                    createdBy=default_identity,
                    objectMarking=default_marking
                )

                indicator_mapping[indicator['id']] = new_indicator['id']
                success += 1
                print(f"✓ [{index+1}] Indicatore sincronizzato: {indicator['name']}")
                time.sleep(0.1)

            except Exception as e:
                failed += 1
                print(f"✗ [{index+1}] Errore indicatore '{indicator['name']}': {e}")

        print(f"✅ Indicatori: {success} creati, {failed} falliti.")
        return indicator_mapping

    except Exception as e:
        print(f"✗ Errore sincronizzazione indicatori: {e}")
        return {}

def sync_relationships(identity_mapping, narrative_mapping, indicator_mapping, campaign_mapping):
    """Sincronizza tutte le relazioni STIX"""
    try:
        print("\n🔗 Sincronizzazione Relazioni...")
        relationships = producer_client.stix_core_relationship.list(get_all=True)
        print(f"✓ {len(relationships)} relazioni trovate nel Producer.")

        success = 0
        failed = 0
        skipped = 0

        # Combina tutti i mapping per la risoluzione degli ID
        all_mappings = {**identity_mapping, **narrative_mapping, **indicator_mapping, **campaign_mapping}

        for index, rel in enumerate(relationships):
            try:
                from_id = rel["from"]["id"]
                to_id = rel["to"]["id"]
                
                # Mappa gli ID dal producer al consumer
                new_from_id = all_mappings.get(from_id)
                new_to_id = all_mappings.get(to_id)
                
                if not new_from_id or not new_to_id:
                    print(f"  ↪ Saltando relazione {rel['relationship_type']}: ID non mappati ({from_id} -> {to_id})")
                    skipped += 1
                    continue

                # Controlla se la relazione esiste già
                filters = {
                    "mode": "and",
                    "filters": [
                        {"key": "fromId", "values": [new_from_id]},
                        {"key": "toId", "values": [new_to_id]},
                        {"key": "relationship_type", "values": [rel["relationship_type"]]}
                    ],
                    "filterGroups": []
                }
                existing_relationships = consumer_client.stix_core_relationship.list(filters=filters)
                
                if existing_relationships and len(existing_relationships) > 0:
                    print(f"  ↪ Relazione già esistente: {rel['relationship_type']}")
                    continue

                consumer_client.stix_core_relationship.create(
                    fromId=new_from_id,
                    toId=new_to_id,
                    relationship_type=rel["relationship_type"],
                    description=rel.get("description", ""),
                    confidence=rel.get("confidence", 50)
                )

                success += 1
                print(f"✓ [{index+1}] Relazione sincronizzata: {rel['relationship_type']}")
                time.sleep(0.1)

            except Exception as e:
                failed += 1
                print(f"✗ [{index+1}] Errore relazione '{rel.get('relationship_type', 'N/A')}': {e}")

        print(f"✅ Relazioni: {success} create, {failed} fallite, {skipped} saltate.")
        return True

    except Exception as e:
        print(f"✗ Errore sincronizzazione relazioni: {e}")
        return False

def full_sync():
    """Esegue la sincronizzazione completa di tutto il modello semantico"""
    print("🚀 SINCRONIZZAZIONE COMPLETA DEL MODELLO SEMANTICO")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Fase 1: Labels (prerequisito per tutti gli altri oggetti)
        if not sync_labels():
            print("✗ Errore critico: impossibile sincronizzare le labels")
            return False
        
        # Fase 2: Campagne
        campaign_mapping = sync_campaigns()
        
        # Fase 3: Identità (soggetti e oggetti)
        identity_mapping = sync_identities()
        
        # Fase 4: Narrative (verbi)
        narrative_mapping = sync_narratives()
        
        # Fase 5: Indicatori (triple complete)
        indicator_mapping = sync_indicators(identity_mapping, narrative_mapping, campaign_mapping)
        
        # Fase 6: Relazioni (struttura semantica)
        sync_relationships(identity_mapping, narrative_mapping, indicator_mapping, campaign_mapping)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 SINCRONIZZAZIONE COMPLETATA!")
        print(f"⏱️  Tempo totale: {duration:.2f} secondi")
        print("📊 Oggetti sincronizzati:")
        print(f"   • Labels: prerequisiti per categorizzazione")
        print(f"   • Campagne: {len(campaign_mapping)} mappate")
        print(f"   • Identità: {len(identity_mapping)} mappate (soggetti/oggetti)")
        print(f"   • Narrative: {len(narrative_mapping)} mappate (verbi)")
        print(f"   • Indicatori: {len(indicator_mapping)} mappati (triple complete)")
        print(f"   • Relazioni: struttura semantica completa")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Errore critico durante la sincronizzazione: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔄 Avvio sincronizzazione completa tra Producer e Consumer OpenCTI")
    print(f"📡 Producer: {PRODUCER_URL}")
    print(f"📨 Consumer: {CONSUMER_URL}")
    
    try:
        # Test connessioni
        print("\n🔍 Test connessioni...")
        
        producer_ok, producer_msg = test_connection(producer_client, "Producer")
        consumer_ok, consumer_msg = test_connection(consumer_client, "Consumer")
        
        print(f"✓ {producer_msg}")
        print(f"✓ {consumer_msg}")
        
        if not producer_ok or not consumer_ok:
            print("❌ Impossibile procedere: una o più connessioni non riuscite")
            exit(1)
        
        # Esegui sincronizzazione completa
        success = full_sync()
        
        if success:
            print("\n✅ Sincronizzazione completata con successo!")
            print("🎯 Il modello semantico completo è ora disponibile nel Consumer:")
            print("   • Triple semantiche come Indicatori")
            print("   • Soggetti/Oggetti come Identità")
            print("   • Verbi come Narrative")
            print("   • Campagne raggruppanti")
            print("   • Relazioni semantiche complete")
            print("   • Labels per categorizzazione")
        else:
            print("\n❌ Sincronizzazione fallita. Controllare i log per dettagli.")

    except Exception as e:
        print(f"✗ Errore fatale: {e}")
        traceback.print_exc()
        exit(1)