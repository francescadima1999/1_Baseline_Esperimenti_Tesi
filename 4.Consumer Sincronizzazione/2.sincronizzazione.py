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

def sync_indicators():
    try:
        print("✓ Recupero degli indicatori dal Producer in corso...")
        indicators = producer_client.indicator.list(get_all=True, first=10000)
        print(f"✓ {len(indicators)} indicatori trovati.")

        success = 0
        failed = 0

        for index, ind in enumerate(indicators):
            try:
                name = ind["name"]
                description = ind.get("description", "")
                pattern = ind["pattern"]
                pattern_type = ind["pattern_type"]
                observable_type = ind.get("x_opencti_main_observable_type", "Text")
                labels = [l["value"] for l in ind.get("objectLabel", [])]
                valid_from = ind.get("valid_from", None)

                consumer_client.indicator.create(
                    name=name,
                    description=description,
                    pattern=pattern,
                    pattern_type=pattern_type,
                    x_opencti_main_observable_type=observable_type,
                    valid_from=valid_from,
                    labels=labels
                )

                success += 1
                print(f"✓ [{index+1}] Indicatore sincronizzato: {name}")
                time.sleep(0.1)  # evita overload API

            except Exception as e:
                failed += 1
                print(f"✗ [{index+1}] Errore nel creare indicatore '{name}': {e}")

        print(f"\n✅ Sincronizzazione completata: {success} creati, {failed} falliti.")

    except Exception as e:
        print("✗ Errore durante la sincronizzazione:")
        traceback.print_exc()

if __name__ == "__main__":
    sync_indicators()
