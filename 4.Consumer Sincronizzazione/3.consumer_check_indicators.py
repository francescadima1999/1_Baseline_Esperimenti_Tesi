from pycti import OpenCTIApiClient
import traceback
import requests
import time
import sys

# Connessione a OpenCTI Consumer
client = OpenCTIApiClient("http://localhost:8080", "f5feb3a4-e152-21f2-c4f5-181138731d2e")


def check_server_connection(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Connessione al server OpenCTI riuscita.")
            return True
        else:
            print(f"Errore di connessione al server OpenCTI: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Errore durante la connessione al server OpenCTI: {e}")
        return False

from openpyxl import Workbook
import re

def get_disinformation_indicators():
    if not check_server_connection("http://localhost:8080"):
        print("Impossibile connettersi al server OpenCTI.")
        return

    try:
        print("Cerco indicatori con ...")

        indicators = client.indicator.list( first=50)

        print(f"Totale indicatori trovati: {len(indicators)}")

        # Creazione workbook Excel
        wb = Workbook()
        ws = wb.active
        ws.title = "Indicatori"
        ws.append(["Tupla", "ID Articolo", "Campagna"])

        def parse_indicator(name):
            pattern = r"^(.*?) - (ID Articolo: \d+ -) (Campagna: .+)$"
            match = re.match(pattern, name)
            if match:
                return match.group(1).strip() + " -", match.group(2).strip(), match.group(3).strip()
            else:
                return name, "", ""

        for indicator in indicators:
            name = indicator.get('name', 'N/A')
            pattern = indicator.get('pattern', 'N/A')
            labels = [l['value'] for l in indicator.get('objectLabel', [])]
            print(f"Indicatore: {name}")
            print(f"Etichette: {labels}")
            print(f"Pattern: {pattern}")
            print("-----")

            tupla, id_articolo, campagna = parse_indicator(name)
            ws.append([tupla, id_articolo, campagna])

        # Salvataggio file Excel
        filename = "indicatori_export.xlsx"
        wb.save(filename)
        print(f"File Excel '{filename}' creato con successo.")

    except Exception as e:
        print("Errore durante il recupero degli indicatori:")
        traceback.print_exc()

get_disinformation_indicators()