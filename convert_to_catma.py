import json
import re
import sys

def create_tei_xml(text_file, json_file, output_file):
    # 1. Dateien laden
    with open(text_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data.get("annotations", [])
    
    # 2. Annotationen sortieren (wichtig, um den Text von hinten nach vorne zu bearbeiten,
    # damit sich die Indizes nicht verschieben)
    # Wir müssen erst die Startpositionen finden
    anns_with_pos = []
    
    for ann in annotations:
        quote = ann.get("quote", "").strip()
        tag = ann.get("tag", "Unknown")
        category = ann.get("category", "General")
        
        if not quote:
            continue
            
        # Finde das Zitat im Text
        # Hinweis: Wir nehmen das erste Vorkommen. Für komplexe Texte müsste man intelligenter suchen.
        start_index = original_text.find(quote)
        
        if start_index == -1:
            print(f"WARNUNG: Zitat nicht gefunden: '{quote[:20]}...'")
            continue
            
        end_index = start_index + len(quote)
        anns_with_pos.append({
            "start": start_index,
            "end": end_index,
            "tag": tag,
            "category": category,
            "quote": quote
        })

    # Sortieren: Von hinten nach vorne, damit wir beim Einfügen die Indizes vorne nicht kaputt machen
    anns_with_pos.sort(key=lambda x: x["start"], reverse=True)

    # 3. XML-Tags in den Text einfügen
    # Wir nutzen einfaches TEI: <seg type="Kategorie_Tag">Zitat</seg>
    
    xml_text = original_text
    
    for ann in anns_with_pos:
        # XML einfügen
        before = xml_text[:ann["start"]]
        content = xml_text[ann["start"]:ann["end"]]
        after = xml_text[ann["end"]:]
        
        # Achtung vor Überlappungen! (Einfache Lösung: Nur taggen, wenn keine Tags im Weg sind)
        if ">" in content or "<" in content:
            print(f"WARNUNG: Überlappung bei '{content[:20]}...', wird übersprungen.")
            continue
            
        # Wir erstellen zwei verschachtelte Tags: Außen Kategorie, Innen der spezifische Tag
        # So sind beide Informationen als eigene Tags in CATMA verfügbar
        xml_text = f'{before}<seg type="{ann["category"]}"><seg type="{ann["tag"]}">{content}</seg></seg>{after}'

    # 4. TEI Header drumherum bauen
    tei_output = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
      <body>
         <p>{xml_text}</p>
      </body>
  </text>
</TEI>"""

    # Speichern
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(tei_output)
    print(f"Erfolg! Datei gespeichert unter: {output_file}")

# parameter bei aufrufen der python datei angeben: z.B. python convert_to_catma.py urteil.txt annotation.json import_for_catma


if len(sys.argv) != 4:
    print("Falsche Anzahl an Argumenten.")
    print("Nutzen Sie: python convert_to_catma.py <text_datei> <json_datei> <output_datei_name>")
    sys.exit(1)

try:
    create_tei_xml(sys.argv[1], sys.argv[2], sys.argv[3] + ".xml")
except FileNotFoundError:
    print("Fehler: Bitte stellen Sie sicher, dass die Dateien existieren.")