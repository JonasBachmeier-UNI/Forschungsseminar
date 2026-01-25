import os
import re
from striprtf.striprtf import rtf_to_text

# --- KONFIGURATION ---
input_folder = "./Urteile"

# Drei Ausgabeordner
output_folder_kapitel5 = "./ergebnis_kapitel_5"
output_folder_unterbringung = "./ergebnis_unterbringung"
output_folder_undefined = "./ergebnis_undefined"

# Dateinamen für die Log-Listen
log_success_file = "log_erfolgreich_kapitel5.txt"
log_unterbringung_file = "log_erfolgreich_unterbringung.txt"
log_undefined_file = "log_undefined.txt"
log_error_file = "log_systemfehler.txt"

# Alle Ordner erstellen
for folder in [output_folder_kapitel5, output_folder_unterbringung, output_folder_undefined]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Listen zum Speichern der Ergebnisse für die Logs
liste_kapitel5 = []
liste_unterbringung = []
liste_undefined = []
liste_fehlerhaft = []

# --- SUCHMUSTER FÜR KAPITEL 5 (AKTUALISIERT) ---
# Erklärung des Regex:
# (?:^|\n)\s* -> Zeilenanfang und optionale Leerzeichen
# (?:V\.|5\.)                  -> Findet "V." oder "5."
# (?:\s*Strafzumessung)?       -> Findet optional das Wort "Strafzumessung" direkt danach
# \s* -> Ignoriert Leerzeichen nach der Überschrift
# (?P<inhalt>.*?)              -> Speichert den eigentlichen Text in die Gruppe "inhalt"
# (?=(?:^|\n)\s*(?:VI\.|6\.)|$) -> Stoppt bei "VI." oder "6." oder am Dateiende
pattern_kapitel5 = re.compile(
    r"(?:^|\n)\s*(?:V\.|5\.)(?:\s*Strafzumessung)?\s*(?P<inhalt>.*?)(?=(?:^|\n)\s*(?:VI\.|6\.)|$)", 
    re.DOTALL | re.IGNORECASE
)

# --- HELFER-FUNKTION ZUM SPEICHERN ---
def speichere_datei(ordner, basis_dateiname, suffix, inhalt):
    new_filename = basis_dateiname.replace(".rtf", suffix)
    output_path = os.path.join(ordner, new_filename)
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(inhalt)

# --- VERARBEITUNG ---
files = [f for f in os.listdir(input_folder) if f.endswith(".rtf")]
files.sort()
print(f"Starte Verarbeitung von {len(files)} Dateien...")

for filename in files:
    filepath = os.path.join(input_folder, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            rtf_content = file.read()
            # RTF zu Text konvertieren
            text_content = rtf_to_text(rtf_content)
            
            # --- PRÜFUNG 1: KAPITEL 5 SUCHEN ---
            match = pattern_kapitel5.search(text_content)
            
            if match:
                # Fall A: Kapitel 5 gefunden
                extracted_text = match.group("inhalt").strip()
                
                # Wir fügen eine saubere Überschrift hinzu, da wir die originale im Regex "verschluckt" haben könnten
                final_text = f"V. Strafzumessung (Extrakt aus {filename})\n\n{extracted_text}"
                
                speichere_datei(output_folder_kapitel5, filename, "_Kapitel5.txt", final_text)
                liste_kapitel5.append(filename)
                print(f"[KAPITEL 5]   {filename}")
            
            elif "unterbringung" in text_content.lower():
                # Fall B: Unterbringung gefunden (aber kein Kap 5) -> Ganzen Text speichern
                header = f"--- FALL MIT UNTERBRINGUNG (Kein Kap. 5 gefunden) ---\nOriginaldatei: {filename}\n\n"
                speichere_datei(output_folder_unterbringung, filename, "_Unterbringung.txt", header + text_content)
                
                liste_unterbringung.append(filename)
                print(f"[UNTERBRINGUNG] {filename}")
                
            else:
                # Fall C: Weder noch -> Undefined -> Ganzen Text speichern
                header = f"--- UNDEFINED (Weder Kap. 5 noch 'Unterbringung') ---\nOriginaldatei: {filename}\n\n"
                speichere_datei(output_folder_undefined, filename, "_Undefined.txt", header + text_content)
                
                liste_undefined.append(filename)
                print(f"[UNDEFINED]     {filename}")
                
    except Exception as e:
        # Fall D: Datei konnte technisch nicht gelesen werden
        error_msg = f"{filename} -> Systemfehler: {str(e)}"
        liste_fehlerhaft.append(error_msg)
        print(f"[ERROR]         {filename}")

# --- LOG-DATEIEN SCHREIBEN ---
def schreibe_log(dateiname, titel, liste):
    with open(dateiname, "w", encoding="utf-8") as f:
        f.write(f"--- {titel} ({len(liste)} Dateien) ---\n")
        for item in liste:
            f.write(f"{item}\n")

schreibe_log(log_success_file, "KAPITEL 5 EXTRAHIERT", liste_kapitel5)
schreibe_log(log_unterbringung_file, "UNTERBRINGUNG GEFUNDEN", liste_unterbringung)
schreibe_log(log_undefined_file, "UNDEFINED (Keine Zuordnung)", liste_undefined)
schreibe_log(log_error_file, "SYSTEMFEHLER (Nicht lesbar)", liste_fehlerhaft)

print("-" * 30)
print("Verarbeitung abgeschlossen!")
print(f"1. Kapitel 5:    {len(liste_kapitel5)} Dateien -> {output_folder_kapitel5}")
print(f"2. Unterbringung:{len(liste_unterbringung)} Dateien -> {output_folder_unterbringung}")
print(f"3. Undefined:    {len(liste_undefined)} Dateien -> {output_folder_undefined}")
if liste_fehlerhaft:
    print(f"4. Fehler:       {len(liste_fehlerhaft)} Dateien -> siehe {log_error_file}")