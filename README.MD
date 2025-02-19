## Erste Version der COLMAP PIPELINE

### Nutzeroberfläche und Abfolge der Abfragen an Nutzer
    1. Eingabe von video-Path + Frage nach Output_dir 
        - wenn User kein output_dir --> dann standard output_dir (Erstellen eines COLMAP-Ordnes, falls vorhanden unterornder erstellen (Zeitstemptel))


PIPELINE-Anteile
1. Video-Analyse/Bearbeitung
    - video_edit.py --> Zerlegen von Video in Bilder + Bilder auf Sharpness und Äquidistanz auswählen
    - mask_edit.py --> automatische Erstellung von Masken (Tresholding)
        --> Optional: Retinex oder weitere Farbhomogenisierungen verwenden

    FRAGEN: Wo werden bilder gespeichert? soll gesamte Pipeline in einem Ordner landen? 
2. COLMAP/SFM-PIPELINE
    - Parameter sollen in GUI oder durch eine Projectfile bestimmt werden 
        - Automatische Erstellung eines Project --> in der Parameter und Einstellungen angegeben sind
        - Ansonsten Durchführung der COLMAP Pipeline mit "vordefinierten" Standardwerten --> optimierung noch ausstehend, sollte aber erfolgen
        -
    - Speichern als sparse und dense in der normalen COLMAP Folderstruktur

3. Nachbearbeitung
    - Möglicherweise Filtern von Hintergründen, Ausstehenden Kanten etc.




Möglichkeiten der Visualisierung
- Dash-app 
    -COLMAP PIPELINE im Hintergrund
    - Visualisierung wie bei beispiel-app Möglichkeiten
    - Vielleicht wäre RERUN auch eine Option

- TKtinker/Qt app
    - leichteste und rudimentärste GUI ohne viel Vorwissen und Komplikationen

- Dash + RERUN-Viewer 
    --> aktueller Stand: SfM-Viewer in Rerun mit reading der BIN-Files und des master
