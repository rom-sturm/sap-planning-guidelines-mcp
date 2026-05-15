# SAC Planning Guidelines MCP Server — ChromaDB Edition

Lokaler, vollständig **offline**-fähiger MCP Server für Claude Desktop.
Verwandelt SAP Analytics Cloud Planning Richtlinien (PDFs) in eine semantische Wissensdatenbank mit RAG-angereicherter Code-Generierung.

---

## Dokumente hinzufügen

PDFs oder Textdateien können auf **zwei Wegen** indiziert werden:

**Option A — Datei direkt in den `documents/`-Ordner legen:**
```
SAP Planning MCP Server/
└── documents/
    └── meine_richtlinie.pdf   ← hier ablegen
```
Dann in Claude aufrufen:
```
add_document("meine_richtlinie.pdf", "SAC Planning Handbuch 2024")
```

**Option B — Absoluten Pfad übergeben (Datei liegt irgendwo auf dem PC):**
```
add_document("C:/Users/User/Downloads/SAC_Guide.pdf", "SAC Planning Guide")
```
Die Datei wird automatisch in den `documents/`-Ordner kopiert.

---

## Quick Start

### 1. Installieren

**Windows:**
```bat
setup.bat
```

**macOS / Linux:**
```bash
chmod +x setup.sh && ./setup.sh
```

Beim ersten Start wird `all-MiniLM-L6-v2` (~80 MB) einmalig von HuggingFace geladen und lokal gecacht.

### 2. Claude Desktop konfigurieren

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`  
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sac-planning-guidelines": {
      "command": "C:\\Pfad\\zu\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Pfad\\zu\\server.py"]
    }
  }
}
```

### 3. Claude Desktop neu starten → Fertig.

---

## Verfügbare Tools

| Tool | Beschreibung |
|---|---|
| `add_document` | PDF oder Textdatei indizieren (absoluter Pfad oder Dateiname in `documents/`) |
| `list_documents` | Alle Dokumente + Stale-Status anzeigen |
| `remove_document` | Dokument per Dateiname entfernen |
| `reindex_all` | Geänderte Dateien automatisch neu indizieren |
| `search_documents` | Hybride semantisch + keyword Suche |
| `get_document_section` | Zeichenbereich aus Dokument lesen |
| `sac_planning_snippet` | SAC Planning Template + RAG-Kontext |
| `check_planning_config` | Code-Review + passende Richtlinienabschnitte |

### Snippet-Typen (`sac_planning_snippet`)

`data_action` · `allocation` · `currency_conversion` · `version_management` · `planning_sequence`

---

## Verzeichnisstruktur

```
sap-planning-guidelines-mcp/
├── server.py              ← MCP Server
├── requirements.txt
├── setup.bat              ← Windows Setup
├── setup.sh               ← macOS / Linux Setup
├── documents/             ← PDFs hier ablegen
│   └── guide.cache.txt    ← Plain-Text-Cache (auto-generiert)
├── doc_index.json         ← Checksum + Metadaten (auto-generiert)
└── chroma_store/          ← ChromaDB Vektorindex (auto-generiert)
```
