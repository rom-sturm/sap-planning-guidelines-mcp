# SAP Planning Guidelines MCP Server — ChromaDB Edition

Lokaler, vollständig **offline**-fähiger MCP Server für Claude Desktop.
Verwandelt SAP-Planungsrichtlinien (PDFs) in eine semantische Wissensdatenbank mit RAG-angereicherter Code-Generierung.

## Unterstützte Planungsbereiche

| Bereich | Beschreibung |
|---|---|
| **SAC Planning** | SAP Analytics Cloud — Data Actions, Allocations, Versionsverwaltung |
| **SAP IBP** | Integrated Business Planning — Supply Chain, Schlüsselzahlen, Alerts |
| **SAP BPC** | Business Planning & Consolidation — Script Logic, Fox Formulas, ETL |
| **S/4HANA Planning** | PP/FI/CO-Planung — MRP, CDS Views, Profit Center Planning |

## Features

| Feature | Details |
|---|---|
| **Semantische Suche** | ChromaDB + `all-MiniLM-L6-v2` — findet *Bedeutung*, nicht nur Stichworte |
| **Hybrid-Suche** | 70 % semantisch + 30 % keyword |
| **Auto-Chunking** | 500-Zeichen-Chunks mit 100-Zeichen-Overlap |
| **Re-Index-Erkennung** | MD5-Checksum pro Datei — `reindex_all` aktualisiert nur geänderte Dokumente |
| **RAG-Snippets** | Code-Templates für SAC/IBP/BPC/S4 mit passenden Richtlinienabschnitten |
| **Code-Review** | Best-Practice-Checks + relevante Richtlinienpassagen per RAG |
| **100 % offline** | Keine Cloud-Aufrufe. Embeddings lokal via sentence-transformers. |

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

Das Setup-Script erstellt das virtuelle Environment, installiert alle Abhängigkeiten und gibt den fertigen Konfigurationsblock für Claude Desktop aus.

Beim ersten Start wird `all-MiniLM-L6-v2` (~80 MB) einmalig von HuggingFace geladen und lokal gecacht.

### 2. Claude Desktop konfigurieren

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sap-planning-guidelines": {
      "command": "/absoluter/pfad/zu/.venv/bin/python",
      "args": ["/absoluter/pfad/zu/server.py"]
    }
  }
}
```

### 3. Claude Desktop neu starten → Fertig.

---

## Verfügbare Tools

| Tool | Beschreibung |
|---|---|
| `add_document` | PDF oder Textdatei indizieren |
| `list_documents` | Alle Dokumente + Stale-Status anzeigen |
| `remove_document` | Dokument per Dateiname entfernen |
| `reindex_all` | Geänderte Dateien automatisch neu indizieren |
| `search_documents` | Hybride semantisch + keyword Suche |
| `get_document_section` | Zeichenbereich aus Dokument lesen |
| `sac_planning_snippet` | SAC Planning Template + RAG |
| `ibp_snippet` | SAP IBP Template + RAG |
| `bpc_snippet` | SAP BPC Template + RAG |
| `s4_planning_snippet` | S/4HANA Planning Template + RAG |
| `check_planning_config` | Code-Review + passende Richtlinienabschnitte |

### Snippet-Typen

**SAC Planning:** `data_action` · `allocation` · `currency_conversion` · `version_management` · `planning_sequence`

**SAP IBP:** `key_figure_calculation` · `supply_heuristic` · `alert_definition` · `macros_excel`

**SAP BPC:** `script_logic` · `fox_formula` · `validation_rule` · `etl_transformation`

**S/4HANA Planning:** `mrp_configuration` · `fi_planning_cds` · `pp_planned_order` · `profit_center_planning`

---

## Beispiel-Workflow

```
1. Richtlinien hinzufügen:
   add_document("/pfad/zu/SAC_Planning_Guide.pdf", "SAC Planning Best Practices")
   add_document("/pfad/zu/IBP_Reference.pdf", "SAP IBP Konfigurationshandbuch")

2. Semantisch suchen:
   search_documents("Top-Down Verteilung nach Kostenstelle")

3. RAG-angereicherten Snippet generieren:
   sac_planning_snippet(snippet_type="allocation", context="Revenue nach Headcount verteilen")

4. Code prüfen:
   check_planning_config(area="bpc", code="*XDIM_MEMBERSET CATEGORY = BAS(FORECAST) ...")

5. Dokument geändert? Einfach ausführen:
   reindex_all
```

---

## Verzeichnisstruktur

```
sap-planning-guidelines-mcp/
├── server.py              ← MCP Server (einzelne Datei)
├── requirements.txt
├── setup.bat              ← Windows Setup
├── setup.sh               ← macOS / Linux Setup
├── doc_index.json         ← Checksum + Metadaten (auto-generiert)
├── documents/             ← PDFs hier ablegen
│   └── guide.cache.txt    ← Plain-Text-Cache (auto-generiert)
└── chroma_store/          ← ChromaDB Vektorindex (auto-generiert)
```

---

## Tuning

In `server.py` anpassbar:

```python
CHUNK_SIZE    = 500   # Zeichen pro Chunk
CHUNK_OVERLAP = 100   # Überlappung zwischen Chunks

# Hybrid-Search-Gewichtung:
final_score = 0.70 * sem_score + 0.30 * kw_score
```
