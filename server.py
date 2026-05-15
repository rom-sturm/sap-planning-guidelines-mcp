#!/usr/bin/env python3
"""
SAP Planning Guidelines MCP Server — ChromaDB Edition
- Wissensdatenbank für SAP-Planungsthemen: SAC Planning, IBP, BPC, S/4HANA PP/FI-Planung
- Local sentence-transformers embeddings (vollständig offline)
- Hybrid search: semantisch (ChromaDB) + keyword (BM25-style)
- Auto-Chunking mit Overlap
- Re-Index-Erkennung via MD5-Checksum
- RAG-angereicherte Code-/Konfigurations-Snippets für SAP Planning
"""

import json
import os
import re
import hashlib
import shutil
import asyncio
import textwrap
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ─── Optional PDF backend ──────────────────────────────────────────────────────
try:
    import fitz
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        import pdfplumber
        PDF_BACKEND = "pdfplumber"
    except ImportError:
        PDF_BACKEND = None

# ─── Lazy-loaded heavy deps ────────────────────────────────────────────────────
_chroma_client = None
_collection    = None
_embed_fn      = None

def _get_embedding_fn():
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embed_fn

def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name="sap_planning_guidelines",
            embedding_function=_get_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DOCS_DIR   = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_store"
INDEX_FILE = BASE_DIR / "doc_index.json"
for d in [DOCS_DIR, CHROMA_DIR]:
    d.mkdir(exist_ok=True)

# ─── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

def chunk_text(text: str, doc_name: str) -> list[dict]:
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(text), step)):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) < 80:
            continue
        chunks.append({
            "id":        f"{doc_name}__chunk_{i}",
            "text":      chunk,
            "doc_name":  doc_name,
            "chunk_idx": i,
            "start_char": start,
        })
        if end >= len(text):
            break
    return chunks

# ─── Index helpers ─────────────────────────────────────────────────────────────

def load_index() -> dict:
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {}

def save_index(index: dict):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

def file_checksum(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def extract_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return path.read_text(errors="replace")
    if PDF_BACKEND == "pymupdf":
        doc = fitz.open(str(path))
        return "\n\n".join(page.get_text() for page in doc)
    elif PDF_BACKEND == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    return "[PDF-Extraktion nicht verfügbar — installiere: pip install pymupdf]"

def index_document(dest: Path, description: str, index: dict) -> dict:
    text = extract_text(dest)
    cache = DOCS_DIR / (dest.stem + ".cache.txt")
    cache.write_text(text, encoding="utf-8")
    checksum = file_checksum(dest)

    chunks = chunk_text(text, dest.name)
    col = _get_collection()

    existing = col.get(where={"doc_name": dest.name})
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    BATCH = 64
    for b in range(0, len(chunks), BATCH):
        batch = chunks[b:b+BATCH]
        col.upsert(
            ids       = [c["id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [{"doc_name": c["doc_name"],
                          "chunk_idx": c["chunk_idx"],
                          "start_char": c["start_char"]} for c in batch],
        )

    entry = {
        "description": description,
        "size":        dest.stat().st_size,
        "chars":       len(text),
        "chunks":      len(chunks),
        "checksum":    checksum,
        "cache":       str(cache),
    }
    index[dest.name] = entry
    save_index(index)
    return entry

def check_stale_documents(index: dict) -> list[str]:
    stale = []
    for name, meta in index.items():
        path = DOCS_DIR / name
        if path.exists() and file_checksum(path) != meta.get("checksum", ""):
            stale.append(name)
    return stale

# ─── Hybrid search ─────────────────────────────────────────────────────────────

def keyword_score(text: str, query: str) -> float:
    tokens = re.findall(r'\w+', query.lower())
    if not tokens:
        return 0.0
    t = text.lower()
    return sum(t.count(tok) for tok in tokens) / len(tokens)

def hybrid_search(query: str, n_results: int = 5, doc_filter: Optional[str] = None) -> list[dict]:
    col = _get_collection()
    where = {"doc_name": doc_filter} if doc_filter else None

    semantic_n = min(n_results * 3, col.count() or 1)
    kwargs = dict(query_texts=[query], n_results=semantic_n)
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)

    combined = []
    for doc_id, text, meta, dist in zip(
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0]
    ):
        sem_score  = 1.0 - float(dist)
        kw_score   = keyword_score(text, query)
        final_score = 0.70 * sem_score + 0.30 * min(kw_score / 5.0, 1.0)
        combined.append({
            "id":        doc_id,
            "text":      text,
            "doc_name":  meta["doc_name"],
            "chunk_idx": meta["chunk_idx"],
            "sem_score": round(sem_score, 3),
            "kw_score":  round(kw_score, 3),
            "score":     round(final_score, 3),
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:n_results]

def build_rag_context(query: str, n: int = 4) -> str:
    try:
        hits = hybrid_search(query, n_results=n)
        if not hits:
            return ""
        parts = []
        for h in hits:
            parts.append(
                f"[{h['doc_name']} | chunk {h['chunk_idx']} | score {h['score']}]\n"
                f"{h['text']}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception:
        return ""

# ─── SAC Planning Templates ────────────────────────────────────────────────────

SAC_TEMPLATES = {
    "data_action": textwrap.dedent("""\
        // SAC Planning: Data Action — Werte auf Zielversion kopieren
        // Trigger: manuell oder per Story-Button

        MEMBERSET [d/Version] = ("Actual")
        MEMBERSET [d/Category] = ("Forecast")

        // Quell-Daten lesen
        DATA([d/Version] = "Actual") -> [d/Version] = "Plan"

        // Optional: Faktor anwenden
        // [d/Version] = "Plan" * 1.05
        """),

    "allocation": textwrap.dedent("""\
        // SAC Planning: Allocation — Top-Down-Verteilung nach Schlüssel
        // Voraussetzung: Schlüsselfigur (z.B. Headcount) muss befüllt sein

        ALLOCATION
          SOURCE      = [d/Version] = "Plan", [d/Measure] = "Revenue_Total"
          TARGET      = [d/Version] = "Plan", [d/Measure] = "Revenue_Allocated"
          DRIVER      = [d/Measure] = "Headcount"
          BREAKDOWN   = [d/CostCenter]
        END ALLOCATION
        """),

    "currency_conversion": textwrap.dedent("""\
        // SAC Planning: Währungsumrechnung in Data Action
        // Voraussetzung: Exchange Rate-Tabelle im Modell konfiguriert

        DATA([d/Measure] = "Revenue_LC") ->
            CURRENCY_CONVERT(
                SOURCE_CURRENCY = [d/Currency],
                TARGET_CURRENCY = "EUR",
                RATE_TYPE       = "AVG",
                TIME            = [d/Date]
            ) -> [d/Measure] = "Revenue_EUR"
        """),

    "version_management": textwrap.dedent("""\
        // SAC Planning: Versionsverwaltung — Best Practices

        // 1. Planversionen immer mit Kategorie kennzeichnen:
        //    Actual | Budget | Forecast | Rolling Forecast

        // 2. Öffentliche Version sperren nach Freigabe:
        //    Administration → Versionen → Status = "Gesperrt"

        // 3. Private Versionen für Draft-Planung nutzen,
        //    dann in öffentliche Version publizieren.

        // Beispiel: Version kopieren via Data Action
        COPY
          FROM [d/Version] = "Budget_2024"
          TO   [d/Version] = "Forecast_Q2_2024"
          FILTER [d/Date] >= "2024-04" AND [d/Date] <= "2024-12"
        END COPY
        """),

    "planning_sequence": textwrap.dedent("""\
        // SAC Planning: Planning Sequence — mehrere Data Actions verketten
        // Reihenfolge:
        //   1. Daten vorbereiten / cleanen
        //   2. Verteilung / Allocation
        //   3. Währungsumrechnung
        //   4. Aggregation / Summation

        SEQUENCE "Monthly_Forecast_Refresh"
          STEP 1: Data Action "DA_Copy_Actuals"
          STEP 2: Data Action "DA_Allocate_Revenue"
          STEP 3: Data Action "DA_Currency_Convert"
          STEP 4: Data Action "DA_Aggregate_KPIs"
        END SEQUENCE
        """),
}

# ─── IBP Templates ─────────────────────────────────────────────────────────────

IBP_TEMPLATES = {
    "key_figure_calculation": textwrap.dedent("""\
        /* SAP IBP: Schlüsselzahl-Berechnung (Calculated Key Figure)
           Pfad: Master Data → Key Figures → Calculated */

        -- Beispiel: Forecast Accuracy (FA)
        -- FA = 1 - ABS(Statistical Forecast - Consensus Demand) / Consensus Demand

        CASE
          WHEN [CONSENSUSDEMANDQTY] = 0 THEN NULL
          ELSE 1 - ABS([STATISTICALFORECASTQTY] - [CONSENSUSDEMANDQTY])
                   / [CONSENSUSDEMANDQTY]
        END
        """),

    "supply_heuristic": textwrap.dedent("""\
        /* SAP IBP: Supply Heuristic — Konfigurationsmuster
           Pfad: Supply Planning → Heuristic Run */

        -- Planungsparameter (Beispielwerte):
        -- Planungshorizont:       12 Monate
        -- Zeitbucket:             Woche
        -- Sicherheitsbestand:     2 Wochen Coverage
        -- Lieferzeit (Lead Time): 4 Wochen
        -- Losgrößenverfahren:     Lot-for-Lot

        -- Empfohlene Reihenfolge:
        --   1. Demand Planning (stat. Forecast + Konsensus)
        --   2. Inventory Optimization (Safety Stock)
        --   3. Supply Heuristic / Optimization
        --   4. Deployment & Transportation Load Building
        """),

    "alert_definition": textwrap.dedent("""\
        /* SAP IBP: Alert-Definition — Ausnahmeüberwachung
           Pfad: Alert Management → Alert Types */

        -- Beispiel: Unterdeckungsalert
        -- Bedingung: Projected Stock < Safety Stock

        ALERT TYPE: "Stock Shortage Risk"
          CONDITION:  [PROJECTEDSTOCKQTY] < [SAFETYSTOCKQTY]
          SEVERITY:   High
          NOTIFY:     Supply Planner, Demand Planner
          THRESHOLD:  10 %  -- nur auslösen wenn Abweichung > 10 %
          HORIZON:    8 Wochen
        """),

    "macros_excel": textwrap.dedent("""\
        ' SAP IBP Excel Add-In: VBA-Makro für automatischen Refresh
        ' Voraussetzung: SAP IBP Excel Add-In installiert

        Sub RefreshIBPData()
            Dim oIBP As Object
            On Error GoTo ErrHandler

            ' IBP Add-In Objekt referenzieren
            Set oIBP = Application.COMAddIns("SAPBPCExcelClient.Connect").Object

            ' Daten aktualisieren
            oIBP.Refresh

            MsgBox "IBP-Daten erfolgreich aktualisiert.", vbInformation
            Exit Sub

        ErrHandler:
            MsgBox "Fehler beim Refresh: " & Err.Description, vbCritical
        End Sub
        """),
}

# ─── BPC Templates ─────────────────────────────────────────────────────────────

BPC_TEMPLATES = {
    "script_logic": textwrap.dedent("""\
        *--- SAP BPC Script Logic: Top-Down-Verteilung ---*
        *  Pfad: Administration → Rules → Script Logic

        *XDIM_MEMBERSET CATEGORY = BAS(FORECAST)
        *XDIM_MEMBERSET TIME = %TIME_SET%
        *XDIM_MEMBERSET ACCOUNT = BAS(REVENUE)

        *WHEN COSTCENTER
          *IS "TOTAL"
            *REC(EXPRESSION = %VALUE% / [COSTCENTER].[HEADCOUNT] \
                              * [COSTCENTER].[TOTAL_HEADCOUNT], \
                 COSTCENTER = %COSTCENTER_MEMBERS%)
        *ENDWHEN
        """),

    "fox_formula": textwrap.dedent("""\
        *--- SAP BPC Fox Formula: Wachstumsrate anwenden ---*

        DATA lv_growth TYPE f.
        lv_growth = 0.05.  " 5 % Wachstum

        *FOR %ENTITY% = BAS(COMPANY)
          *REC(FACTOR = 1 + lv_growth, \
               CATEGORY = "BUDGET", \
               TIME = %NEXT_YEAR%)
        *NEXT
        """),

    "validation_rule": textwrap.dedent("""\
        *--- SAP BPC Validierungsregel ---*
        *  Sicherstellen dass Summe der Kostenstellen = Gesamtkosten

        *VALIDATION
          *XDIM_MEMBERSET ACCOUNT = "TOTAL_COSTS"
          *FORMULA
            [COSTCENTER].[ALL_COSTCENTERS_SUM] = [COSTCENTER].[TOTAL]
          *MESSAGE "Kostenstellensumme stimmt nicht mit Gesamtkosten überein"
        *END_VALIDATION
        """),

    "etl_transformation": textwrap.dedent("""\
        *--- SAP BPC Data Manager: Transformation-Datei (*.xls) ---*
        *  Spalten: EXTERNAL_FIELD → BPC_DIMENSION

        *OPTIONS
          FORMAT = DELIMITED
          DELIMITER = ;
          HEADER = YES
          SKIPROWS = 0

        *MAPPING
          BUKRS    → ENTITY
          GJAHR    → TIME      (TRANSFORM: "20" & LEFT(%VALUE%,2) & "." & RIGHT(%VALUE%,2))
          KSTAR    → ACCOUNT
          WKGBTR   → SIGNEDDATA
          KOKRS    → COSTCENTER
        """),
}

# ─── S/4HANA Planning Templates ───────────────────────────────────────────────

S4_TEMPLATES = {
    "mrp_configuration": textwrap.dedent("""\
        *--- S/4HANA MRP: Planungsparameter (MRP-Sicht im Materialstamm) ---*

        * Transaktion: MM02 → Dispositionsansichten

        * MRP-Verfahren:        PD  (Plangesteuerte Disposition)
        * Losgrößenverfahren:   EX  (Exakte Losgröße / Lot-for-Lot)
        * Bestellpolitik:       V   (Verbrauchsgesteuert) ODER P (Plangesteuert)
        * Planungszeitraum:     90 Tage
        * Sicherheitsbestand:   100 Stück (statisch) ODER dynamisch via Safety Stock Planning
        * Meldebestand:         50 Stück
        * Wiederbeschaffungszeit: 14 Tage

        * ABAP: MRP-Lauf per BAdI erweitern
        INTERFACE if_ex_md_change_mrp_data
          METHOD change_mrp_data.
            " Planungsparameter dynamisch anpassen
          ENDMETHOD.
        ENDINTERFACE.
        """),

    "fi_planning_cds": textwrap.dedent("""\
        -- S/4HANA FI-Planung: CDS View für Planungsberichte
        -- Basis: Tabelle FAGLFLEXP (Planung Hauptbuch)

        @AbapCatalog.sqlViewName: 'ZV_FI_PLAN'
        @Analytics.dataCategory:  #CUBE
        define view ZI_FI_PlanData
          as select from faglflexp
        {
          key rbukrs    as CompanyCode,
          key ryear     as FiscalYear,
          key poper     as FiscalPeriod,
          key racct     as GLAccount,
          key rcntr     as CostCenter,
              hsl       as AmountLC,
              tsl       as AmountGC,
              hwaer     as LocalCurrency,
              versn     as PlanVersion
        }
        where rwtype = '0'   -- Planwerte
          and rldnr  = '0L'  -- Hauptbuch-Ledger
        """),

    "pp_planned_order": textwrap.dedent("""\
        * S/4HANA PP: Planauftrag per ABAP anlegen
        * Transaktion MD11 / BAPI: BAPI_PLANNEDORDER_CREATE

        DATA: ls_order  TYPE bapi_pp_planned_order,
              ls_return TYPE bapiret2.

        ls_order-material    = 'FG-1000'.
        ls_order-plant       = '1000'.
        ls_order-order_type  = 'LA'.           " LA = Planauftrag
        ls_order-quantity    = '100'.
        ls_order-basic_end   = '20241231'.
        ls_order-order_start = '20241201'.

        CALL FUNCTION 'BAPI_PLANNEDORDER_CREATE'
          EXPORTING
            planned_order = ls_order
          IMPORTING
            return        = ls_return.

        IF ls_return-type = 'E'.
          " Fehlerbehandlung
        ELSE.
          CALL FUNCTION 'BAPI_TRANSACTION_COMMIT'.
        ENDIF.
        """),

    "profit_center_planning": textwrap.dedent("""\
        -- S/4HANA Profit-Center-Planung: SQL-Abfrage Ist vs. Plan
        -- Quelle: ACDOCA (Ist) + FAGLFLEXP (Plan)

        SELECT
            a.prctr      AS ProfitCenter,
            a.gjahr      AS FiscalYear,
            a.poper      AS Period,
            SUM(a.hsl)   AS Actual_LC,
            p.plan_amt   AS Plan_LC,
            SUM(a.hsl) - p.plan_amt AS Variance
        FROM acdoca AS a
        JOIN (
            SELECT rcntr, ryear, poper, SUM(hsl) AS plan_amt
            FROM faglflexp
            WHERE versn = '001' AND rwtype = '0'
            GROUP BY rcntr, ryear, poper
        ) AS p
          ON a.prctr = p.rcntr
         AND a.gjahr = p.ryear
         AND a.poper = p.poper
        WHERE a.rbukrs = '1000'
          AND a.gjahr  = '2024'
        GROUP BY a.prctr, a.gjahr, a.poper, p.plan_amt
        ORDER BY a.prctr, a.poper;
        """),
}

# ─── Best-practice checks ──────────────────────────────────────────────────────

def check_sac_planning(config: str) -> tuple[list, list]:
    issues, tips = [], []
    cu = config.upper()

    if "SELECT *" in cu or "ALL MEMBERS" in cu:
        issues.append("Zu breite Memberset-Selektion — schränke auf relevante Mitglieder ein (Performance).")
    if "COPY" in cu and "FILTER" not in cu:
        tips.append("Versions-COPY ohne FILTER — prüfe ob der gesamte Planungshorizont kopiert werden soll.")
    if "PUBLIC" in cu and "LOCK" not in cu and "GESPERR" not in cu:
        tips.append("Öffentliche Version erkannt — nach Freigabe sperren (Administration → Versionen → Gesperrt).")
    if "CURRENCY_CONVERT" in cu and "RATE_TYPE" not in cu:
        issues.append("CURRENCY_CONVERT ohne RATE_TYPE — gib explizit AVG, END oder HIST an.")
    if "ALLOCATION" in cu and "DRIVER" not in cu:
        issues.append("ALLOCATION ohne DRIVER — Treiber-Schlüsselzahl angeben, sonst gleichmäßige Verteilung.")
    if not issues and not tips:
        tips.append("Keine offensichtlichen Probleme gefunden.")
    return issues, tips

def check_ibp_config(config: str) -> tuple[list, list]:
    issues, tips = [], []
    cu = config.upper()

    if "SAFETY STOCK" not in cu and "SAFETYSTOCKQTY" not in cu:
        tips.append("Sicherheitsbestand nicht referenziert — prüfe ob Safety Stock Planning aktiviert ist.")
    if "LEAD TIME" not in cu and "LEADTIME" not in cu:
        tips.append("Keine Lieferzeitangabe gefunden — stelle sicher, dass die Lead Time im Materialstamm gepflegt ist.")
    if "ALERT" not in cu:
        tips.append("Keine Alert-Logik erkannt — empfehle Ausnahmeüberwachung für Unter-/Überdeckungen zu aktivieren.")
    if "HORIZON" not in cu and "HORIZONT" not in cu:
        issues.append("Kein Planungshorizont definiert — IBP-Läufe ohne Horizont können Gesamtdaten verarbeiten.")
    if not issues and not tips:
        tips.append("Konfiguration sieht plausibel aus.")
    return issues, tips

def check_bpc_logic(code: str) -> tuple[list, list]:
    issues, tips = [], []
    cu = code.upper()

    if "*XDIM_MEMBERSET" not in cu:
        issues.append("Kein *XDIM_MEMBERSET gesetzt — Script Logic ohne Memberset verarbeitet alle Daten (Performance-Risiko).")
    if "COMMIT" not in cu and "*REC" in cu:
        tips.append("*REC ohne expliziten COMMIT — prüfe ob automatischer COMMIT nach Script ausgeführt wird.")
    if "SIGNEDDATA" not in cu and "HSL" not in cu and "WKGBTR" not in cu:
        tips.append("Kein Betragsfeld erkannt — stelle sicher das richtige Betragsfeld (SIGNEDDATA / HSL) zu verwenden.")
    if "*WHEN" in cu and "*ENDWHEN" not in cu:
        issues.append("*WHEN ohne *ENDWHEN — unvollständige Bedingungsstruktur.")
    if not issues and not tips:
        tips.append("Keine offensichtlichen Probleme in der Script Logic gefunden.")
    return issues, tips

def check_s4_planning(code: str) -> tuple[list, list]:
    issues, tips = [], []
    cu = code.upper()

    if "BAPI_TRANSACTION_COMMIT" not in cu and ("BAPI_" in cu or "CALL FUNCTION" in cu):
        issues.append("BAPI-Aufruf ohne BAPI_TRANSACTION_COMMIT — Daten werden nicht dauerhaft gespeichert.")
    if "MRP" in cu and "PLANT" not in cu and "WERK" not in cu:
        tips.append("MRP-Bezug ohne Werk (PLANT/WERKS) — MRP ist immer werksbezogen.")
    if "FAGLFLEXP" in cu and "VERSN" not in cu:
        tips.append("Zugriff auf FAGLFLEXP ohne Planversions-Filter (VERSN) — alle Planversionen werden gelesen.")
    if "ACDOCA" in cu and "RBUKRS" not in cu:
        issues.append("ACDOCA-Abfrage ohne Buchungskreis-Filter (RBUKRS) — mandantenübergreifende Daten können gelesen werden.")
    if "SELECT *" in cu:
        issues.append("SELECT * — explizite Feldliste angeben (Performance + Stabilität).")
    if not issues and not tips:
        tips.append("Keine offensichtlichen Probleme gefunden.")
    return issues, tips

# ─── Server ────────────────────────────────────────────────────────────────────

app = Server("sap-planning-guidelines-v1")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="add_document",
             description="Fügt eine PDF- oder Textdatei zur Wissensdatenbank hinzu. Absoluter Pfad oder Dateiname relativ zu documents/.",
             inputSchema={"type": "object", "properties": {
                 "path":        {"type": "string", "description": "Absoluter oder relativer Dateipfad"},
                 "description": {"type": "string", "description": "Kurzbeschreibung des Dokuments"},
             }, "required": ["path"]}),

        Tool(name="list_documents",
             description="Listet alle indizierten Dokumente und ihren Status auf.",
             inputSchema={"type": "object", "properties": {}}),

        Tool(name="remove_document",
             description="Entfernt ein Dokument anhand des Dateinamens aus der Wissensdatenbank.",
             inputSchema={"type": "object", "properties": {
                 "name": {"type": "string", "description": "Dateiname des zu entfernenden Dokuments"},
             }, "required": ["name"]}),

        Tool(name="reindex_all",
             description="Erkennt geänderte Dokumente anhand der MD5-Checksumme und indiziert diese neu.",
             inputSchema={"type": "object", "properties": {}}),

        Tool(name="search_documents",
             description="Hybride semantisch+keyword Suche über alle indizierten SAP-Planungsrichtlinien.",
             inputSchema={"type": "object", "properties": {
                 "query":       {"type": "string"},
                 "max_results": {"type": "integer", "default": 5},
                 "doc_filter":  {"type": "string", "description": "Optional: Suche auf ein Dokument einschränken"},
             }, "required": ["query"]}),

        Tool(name="get_document_section",
             description="Liest einen Zeichenbereich aus einem indizierten Dokument.",
             inputSchema={"type": "object", "properties": {
                 "name":       {"type": "string"},
                 "start_char": {"type": "integer", "default": 0},
                 "length":     {"type": "integer", "default": 3000},
             }, "required": ["name"]}),

        Tool(name="sac_planning_snippet",
             description=(
                 "Generiert ein SAP Analytics Cloud Planning Template, angereichert mit Richtlinienkontext (RAG). "
                 "Typen: data_action, allocation, currency_conversion, version_management, planning_sequence."
             ),
             inputSchema={"type": "object", "properties": {
                 "snippet_type": {"type": "string",
                                  "enum": ["data_action", "allocation", "currency_conversion",
                                           "version_management", "planning_sequence"]},
                 "context": {"type": "string", "description": "Beschreibe was der Code tun soll"},
             }, "required": ["snippet_type"]}),

        Tool(name="ibp_snippet",
             description=(
                 "Generiert ein SAP IBP (Integrated Business Planning) Template, angereichert mit Richtlinienkontext (RAG). "
                 "Typen: key_figure_calculation, supply_heuristic, alert_definition, macros_excel."
             ),
             inputSchema={"type": "object", "properties": {
                 "snippet_type": {"type": "string",
                                  "enum": ["key_figure_calculation", "supply_heuristic",
                                           "alert_definition", "macros_excel"]},
                 "context": {"type": "string", "description": "Beschreibe den Anwendungsfall"},
             }, "required": ["snippet_type"]}),

        Tool(name="bpc_snippet",
             description=(
                 "Generiert ein SAP BPC (Business Planning and Consolidation) Template, angereichert mit Richtlinienkontext (RAG). "
                 "Typen: script_logic, fox_formula, validation_rule, etl_transformation."
             ),
             inputSchema={"type": "object", "properties": {
                 "snippet_type": {"type": "string",
                                  "enum": ["script_logic", "fox_formula",
                                           "validation_rule", "etl_transformation"]},
                 "context": {"type": "string", "description": "Beschreibe den Anwendungsfall"},
             }, "required": ["snippet_type"]}),

        Tool(name="s4_planning_snippet",
             description=(
                 "Generiert ein SAP S/4HANA Planungs-Template (PP/FI/CO), angereichert mit Richtlinienkontext (RAG). "
                 "Typen: mrp_configuration, fi_planning_cds, pp_planned_order, profit_center_planning."
             ),
             inputSchema={"type": "object", "properties": {
                 "snippet_type": {"type": "string",
                                  "enum": ["mrp_configuration", "fi_planning_cds",
                                           "pp_planned_order", "profit_center_planning"]},
                 "context": {"type": "string", "description": "Beschreibe den Anwendungsfall"},
             }, "required": ["snippet_type"]}),

        Tool(name="check_planning_config",
             description=(
                 "Prüft SAP-Planungskonfiguration oder -Code gegen Best Practices, "
                 "angereichert mit indizierten Richtlinien. "
                 "Bereiche: sac_planning, ibp, bpc, s4_planning."
             ),
             inputSchema={"type": "object", "properties": {
                 "code":   {"type": "string", "description": "Zu prüfender Code oder Konfigurationstext"},
                 "area":   {"type": "string",
                            "enum": ["sac_planning", "ibp", "bpc", "s4_planning"],
                            "description": "SAP-Planungsbereich"},
             }, "required": ["code", "area"]}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):

    # ── add_document ───────────────────────────────────────────────────────────
    if name == "add_document":
        raw  = arguments["path"]
        desc = arguments.get("description", "")
        fp   = Path(raw) if Path(raw).is_absolute() else DOCS_DIR / raw
        if not fp.exists():
            return [TextContent(type="text", text=f"Datei nicht gefunden: {fp}")]

        dest = DOCS_DIR / fp.name
        if fp.resolve() != dest.resolve():
            shutil.copy2(fp, dest)

        index = load_index()
        if dest.name in index and index[dest.name].get("checksum") == file_checksum(dest):
            return [TextContent(type="text", text=
                f"Bereits aktuell: {dest.name} ({index[dest.name]['chunks']} Chunks). "
                f"Kein Re-Indexing nötig.")]

        entry = index_document(dest, desc, index)
        return [TextContent(type="text", text=
            f"Indiziert: {dest.name}\n"
            f"  Zeichen    : {entry['chars']:,}\n"
            f"  Chunks     : {entry['chunks']:,} (Größe={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})\n"
            f"  Beschreibung: {desc or '(keine)'}\n"
            f"  PDF-Backend: {PDF_BACKEND or 'keins (nur Textdateien)'}")]

    # ── list_documents ─────────────────────────────────────────────────────────
    elif name == "list_documents":
        index = load_index()
        stale = check_stale_documents(index)
        if not index:
            return [TextContent(type="text", text="Keine Dokumente indiziert. Bitte add_document verwenden.")]
        lines = [f"Indizierte Dokumente ({len(index)}):\n"]
        for n, m in index.items():
            flag = " [VERALTET — bitte reindex_all ausführen]" if n in stale else ""
            lines.append(
                f"  {n}{flag}\n"
                f"    {m.get('description', '(keine Beschreibung)')}\n"
                f"    {m.get('chars', 0):,} Zeichen | {m.get('chunks', 0)} Chunks"
            )
        if stale:
            lines.append(f"\nreindex_all ausführen um {len(stale)} geänderte(s) Dokument(e) zu aktualisieren.")
        return [TextContent(type="text", text="\n".join(lines))]

    # ── remove_document ────────────────────────────────────────────────────────
    elif name == "remove_document":
        n = arguments["name"]
        index = load_index()
        if n not in index:
            return [TextContent(type="text", text=f"Nicht gefunden: {n}")]
        col = _get_collection()
        existing = col.get(where={"doc_name": n})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
        cache = Path(index[n].get("cache", ""))
        if cache.exists():
            cache.unlink()
        del index[n]
        save_index(index)
        return [TextContent(type="text", text=f"Entfernt: {n}")]

    # ── reindex_all ────────────────────────────────────────────────────────────
    elif name == "reindex_all":
        index = load_index()
        stale = check_stale_documents(index)
        if not stale:
            return [TextContent(type="text", text="Alle Dokumente sind aktuell. Kein Re-Indexing nötig.")]
        results = []
        for n in stale:
            path = DOCS_DIR / n
            desc = index[n].get("description", "")
            entry = index_document(path, desc, index)
            results.append(f"  Re-indiziert: {n} → {entry['chunks']} Chunks")
        return [TextContent(type="text", text=f"{len(stale)} Dokument(e) neu indiziert:\n" + "\n".join(results))]

    # ── search_documents ───────────────────────────────────────────────────────
    elif name == "search_documents":
        q          = arguments["query"]
        max_r      = int(arguments.get("max_results", 5))
        doc_filter = arguments.get("doc_filter")
        try:
            hits = hybrid_search(q, n_results=max_r, doc_filter=doc_filter)
        except Exception as e:
            return [TextContent(type="text", text=
                f"Suchfehler: {e}\nStelle sicher dass ChromaDB installiert und Dokumente indiziert sind.")]
        if not hits:
            return [TextContent(type="text", text=f"Keine Ergebnisse für: {q}")]
        parts = [f"Hybride Suchergebnisse für '{q}' ({len(hits)} Treffer):\n"]
        for h in hits:
            parts.append(
                f"[{h['doc_name']} | Chunk {h['chunk_idx']} | "
                f"sem={h['sem_score']} kw={h['kw_score']} score={h['score']}]\n"
                f"{h['text']}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(parts))]

    # ── get_document_section ───────────────────────────────────────────────────
    elif name == "get_document_section":
        n      = arguments["name"]
        start  = int(arguments.get("start_char", 0))
        length = int(arguments.get("length", 3000))
        index  = load_index()
        if n not in index:
            return [TextContent(type="text", text=f"Nicht gefunden: {n}")]
        cp = Path(index[n].get("cache", ""))
        text = cp.read_text(encoding="utf-8") if cp.exists() else "(kein Cache)"
        return [TextContent(type="text", text=
            f"{n} [Zeichen {start}–{start+length}]:\n\n{text[start:start+length]}")]

    # ── sac_planning_snippet ───────────────────────────────────────────────────
    elif name == "sac_planning_snippet":
        stype = arguments["snippet_type"]
        ctx   = arguments.get("context", "")
        tmpl  = SAC_TEMPLATES.get(stype, "// Unbekannter Snippet-Typ")
        rag_ctx = build_rag_context(f"SAC Planning {stype} {ctx}".strip(), n=4)
        out = f"SAC Planning Template: {stype}\n\n```\n{tmpl}\n```"
        if ctx:
            out += f"\n\nKontext: {ctx}"
        out += f"\n\n── Relevante Richtlinienabschnitte (RAG) ──\n\n{rag_ctx}" if rag_ctx else \
               "\n\n(Noch keine Richtliniendokumente indiziert. Füge PDFs mit add_document hinzu.)"
        return [TextContent(type="text", text=out)]

    # ── ibp_snippet ────────────────────────────────────────────────────────────
    elif name == "ibp_snippet":
        stype = arguments["snippet_type"]
        ctx   = arguments.get("context", "")
        tmpl  = IBP_TEMPLATES.get(stype, "// Unbekannter Snippet-Typ")
        rag_ctx = build_rag_context(f"SAP IBP {stype} {ctx}".strip(), n=4)
        out = f"SAP IBP Template: {stype}\n\n```\n{tmpl}\n```"
        if ctx:
            out += f"\n\nKontext: {ctx}"
        out += f"\n\n── Relevante Richtlinienabschnitte (RAG) ──\n\n{rag_ctx}" if rag_ctx else \
               "\n\n(Noch keine Richtliniendokumente indiziert. Füge PDFs mit add_document hinzu.)"
        return [TextContent(type="text", text=out)]

    # ── bpc_snippet ────────────────────────────────────────────────────────────
    elif name == "bpc_snippet":
        stype = arguments["snippet_type"]
        ctx   = arguments.get("context", "")
        tmpl  = BPC_TEMPLATES.get(stype, "// Unbekannter Snippet-Typ")
        rag_ctx = build_rag_context(f"SAP BPC {stype} {ctx}".strip(), n=4)
        out = f"SAP BPC Template: {stype}\n\n```\n{tmpl}\n```"
        if ctx:
            out += f"\n\nKontext: {ctx}"
        out += f"\n\n── Relevante Richtlinienabschnitte (RAG) ──\n\n{rag_ctx}" if rag_ctx else \
               "\n\n(Noch keine Richtliniendokumente indiziert. Füge PDFs mit add_document hinzu.)"
        return [TextContent(type="text", text=out)]

    # ── s4_planning_snippet ────────────────────────────────────────────────────
    elif name == "s4_planning_snippet":
        stype = arguments["snippet_type"]
        ctx   = arguments.get("context", "")
        tmpl  = S4_TEMPLATES.get(stype, "// Unbekannter Snippet-Typ")
        rag_ctx = build_rag_context(f"S/4HANA Planning {stype} {ctx}".strip(), n=4)
        out = f"S/4HANA Planning Template: {stype}\n\n```\n{tmpl}\n```"
        if ctx:
            out += f"\n\nKontext: {ctx}"
        out += f"\n\n── Relevante Richtlinienabschnitte (RAG) ──\n\n{rag_ctx}" if rag_ctx else \
               "\n\n(Noch keine Richtliniendokumente indiziert. Füge PDFs mit add_document hinzu.)"
        return [TextContent(type="text", text=out)]

    # ── check_planning_config ──────────────────────────────────────────────────
    elif name == "check_planning_config":
        code = arguments["code"]
        area = arguments["area"]

        checker_map = {
            "sac_planning": (check_sac_planning, "SAC Planning"),
            "ibp":          (check_ibp_config,   "SAP IBP"),
            "bpc":          (check_bpc_logic,    "SAP BPC"),
            "s4_planning":  (check_s4_planning,  "S/4HANA Planning"),
        }
        checker_fn, area_label = checker_map[area]
        issues, tips = checker_fn(code)

        rag_query = f"SAP {area_label} best practices " + " ".join(
            re.findall(r'\b[A-Z_][A-Z_0-9]{2,}\b', code)[:8])
        rag_ctx = build_rag_context(rag_query, n=3)

        parts = [f"Code-Review ({area_label})\n"]
        if issues:
            parts.append("Probleme:\n" + "\n".join(f"  ⚠  {i}" for i in issues))
        if tips:
            parts.append("Hinweise:\n" + "\n".join(f"  💡 {t}" for t in tips))
        if rag_ctx:
            parts.append(f"── Passende Richtlinienabschnitte ──\n\n{rag_ctx}")
        return [TextContent(type="text", text="\n\n".join(parts))]

    return [TextContent(type="text", text=f"Unbekanntes Tool: {name}")]


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
