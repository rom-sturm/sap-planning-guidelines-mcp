#!/usr/bin/env python3
"""
SAP SAC Planning Guidelines MCP Server — ChromaDB Edition
- Wissensdatenbank für SAP Analytics Cloud Planning
- Local sentence-transformers embeddings (vollständig offline)
- Hybrid search: semantisch (ChromaDB) + keyword (BM25-style)
- Auto-Chunking mit Overlap
- Re-Index-Erkennung via MD5-Checksum
- RAG-angereicherte SAC Planning Snippets
"""

import json
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
        _embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return _embed_fn

def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name="sac_planning_guidelines",
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
            "id":         f"{doc_name}__chunk_{i}",
            "text":       chunk,
            "doc_name":   doc_name,
            "chunk_idx":  i,
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
        sem_score   = 1.0 - float(dist)
        kw_score    = keyword_score(text, query)
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

        // Quell-Daten in Plan-Version schreiben
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
        // Empfohlene Reihenfolge:
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

# ─── Best-practice check ───────────────────────────────────────────────────────

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

# ─── Server ────────────────────────────────────────────────────────────────────

app = Server("sac-planning-guidelines-v1")

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
             description="Hybride semantisch+keyword Suche über alle indizierten SAC-Planungsrichtlinien.",
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

        Tool(name="check_planning_config",
             description="Prüft SAC Planning Data Actions oder Konfiguration gegen Best Practices, angereichert mit indizierten Richtlinien.",
             inputSchema={"type": "object", "properties": {
                 "code": {"type": "string", "description": "Zu prüfender Code oder Konfigurationstext"},
             }, "required": ["code"]}),
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
            f"  Zeichen     : {entry['chars']:,}\n"
            f"  Chunks      : {entry['chunks']:,} (Größe={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})\n"
            f"  Beschreibung: {desc or '(keine)'}\n"
            f"  PDF-Backend : {PDF_BACKEND or 'keins (nur Textdateien)'}")]

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
        parts = [f"Suchergebnisse für '{q}' ({len(hits)} Treffer):\n"]
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
        out += (f"\n\n── Relevante Richtlinienabschnitte (RAG) ──\n\n{rag_ctx}" if rag_ctx else
                "\n\n(Noch keine Richtliniendokumente indiziert. Füge PDFs mit add_document hinzu.)")
        return [TextContent(type="text", text=out)]

    # ── check_planning_config ──────────────────────────────────────────────────
    elif name == "check_planning_config":
        code = arguments["code"]
        issues, tips = check_sac_planning(code)

        rag_query = "SAC Planning best practices " + " ".join(
            re.findall(r'\b[A-Z_][A-Z_0-9]{2,}\b', code)[:8])
        rag_ctx = build_rag_context(rag_query, n=3)

        parts = ["Code-Review (SAC Planning)\n"]
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
