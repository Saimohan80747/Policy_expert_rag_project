import mimetypes
import re
import time
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

try:  # newer PyMuPDF exposes `pymupdf`; older exposes `fitz`
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from pipeline import fail


def _decode(data: bytes) -> str:
    """Decode text files: UTF-16/UTF-8 (with BOM), else UTF-8, else cp1252."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _df_to_records(df: pd.DataFrame) -> list[str]:
    """One line per row: 'Header: value | Header: value'. Self-contained for retrieval."""
    df = df.fillna("")
    columns = [str(c).strip() for c in df.columns]
    lines = []
    for values in df.itertuples(index=False, name=None):
        parts = [
            f"{col}: {str(val).strip()}"
            for col, val in zip(columns, values)
            if str(val).strip() != ""
        ]
        if parts:
            lines.append(" | ".join(parts))
    return lines


def _docx_text(doc) -> str:
    """Paragraphs and tables in document order; headings become markdown '#' lines."""
    blocks = []
    for block in doc.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name if block.style is not None else "") or ""
            match = re.match(r"Heading (\d)", style)
            if match:
                text = "#" * int(match.group(1)) + " " + text
            elif style == "Title":
                text = "# " + text
            blocks.append(text)
        elif isinstance(block, Table):
            rows = []
            for row in block.rows:
                seen, cells = set(), []
                for cell in row.cells:
                    if id(cell._tc) in seen:  # merged cells repeat
                        continue
                    seen.add(id(cell._tc))
                    cells.append(cell.text.strip().replace("\n", " "))
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _pdf_text(file_bytes: bytes, file_metadata: dict) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ValueError("PDF is password protected.")
        meta = doc.metadata or {}
        file_metadata["pdf_metadata"] = {
            "author": meta.get("author"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "title": meta.get("title"),
            "subject": meta.get("subject"),
            "creation_date": meta.get("creationDate"),
            "modification_date": meta.get("modDate"),
            "pages": doc.page_count,
        }
        text = ""
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:  # image block
                    continue
                block_lines = []
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"]).strip()
                    if line_text:
                        block_lines.append(line_text)
                if block_lines:
                    text += " ".join(block_lines) + "\n\n"
            text += "\n"
        return text.strip()
    finally:
        doc.close()


def extract_text(state):
    """Step 2: bytes -> text (+ file metadata). Fails cleanly on unreadable files."""
    start = time.time()
    extension = state["extension"]
    file_bytes = state["file_bytes"]
    file_name = state["file_name"]

    file_metadata = {
        "file_name": file_name,
        "extension": extension,
        "url": state.get("url", ""),
        "file_size_bytes": len(file_bytes),
        "mime_type": mimetypes.guess_type(file_name)[0],
    }

    if extension == ".pdf":
        text = _pdf_text(file_bytes, file_metadata)

    elif extension == ".csv":
        csv_text = _decode(file_bytes)
        try:
            df = pd.read_csv(
                StringIO(csv_text), dtype=str, keep_default_na=False, on_bad_lines="skip"
            )
        except (pd.errors.ParserError, ValueError):  # odd delimiter: let pandas sniff it
            df = pd.read_csv(
                StringIO(csv_text),
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
                sep=None,
                engine="python",
            )
        file_metadata["csv_metadata"] = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": [str(c) for c in df.columns],
        }
        text = "\n".join(_df_to_records(df))

    elif extension == ".xlsx":
        sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None, dtype=str)
        excel_meta, parts = {}, []
        for sheet, df in sheets.items():
            excel_meta[sheet] = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": [str(c) for c in df.columns],
            }
            records = _df_to_records(df)
            if records:
                parts.append(f"## Sheet: {sheet}\n" + "\n".join(records))
        file_metadata["excel_metadata"] = excel_meta
        text = "\n\n".join(parts)

    elif extension == ".docx":
        doc = Document(BytesIO(file_bytes))
        props = doc.core_properties
        file_metadata["docx_metadata"] = {
            "author": props.author,
            "title": props.title,
            "subject": props.subject,
            "created": str(props.created),
            "modified": str(props.modified),
            "last_modified_by": props.last_modified_by,
            "paragraphs": len(doc.paragraphs),
        }
        text = _docx_text(doc)

    elif extension == ".txt":
        text = _decode(file_bytes)

    else:
        return fail(state, f"Unsupported file type '{extension}'.", "extract_text")

    if not text.strip():
        return fail(
            state,
            "No extractable text (image-only or scanned files need OCR, which is not supported yet).",
            "extract_text",
        )

    file_metadata["processing_time"] = round(time.time() - start, 4)
    file_metadata["processed_at"] = datetime.now().isoformat()

    state["text"] = text
    state["metadata"] = {
        "file_metadata": file_metadata,
        "chunking_metadata": {},
        "embedding_metadata": {},
    }
    return state
