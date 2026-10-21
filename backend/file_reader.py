import os
import mimetypes
from io import BytesIO, StringIO
import time
from datetime import datetime
import pandas as pd
from docx import Document

import fitz  

def extract_text(json):
    result = {
        "metadata": {
            "file_metadata": {},
            "chunking_metadata": {},
            "embedding_metadata": {},
        },
        "text": "",
        "chunks": [],

    }
    start = time.time()
    
    if(json.get("status","")=="duplicate"):
        return json
    
    uploaded_file = json.get("uploaded_file",None)
    filename = json.get("file_name", "")
    extension = os.path.splitext(filename)[1].lower()
    file_bytes = json.get("file_bytes", None)


    file_metadata = {
        "file_name": filename,
        "extension": extension,
        "url": json.get("url", ""),
        "file_size_bytes": 0,
        "mime_type": mimetypes.guess_type(filename)[0],
    }
    result["metadata"]["file_metadata"] = file_metadata
   

    try:

        if file_bytes is None:
            uploaded_file.file.seek(0)
            file_bytes = uploaded_file.file.read()

# ---------------- PDF ----------------

        if extension == ".pdf":

            pdf_bytes = file_bytes

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            metadata = doc.metadata or {}

            file_metadata["pdf_metadata"] = {
                "author": metadata.get("author"),
                "creator": metadata.get("creator"),
                "producer": metadata.get("producer"),
                "title": metadata.get("title"),
                "subject": metadata.get("subject"),
                "creation_date": metadata.get("creationDate"),
                "modification_date": metadata.get("modDate"),
                "pages": doc.page_count,
            }

       

            text = ""

            for page in doc:

                page_dict = page.get_text("dict")

                for block in page_dict["blocks"]:

                    # Skip image blocks
                    if "lines" not in block:
                        continue

                    block_lines = []

                    for line in block["lines"]:

                        line_text = "".join(
                            span["text"] for span in line["spans"]
                        ).strip()

                        if line_text:
                            block_lines.append(line_text)

                    if block_lines:
                        # Join lines inside the block
                        text += " ".join(block_lines)
                        text += "\n\n"

                text += "\n"

            doc.close()

            result["text"] = text.strip()


        # ---------------- CSV ----------------
       

        elif extension == ".csv":

            df = pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))

            file_metadata["csv_metadata"] = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns)
            }

            result["text"] = df.to_string(index=False)

        # ---------------- Excel ----------------

        elif extension in [".xlsx", ".xls"]:

            sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)

            excel_meta = {}

            text = ""

            for sheet, df in sheets.items():

                excel_meta[sheet] = {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "column_names": list(df.columns)
                }

                text += f"\n----- {sheet} -----\n"
                text += df.to_string(index=False)
                text += "\n"

            file_metadata["excel_metadata"] = excel_meta

            result["text"] = text

        # ---------------- DOCX ----------------

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
                "paragraphs": len(doc.paragraphs)
            }

            result["text"] = "\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            )

        # ---------------- TXT ----------------

        elif extension == ".txt":

            result["text"] = file_bytes.decode(
                "utf-8",
                errors="ignore"
            )

        else:

            result["text"] = "Unsupported file type"

        file_metadata["processing_time"] = round(time.time() - start, 4)
        file_metadata["processed_at"] = datetime.now().isoformat()

        return result.get("text", "No text extracted")

    except Exception as e:

        return {
            "error": str(e)
        }