import os
import tempfile
import json
from fastapi import HTTPException
from fastapi.responses import FileResponse
import pprint
from .supabase_db import supabase_var

BUCKET = "documents"


def download_file(file_name: str):
    try:
        # Download bytes from Supabase
        response = (
            supabase_var.storage
            .from_(BUCKET)
            .download(file_name)
        )
        print(type(response))
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file_name)

        # Write bytes to the temporary file
        with open(temp_path, "wb") as f:
            f.write(response)

        
        file_response=FileResponse(
            path=temp_path,
            filename=file_name,
            media_type="application/octet-stream"
        )
        # Return the file
        return file_response
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to download file: {str(e)}"
        )