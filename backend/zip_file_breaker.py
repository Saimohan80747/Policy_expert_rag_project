import os
import tempfile
from zipfile import ZipFile
from fastapi import UploadFile

def zip_file_breaker(zip_upload: UploadFile, pipeline):
    uploaded_files = []
    duplicate_files = []
    failed_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        with ZipFile(zip_upload.file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "rb") as f:
                        upload_file = UploadFile(
                            filename=file,
                            file=f
                        )
                            
                        upload_file.file.seek(0)
                        result = pipeline.run(upload_file)

                        if(result.get("status", "") == "duplicate"):
                            duplicate_files.append({
                                "filename": file,
                                "result": result
                            })

                        else:
                            uploaded_files.append({
                                "filename": file,
                                "result": result
                            })

                except Exception as e:
                    failed_files.append({
                        "filename": file,
                        "error": str(e)
                    })

    return {
        "uploaded_files": uploaded_files,
        "duplicate_files": duplicate_files,
        "failed_files": failed_files
    }