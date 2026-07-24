import os
from pathlib import Path

from supabase import create_client
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
	raise RuntimeError(
		"Missing SUPABASE_URL or SUPABASE_KEY. Check backend_analyser/.env and how the app is launched."
	)

supabase_var = create_client(url, key)

