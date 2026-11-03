from supabase import create_client
from config import SUPABASE_KEY, SUPABASE_URL

if not SUPABASE_URL or not SUPABASE_KEY:
	raise RuntimeError(
		"Missing SUPABASE_URL or SUPABASE_KEY. Check backend_analyser/.env and how the app is launched."
	)

supabase_var = create_client(SUPABASE_URL, SUPABASE_KEY)
