# Movie Blasters frontend

The React client calls `POST /api/v1/chat`. During development, Vite proxies
`/api` to FastAPI at `http://localhost:8000`.

Start the backend from the repository root with the Python environment that
contains the fine-tuned model dependencies:

```powershell
.\training\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

Then start the frontend in a second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173`. Recommendation metadata identifies whether the
backend used the fine-tuned Gemma explanation or deterministic fallback.

For a separately hosted API, define `VITE_API_URL` with its origin before
building the frontend. Do not put TMDB or Hugging Face secrets in a Vite
environment variable.

```powershell
$env:VITE_API_URL = "https://api.example.com"
npm.cmd run build
```
