## Song Genie (Music-Akenator)

An interactive “20 questions” style song guessing oracle.

- **Backend**: Flask + Bayesian belief update + entropy-based question selection with a **Thompson Sampling** learning layer.
- **Frontend**: Next.js App Router UI with multiple pages (Game, History, Insights).

### Run locally

#### Backend

From `song-genie/`:

```bash
python -m pip install -r requirements.txt
python app.py
```

Backend runs on `http://127.0.0.1:5000`.

#### Frontend

From `song-genie-ui/`:

```bash
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL` if your backend isn’t on `http://127.0.0.1:5000`.

### Run with Docker

From repo root:

```bash
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5000`

### API

- `GET /start`: start a new session and return the first question
- `POST /answer`: answer a question and receive next question or result
- `POST /feedback`: mark whether the guess was correct (improves learning stats)
- `GET /sessions`: list past sessions (for History page)
- `GET /insights`: question performance stats (for Insights page)
- `GET /health`: health check

### Tests (backend)

From `song-genie/`:

```bash
python -m pytest -q
```

