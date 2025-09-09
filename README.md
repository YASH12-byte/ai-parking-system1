# AI Parking System (Minimal)

## Quickstart

1. Create a Python 3.11+ venv
2. Install requirements
3. Run the server

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Endpoints
- POST `/auth/token?user_id=...` → returns JWT
- POST `/detect` (multipart with `file`) → occupancy results
- POST `/assign` (JSON) → best slot suggestion
- POST `/crypto/encrypt`, `/crypto/decrypt` → symmetric encryption utilities

## Notes
- Detection uses simple edge-density per-slot heuristic as a placeholder.
- `src/fuzzy.py` implements a soft-computing score for slot prioritization.
- `src/crypto_utils.py` provides JWT and AES-GCM helpers.
- Configure secrets in environment variables: `SECRET_KEY`.

## Features
- Real-time camera detection with webcam support
- Multi-camera dashboard
- Smart slot assignment with fuzzy logic
- User authentication with JWT
- Secure password hashing
- CORS enabled for cross-origin requests