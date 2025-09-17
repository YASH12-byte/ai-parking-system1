from typing import List, Dict, Tuple
import asyncio
import json
import time

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Column, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Session
from passlib.context import CryptContext
import requests
import io

from .config import settings
from .crypto_utils import decrypt_text, encrypt_text, generate_jwt, verify_jwt
from .detector import detect_occupancy
from .fuzzy import compute_slot_score
from .schemas import AssignmentRequest, AssignmentResponse, DetectionResult, Slot

app = FastAPI(title="AI Parking System")
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)
import asyncio
from typing import AsyncGenerator

# In-memory pubsub for SSE (simple; replace with Redis for scale)
class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    async def publish(self, data: str) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(data)
            except Exception:
                pass


event_bus = EventBus()

# CORS (allow browser preflight/OPTIONS for JSON fetches from other origins or ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Real-time occupancy state & simple ETA predictions ---
latest_occupancy_results: List[Dict] = []
_occupancy_event = asyncio.Event()
_slot_history: Dict[str, List[Tuple[float, bool]]] = {}


def _update_history(results: List[Dict]) -> None:
    now = time.time()
    # append latest state per slot
    for r in results:
        sid = str(r.get("slot_id"))
        occ = bool(r.get("occupied"))
        hist = _slot_history.setdefault(sid, [])
        hist.append((now, occ))
        # keep only last 200 points per slot
        if len(hist) > 200:
            del hist[: len(hist) - 200]


def _compute_eta_minutes(sid: str) -> float | None:
    # very simple heuristic: if currently occupied, estimate time-to-free as
    # smoothed average of past occupied run lengths; else None
    hist = _slot_history.get(sid) or []
    if not hist:
        return None
    # determine current state
    current_occ = hist[-1][1]
    if not current_occ:
        return None
    # collect durations of past occupied segments
    segments = []
    start = None
    prev = None
    for ts, occ in hist:
        if occ and start is None:
            start = ts
        if prev is not None and prev and not occ and start is not None:
            segments.append(ts - start)
            start = None
        prev = occ
    # if still in occupied run, add partial duration
    if start is not None:
        segments.append(hist[-1][0] - start)
    if not segments:
        return None
    # exponential smoothing of durations (seconds)
    alpha = 0.4
    s = segments[0]
    for d in segments[1:]:
        s = alpha * d + (1 - alpha) * s
    remaining = max(0.0, s - segments[-1])
    return round(remaining / 60.0, 1) if remaining > 30 else 0.5


def _attach_eta(results: List[Dict]) -> Dict[str, float | None]:
    etas: Dict[str, float | None] = {}
    for r in results:
        sid = str(r.get("slot_id"))
        etas[sid] = _compute_eta_minutes(sid)
    return etas


def _publish_occupancy_full(results: List[Dict]) -> None:
    global latest_occupancy_results
    latest_occupancy_results = results or []
    if not _occupancy_event.is_set():
        _occupancy_event.set()


# --- Security headers (CSP, HSTS, etc.) ---
import secrets


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    # Generate a per-request nonce for inline scripts
    csp_nonce = secrets.token_urlsafe(16)
    request.state.csp_nonce = csp_nonce
    response: Response = await call_next(request)

    # Content Security Policy allowing only self and nonce'd inline scripts
    csp = (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{csp_nonce}'; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; "  # allow inline styles from the app CSS
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'"
    )
    response.headers.setdefault("Content-Security-Policy", csp)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    # HSTS (only meaningful over HTTPS)
    response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
    return response


# --- Auth / DB setup ---
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(128), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(256), unique=False, nullable=True)


engine = create_engine("sqlite:///./app.db", echo=False)
Base.metadata.create_all(engine)
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
    with Session(engine) as session:
        yield session


# Lightweight migration to add missing columns if running on an existing DB
with engine.connect() as conn:
    cols = conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
    names = {c[1] for c in cols}  # (cid, name, type, ...)
    if "email" not in names:
        conn.exec_driver_sql("ALTER TABLE users ADD COLUMN email VARCHAR(256)")


def get_current_payload(request: Request):
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    token: str | None = None
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    # Also allow token via query for simple testing/dev
    token = token or request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        return verify_jwt(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "csp_nonce": getattr(request.state, "csp_nonce", "")})
@app.get("/events")
async def sse_events(request: Request, payload: dict = Depends(get_current_payload)):
    async def event_stream() -> AsyncGenerator[bytes, None]:
        queue = event_bus.subscribe()
        try:
            # initial comment to open stream
            yield b":ok\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {data}\n\n".encode("utf-8")
                except asyncio.TimeoutError:
                    # keep-alive
                    yield b":keepalive\n\n"
        finally:
            event_bus.unsubscribe(queue)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/login")
def login_page():
    return RedirectResponse(url="/#login", status_code=303)


@app.get("/signup")
def signup_page():
    return RedirectResponse(url="/#signup", status_code=303)


# Serve SPA index for unknown GET routes (e.g., /signup when user clicks a stale link)
@app.get("/{full_path:path}", response_class=HTMLResponse)
def spa_catch_all(full_path: str, request: Request):
    # Allow other routes above to take precedence; this runs last
    return templates.TemplateResponse("index.html", {"request": request, "csp_nonce": getattr(request.state, "csp_nonce", "")})


@app.post("/auth/token")
def create_token(user_id: str):
    # Development helper to mint a token for quick testing
    token = generate_jwt({"sub": user_id})
    return {"token": token}


@app.post("/auth/signup")
def signup(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    # Basic validation
    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="All fields are required")
    
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    # Check if user already exists
    existing_user = db.scalar(select(User).where(User.username == username))
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email already exists
    existing_email = db.scalar(select(User).where(User.email == email))
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    try:
        # Create new user
        user = User(
            username=username.strip(),
            email=email.strip().lower(),
            password_hash=pwd_ctx.hash(password)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return {
            "ok": True,
            "message": "Account created successfully!",
            "user_id": user.id
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create account: {str(exc)}")


@app.options("/auth/signup")
def signup_options():
    return Response(status_code=204)


# Trailing-slash alias to avoid 405 from proxies adding '/'
@app.post("/auth/signup/")
def signup_alias(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    return signup(username=username, email=email, password=password, confirm_password=confirm_password, db=db)


@app.options("/auth/signup/")
def signup_options_alias():
    return Response(status_code=204)


@app.get("/auth/signup")
def signup_health():
    return {"ok": True, "endpoint": "/auth/signup", "method": "GET", "message": "Use POST to create an account."}


@app.get("/auth/signup/")
def signup_health_alias():
    return {"ok": True, "endpoint": "/auth/signup/", "method": "GET", "message": "Use POST to create an account."}


@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == username))
    if not user or not pwd_ctx.verify(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = generate_jwt({"sub": username})
    return {"token": token}


@app.options("/auth/login")
def login_options():
    return Response(status_code=204)


@app.post("/auth/login/")
def login_alias(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    return login(username=username, password=password, db=db)


@app.options("/auth/login/")
def login_options_alias():
    return Response(status_code=204)


@app.get("/auth/login")
def login_health():
    return {"ok": True, "endpoint": "/auth/login", "method": "GET", "message": "Use POST to login."}


@app.get("/auth/login/")
def login_health_alias():
    return {"ok": True, "endpoint": "/auth/login/", "method": "GET", "message": "Use POST to login."}


# Flexible signup endpoint to tolerate proxies/clients; accepts POST form, POST JSON, or GET query (demo only)
@app.api_route("/api/signup", methods=["GET", "POST", "OPTIONS"])
async def api_signup(request: Request, db: Session = Depends(get_db)):
    try:
        method = request.method.upper()
        username = None
        email = None
        password = None
        confirm_password = None

        if method == "GET":
            qp = request.query_params
            username = (qp.get("username") or "").strip()
            email = (qp.get("email") or "").strip().lower()
            password = qp.get("password") or ""
            confirm_password = qp.get("confirm_password") or password
        elif method == "POST":
            ctype = request.headers.get("content-type", "").lower()
            if "application/json" in ctype:
                body = await request.json()
                username = (body.get("username") or "").strip()
                email = (body.get("email") or "").strip().lower()
                password = body.get("password") or ""
                confirm_password = body.get("confirm_password") or password
            else:
                form = await request.form()
                username = (str(form.get("username")) if form.get("username") else "").strip()
                email = (str(form.get("email")) if form.get("email") else "").strip().lower()
                password = str(form.get("password") or "")
                confirm_password = str(form.get("confirm_password") or password)
        else:
            return Response(status_code=204)

        # Basic validation
        if not username or not email or not password:
            raise HTTPException(status_code=400, detail="All fields are required")
        if password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        existing_user = db.scalar(select(User).where(User.username == username))
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        existing_email = db.scalar(select(User).where(User.email == email))
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already registered")

        user = User(username=username, email=email, password_hash=pwd_ctx.hash(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"ok": True, "message": "Account created successfully!", "user_id": user.id}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create account: {str(exc)}")


@app.api_route("/api/signup/", methods=["GET", "POST", "OPTIONS"])
async def api_signup_alias(request: Request, db: Session = Depends(get_db)):
    return await api_signup(request, db)


@app.post("/auth/reset")
def reset_password(
    username: str = Form(...),
    email: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.scalar(select(User).where(User.username == username))
    if not user or (email and user.email and user.email.lower() != email.lower()):
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = pwd_ctx.hash(new_password)
    db.add(user)
    db.commit()
    return {"ok": True}


@app.options("/auth/reset")
def reset_options():
    return Response(status_code=204)


@app.post("/auth/reset/")
def reset_password_alias(
    username: str = Form(...),
    email: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db),
):
    return reset_password(username=username, email=email, new_password=new_password, db=db)


@app.options("/auth/reset/")
def reset_options_alias():
    return Response(status_code=204)


@app.get("/auth/reset")
def reset_health():
    return {"ok": True, "endpoint": "/auth/reset", "method": "GET", "message": "Use POST to reset password."}


@app.get("/auth/reset/")
def reset_health_alias():
    return {"ok": True, "endpoint": "/auth/reset/", "method": "GET", "message": "Use POST to reset password."}


@app.post("/crypto/encrypt")
def api_encrypt(text: str, secret: str, payload: dict = Depends(get_current_payload)):
    return {"ciphertext": encrypt_text(text, secret)}


@app.post("/crypto/decrypt")
def api_decrypt(ciphertext: str, secret: str, payload: dict = Depends(get_current_payload)):
    try:
        return {"plaintext": decrypt_text(ciphertext, secret)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Decryption failed") from exc


@app.post("/detect", response_model=List[DetectionResult])
async def detect(file: UploadFile = File(...), slotsJson: str | None = Form(None), payload: dict = Depends(get_current_payload)):
    data = await file.read()
    file_bytes = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    slots: List[Slot] = []
    if slotsJson:
        import json

        try:
            raw = json.loads(slotsJson)
            for item in raw:
                slots.append(Slot(**item))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Invalid slotsJson") from exc

    results = detect_occupancy(image, slots)
    # publish summary
    try:
        free = [r.slot_id for r in results if not r.occupied]
        await event_bus.publish(__import__("json").dumps({"type": "occupancy", "free": free}))
    except Exception:
        pass
    return results


@app.post("/assign", response_model=AssignmentResponse)
def assign(req: AssignmentRequest, payload: dict = Depends(get_current_payload)):
    if not req.available_slots:
        return AssignmentResponse(slot_id=None, score=None)

    # Score each candidate slot using soft-computing score
    scores = [compute_slot_score(req.vehicle_size, req.distance_to_gate, req.user_priority) for _ in req.available_slots]
    best_idx = int(np.argmax(scores))
    return AssignmentResponse(slot_id=req.available_slots[best_idx], score=float(scores[best_idx]))


@app.options("/assign")
def assign_options():
    # Allow CORS preflight to succeed explicitly if a proxy blocks default handling
    return Response(status_code=204)


@app.post("/assign/", response_model=AssignmentResponse)
def assign_alias(req: AssignmentRequest, payload: dict = Depends(get_current_payload)):
    return assign(req, payload)


@app.options("/assign/")
def assign_options_alias():
    return Response(status_code=204)


@app.get("/assign")
def assign_health_get():
    return {"ok": True, "endpoint": "/assign", "method": "GET", "message": "Use POST with JSON body to assign."}


@app.get("/assign/")
def assign_health_get_alias():
    return {"ok": True, "endpoint": "/assign/", "method": "GET", "message": "Use POST with JSON body to assign."}


# --- RTSP/HTTP camera snapshot proxy (fetch remote frame and run detection) ---
@app.post("/detect/url", response_model=List[DetectionResult])
def detect_from_url(url: str, slotsJson: str | None = None, payload: dict = Depends(get_current_payload)):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        file_bytes = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image from URL")
        slots: List[Slot] = []
        if slotsJson:
            import json

            raw = json.loads(slotsJson)
            for item in raw:
                slots.append(Slot(**item))
        results = detect_occupancy(image, slots)
        try:
            free = [r.slot_id for r in results if not r.occupied]
            asyncio.create_task(event_bus.publish(__import__("json").dumps({"type": "occupancy", "free": free})))
        except Exception:
            pass
        return results
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Fetch failed") from exc


@app.post("/detect/rtsp", response_model=List[DetectionResult])
def detect_from_rtsp(rtsp: str, slotsJson: str | None = None, payload: dict = Depends(get_current_payload)):
    try:
        cap = cv2.VideoCapture(rtsp)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="RTSP open failed")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise HTTPException(status_code=400, detail="RTSP read failed")
        slots: List[Slot] = []
        if slotsJson:
            import json

            raw = json.loads(slotsJson)
            for item in raw:
                slots.append(Slot(**item))
        results = detect_occupancy(frame, slots)
        try:
            free = [r.slot_id for r in results if not r.occupied]
            asyncio.create_task(event_bus.publish(__import__("json").dumps({"type": "occupancy", "free": free})))
        except Exception:
            pass
        return results
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="RTSP detect failed") from exc


# --- QR decode (server fallback for wide browser support) ---
@app.post("/qr/decode")
async def qr_decode(file: UploadFile = File(...), payload: dict = Depends(get_current_payload)):
    try:
        data = await file.read()
        file_bytes = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        detector = cv2.QRCodeDetector()
        # detectAndDecodeMulti returns (decoded_texts, points, straight_qrcode)
        # OpenCV API differs by version; try multi first, then single
        texts = []
        try:
            retval, decoded, points, _ = detector.detectAndDecodeMulti(image)
            if retval and decoded:
                texts.extend([t for t in decoded if t])
        except Exception:  # noqa: BLE001
            pass
        if not texts:
            text = detector.detectAndDecode(image)[0]
            if text:
                texts.append(text)
        return {"ok": True, "results": texts}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"QR decode failed: {exc}") from exc


# --- QR generator for public access ---
@app.get("/qr/generate")
def qr_generate(request: Request, url: str | None = None, guest: str | None = None, format: str | None = None):
    try:
        import qrcode
        from qrcode.image.svg import SvgImage
        # Build absolute site URL
        base = url or settings.PUBLIC_BASE_URL or str(request.base_url)
        if guest in {"1", "true", "yes"}:
            base = base.rstrip("/") + "/?guest=1"
        # Larger box_size for crisper PNG; SVG scales cleanly
        qr = qrcode.QRCode(border=2, box_size=10)
        qr.add_data(base)
        qr.make(fit=True)
        if (format or "").lower() == "svg":
            img = qr.make_image(image_factory=SvgImage)
            svg_bytes = img.to_string()
            return Response(content=svg_bytes, media_type="image/svg+xml")
        else:
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"QR generation failed: {exc}") from exc
