from typing import List

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Column, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Session
from passlib.context import CryptContext
import requests

from .config import settings
from .crypto_utils import decrypt_text, encrypt_text, generate_jwt, verify_jwt
from .detector import detect_occupancy
from .fuzzy import compute_slot_score
from .schemas import AssignmentRequest, AssignmentResponse, DetectionResult, Slot

app = FastAPI(title="AI Parking System")
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

# CORS (allow browser preflight/OPTIONS for JSON fetches from other origins or ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def get_current_payload(token: str | None = None):
    if token is None:
        raise HTTPException(status_code=401, detail="Missing token")
    try:
        return verify_jwt(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid token") from exc


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
    return templates.TemplateResponse("index.html", {"request": request})


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
def api_encrypt(text: str, secret: str):
    return {"ciphertext": encrypt_text(text, secret)}


@app.post("/crypto/decrypt")
def api_decrypt(ciphertext: str, secret: str):
    try:
        return {"plaintext": decrypt_text(ciphertext, secret)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Decryption failed") from exc


@app.post("/detect", response_model=List[DetectionResult])
async def detect(file: UploadFile = File(...), slotsJson: str | None = Form(None)):
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

    return detect_occupancy(image, slots)


@app.post("/assign", response_model=AssignmentResponse)
def assign(req: AssignmentRequest):
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
def assign_alias(req: AssignmentRequest):
    return assign(req)


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
def detect_from_url(url: str, slotsJson: str | None = None):
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
        return detect_occupancy(image, slots)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Fetch failed") from exc


@app.post("/detect/rtsp", response_model=List[DetectionResult])
def detect_from_rtsp(rtsp: str, slotsJson: str | None = None):
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
        return detect_occupancy(frame, slots)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="RTSP detect failed") from exc
