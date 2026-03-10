"""
Advanced Face Recognition Attendance System  v3.0
==================================================
Features:
- MJPEG live video streaming
- Face recognition with unknown/already-marked detection
- Location lock (GPS)
- Admin login (SQLite, multiple admins, session-based)
- Student registration (name, class, phone, photo upload)
- Date-wise attendance history
- Export to CSV and PDF
- QR code generation for public registration link
"""

import io
import logging
import math
import os
import platform
import secrets
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Generator

import cv2
import numpy as np
import face_recognition
import pandas as pd
import qrcode
import sqlite3
from flask import (Flask, render_template, jsonify, abort,
                   Response, request, redirect, url_for,
                   session, send_file, flash)
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ─────────────────────────── Configuration ─────────────────────────────── #

@dataclass
class Config:
    DATASET_DIR: Path            = Path("dataset")
    ATTENDANCE_FILE: Path        = Path("attendance.xlsx")
    DB_FILE: Path                = Path("attendai.db")
    UPLOAD_FOLDER: Path          = Path("dataset")
    ALLOWED_EXTENSIONS: set      = field(default_factory=lambda: {".jpg", ".jpeg", ".png", ".bmp"})
    SECRET_KEY: str              = secrets.token_hex(32)

    FACE_DISTANCE_THRESHOLD: float = 0.50
    FRAME_SCALE: float           = 0.25
    CAMERA_INDEX: int            = 0
    BEEP_FREQUENCY: int          = 1000
    BEEP_DURATION_MS: int        = 500
    LOG_LEVEL: str               = "INFO"
    DATE_FORMAT: str             = "%d-%m-%Y"
    TIME_FORMAT: str             = "%H:%M:%S"
    JPEG_QUALITY: int            = 80

    LOCATION_LOCK: bool          = True
    ALLOWED_LAT: float           = 29.4706
    ALLOWED_LNG: float           = 77.7082
    ALLOWED_RADIUS_METERS: float = 100.0


config = Config()

# ──────────────────────────── Logging ───────────────────────────────────── #

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AttendAI")

# ──────────────────────────── Flask App ─────────────────────────────────── #

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB max upload

config.DATASET_DIR.mkdir(exist_ok=True)

# ──────────────────────────── Database ──────────────────────────────────── #

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS admins (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                role     TEXT    NOT NULL DEFAULT 'admin',
                created  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS students (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                class_dept TEXT NOT NULL,
                phone      TEXT NOT NULL,
                photo_file TEXT NOT NULL,
                registered TEXT NOT NULL
            );
        """)
        # Create default super-admin if no admins exist
        if not conn.execute("SELECT 1 FROM admins").fetchone():
            conn.execute(
                "INSERT INTO admins (username, password, role, created) VALUES (?,?,?,?)",
                ("admin",
                 generate_password_hash("admin123"),
                 "superadmin",
                 datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            )
            logger.info("Default admin created → username: admin  password: admin123")
        conn.commit()


init_db()

# ──────────────────────── Auth Decorator ────────────────────────────────── #

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "admin_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ──────────────────────── Face Database ─────────────────────────────────── #

class FaceDatabase:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.names: list[str] = []
        self.encodings: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        with self._lock:
            self.names.clear()
            self.encodings.clear()
            if not self.dataset_dir.exists():
                return
            supported = {".jpg", ".jpeg", ".png", ".bmp"}
            loaded = 0
            for file in sorted(self.dataset_dir.iterdir()):
                if file.suffix.lower() not in supported:
                    continue
                try:
                    image = face_recognition.load_image_file(str(file))
                    encs = face_recognition.face_encodings(image)
                    if not encs:
                        continue
                    self.encodings.append(encs[0])
                    self.names.append(file.stem)
                    loaded += 1
                except Exception as exc:
                    logger.error("Failed to load '%s': %s", file.name, exc)
            logger.info("Face database: %d identities loaded.", loaded)

    def reload(self) -> None:
        """Call after adding a new student photo."""
        self._load()

    def match(self, encoding: np.ndarray) -> Optional[str]:
        with self._lock:
            if not self.encodings:
                return None
            distances = face_recognition.face_distance(self.encodings, encoding)
            idx = int(np.argmin(distances))
            if (face_recognition.compare_faces(self.encodings, encoding)[idx]
                    and distances[idx] < config.FACE_DISTANCE_THRESHOLD):
                return self.names[idx].upper()
            return None


face_db = FaceDatabase(config.DATASET_DIR)

# ──────────────────────── Attendance Manager ────────────────────────────── #

class AttendanceManager:
    _lock = threading.Lock()

    @classmethod
    def mark(cls, name: str) -> bool:
        with cls._lock:
            df    = cls._load_sheet()
            today = datetime.now().strftime(config.DATE_FORMAT)
            if ((df["Name"] == name) & (df["Date"] == today)).any():
                return False
            new_row = {
                "Name":   name,
                "Date":   today,
                "Time":   datetime.now().strftime(config.TIME_FORMAT),
                "Status": "Present",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_excel(config.ATTENDANCE_FILE, index=False)
            return True

    @staticmethod
    def _load_sheet() -> pd.DataFrame:
        try:
            return pd.read_excel(config.ATTENDANCE_FILE)
        except FileNotFoundError:
            return pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        except Exception:
            return pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

    @classmethod
    def records_for_date(cls, date_str: str) -> list[dict]:
        df = cls._load_sheet()
        return df[df["Date"] == date_str].to_dict(orient="records")

    @classmethod
    def today_records(cls) -> tuple[list[dict], str]:
        today = datetime.now().strftime(config.DATE_FORMAT)
        return cls.records_for_date(today), today

    @classmethod
    def all_dates(cls) -> list[str]:
        df = cls._load_sheet()
        if df.empty:
            return []
        return sorted(df["Date"].unique().tolist(), reverse=True)

    @classmethod
    def export_csv(cls, date_str: Optional[str] = None) -> io.BytesIO:
        df = cls._load_sheet()
        if date_str:
            df = df[df["Date"] == date_str]
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    @classmethod
    def export_pdf(cls, date_str: Optional[str] = None) -> io.BytesIO:
        df = cls._load_sheet()
        if date_str:
            df = df[df["Date"] == date_str]

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                     textColor=colors.HexColor("#1e293b"),
                                     fontSize=16, spaceAfter=6)
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                     textColor=colors.HexColor("#64748b"),
                                     fontSize=10, spaceAfter=16)

        label = f"Date: {date_str}" if date_str else "All Records"
        story = [
            Paragraph("AttendAI — Attendance Report", title_style),
            Paragraph(f"{label}  ·  Total: {len(df)}", sub_style),
        ]

        if not df.empty:
            table_data = [["#", "Name", "Date", "Time", "Status"]]
            for i, row in enumerate(df.itertuples(), 1):
                table_data.append([str(i), row.Name, row.Date, row.Time, row.Status])

            tbl = Table(table_data, colWidths=[1*cm, 6*cm, 3.5*cm, 3*cm, 3*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e40af")),
                ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,0), 10),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                 [colors.HexColor("#f8fafc"), colors.white]),
                ("FONTSIZE",    (0,1), (-1,-1), 9),
                ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#e2e8f0")),
                ("ALIGN",       (0,0), (-1,-1), "CENTER"),
                ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
                ("ROWHEIGHT",   (0,0), (-1,-1), 0.7*cm),
            ]))
            story.append(tbl)
        else:
            story.append(Paragraph("No records found.", styles["Normal"]))

        doc.build(story)
        buf.seek(0)
        return buf

# ─────────────────────────── Beep ───────────────────────────────────────── #

def _beep() -> None:
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(config.BEEP_FREQUENCY, config.BEEP_DURATION_MS)
        except Exception:
            pass

# ─────────────────── Camera Manager ─────────────────────────────────────── #

class CameraManager:
    def __init__(self):
        self._cap:    Optional[cv2.VideoCapture] = None
        self._frame:  Optional[np.ndarray]       = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if self._running:
            return True
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Could not open camera.")
            return False
        self._cap     = cap
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self) -> None:
        while self._running and self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame


camera = CameraManager()

# ──────────────────────── Scan State ────────────────────────────────────── #

class ScanState:
    _lock   = threading.Lock()
    _name:   Optional[str] = None
    _status: str           = "scanning"

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._name   = None
            cls._status = "scanning"

    @classmethod
    def set_result(cls, name: str, status: str = "done"):
        with cls._lock:
            cls._name   = name
            cls._status = status

    @classmethod
    def get(cls) -> tuple[Optional[str], str]:
        with cls._lock:
            return cls._name, cls._status

# ──────────────────────── Scan Worker ───────────────────────────────────── #

def _scan_worker() -> None:
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        small     = cv2.resize(frame, (0,0), fx=config.FRAME_SCALE, fy=config.FRAME_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        for enc in encodings:
            name = face_db.match(enc)
            if name:
                marked = AttendanceManager.mark(name)
                threading.Thread(target=_beep, daemon=True).start()
                # Stop camera FIRST so MJPEG stream ends cleanly,
                # then set result so browser redirect can fire
                camera.stop()
                ScanState.set_result(name, "done" if marked else "already_marked")
            else:
                camera.stop()
                ScanState.set_result("UNKNOWN", "unknown")
            return

# ──────────────────────── MJPEG Generator ───────────────────────────────── #

def _mjpeg_frames() -> Generator[bytes, None, None]:
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    while camera.is_running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        ok, buf = cv2.imencode(".jpg", frame, enc_params)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

# ──────────────────────── Location ──────────────────────────────────────── #

def _haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def in_zone(lat: float, lng: float) -> tuple[bool, float]:
    if not config.LOCATION_LOCK:
        return True, 0.0
    d = _haversine(lat, lng, config.ALLOWED_LAT, config.ALLOWED_LNG)
    return d <= config.ALLOWED_RADIUS_METERS, round(d, 1)

# ══════════════════════════ ROUTES ══════════════════════════════════════════ #

# ── Attendance / Scan ────────────────────────────────────────────────────── #

@app.route("/")
def home():
    ScanState.reset()
    camera.start()
    threading.Thread(target=_scan_worker, daemon=True).start()
    return render_template("scan.html")


@app.route("/video")
def video():
    return Response(_mjpeg_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    name, state = ScanState.get()
    return jsonify({"status": state, "name": name})


@app.route("/verify-location", methods=["POST"])
def verify_location():
    data = request.get_json(silent=True) or {}
    try:
        lat, lng = float(data["lat"]), float(data["lng"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid coordinates"}), 400
    allowed, distance = in_zone(lat, lng)
    return jsonify({"allowed": allowed, "distance": distance,
                    "radius": config.ALLOWED_RADIUS_METERS, "lock": config.LOCATION_LOCK})


@app.route("/thanks/<name>")
def thanks(name: str):
    safe = name.replace(" ", "").replace("-", "")
    if not safe.isalpha():
        abort(400)
    return render_template("thanks.html", name=name)


# ── Admin Auth ───────────────────────────────────────────────────────────── #

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            admin = conn.execute(
                "SELECT * FROM admins WHERE username=?", (username,)
            ).fetchone()
        if admin and check_password_hash(admin["password"], password):
            session["admin_id"]   = admin["id"]
            session["admin_user"] = admin["username"]
            session["admin_role"] = admin["role"]
            logger.info("Admin login: %s", username)
            return redirect(url_for("trainer"))
        flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Admin Management ─────────────────────────────────────────────────────── #

@app.route("/admin/admins")
@login_required
def manage_admins():
    if session.get("admin_role") != "superadmin":
        abort(403)
    with get_db() as conn:
        admins = conn.execute("SELECT id,username,role,created FROM admins").fetchall()
    return render_template("admin_admins.html", admins=admins)


@app.route("/admin/admins/add", methods=["POST"])
@login_required
def add_admin():
    if session.get("admin_role") != "superadmin":
        abort(403)
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role     = request.form.get("role", "admin")
    if not username or not password:
        flash("Username and password required", "error")
        return redirect(url_for("manage_admins"))
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO admins (username,password,role,created) VALUES (?,?,?,?)",
                (username, generate_password_hash(password), role,
                 datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            )
            conn.commit()
        flash(f"Admin '{username}' added successfully", "success")
    except sqlite3.IntegrityError:
        flash("Username already exists", "error")
    return redirect(url_for("manage_admins"))


@app.route("/admin/admins/delete/<int:aid>", methods=["POST"])
@login_required
def delete_admin(aid: int):
    if session.get("admin_role") != "superadmin":
        abort(403)
    if aid == session["admin_id"]:
        flash("Cannot delete yourself", "error")
        return redirect(url_for("manage_admins"))
    with get_db() as conn:
        conn.execute("DELETE FROM admins WHERE id=?", (aid,))
        conn.commit()
    flash("Admin deleted", "success")
    return redirect(url_for("manage_admins"))


# ── Trainer / Dashboard ──────────────────────────────────────────────────── #

@app.route("/trainer")
@login_required
def trainer():
    date_str = request.args.get("date", datetime.now().strftime(config.DATE_FORMAT))
    records  = AttendanceManager.records_for_date(date_str)
    all_dates = AttendanceManager.all_dates()
    return render_template("trainer.html",
                           tables=records,
                           total=len(records),
                           date=date_str,
                           all_dates=all_dates)


# ── Export ───────────────────────────────────────────────────────────────── #

@app.route("/export/csv")
@login_required
def export_csv():
    date_str = request.args.get("date")
    buf      = AttendanceManager.export_csv(date_str)
    fname    = f"attendance_{date_str or 'all'}.csv"
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=fname)


@app.route("/export/pdf")
@login_required
def export_pdf():
    date_str = request.args.get("date")
    buf      = AttendanceManager.export_pdf(date_str)
    fname    = f"attendance_{date_str or 'all'}.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=fname)


# ── Student Registration ─────────────────────────────────────────────────── #

@app.route("/register", methods=["GET", "POST"])
def register():
    """Public registration page — accessible via QR code link."""
    if request.method == "POST":
        name       = request.form.get("name", "").strip()
        class_dept = request.form.get("class_dept", "").strip()
        phone      = request.form.get("phone", "").strip()
        photo      = request.files.get("photo")

        if not all([name, class_dept, phone, photo]):
            flash("All fields are required", "error")
            return render_template("register.html")

        ext = Path(secure_filename(photo.filename)).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            flash("Only JPG/PNG images allowed", "error")
            return render_template("register.html")

        # Save photo as name.jpg in dataset
        safe_name = name.lower().replace(" ", "_")
        filename  = f"{safe_name}{ext}"
        save_path = config.DATASET_DIR / filename
        photo.save(str(save_path))

        # Store in DB
        with get_db() as conn:
            conn.execute(
                "INSERT INTO students (name,class_dept,phone,photo_file,registered) VALUES (?,?,?,?,?)",
                (name, class_dept, phone, filename,
                 datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            )
            conn.commit()

        # Reload face encodings
        face_db.reload()
        logger.info("New student registered: %s", name)
        flash(f"Registration successful! Welcome {name}", "success")
        return redirect(url_for("register_success", name=name))

    return render_template("register.html")


@app.route("/register/success/<name>")
def register_success(name: str):
    return render_template("register_success.html", name=name)


# ── Admin Student Management ─────────────────────────────────────────────── #

@app.route("/admin/students")
@login_required
def manage_students():
    with get_db() as conn:
        students = conn.execute(
            "SELECT * FROM students ORDER BY registered DESC"
        ).fetchall()
    return render_template("admin_students.html", students=students)


@app.route("/admin/students/delete/<int:sid>", methods=["POST"])
@login_required
def delete_student(sid: int):
    with get_db() as conn:
        student = conn.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
        if student:
            # Remove photo file
            photo_path = config.DATASET_DIR / student["photo_file"]
            if photo_path.exists():
                photo_path.unlink()
            conn.execute("DELETE FROM students WHERE id=?", (sid,))
            conn.commit()
    face_db.reload()
    flash("Student deleted", "success")
    return redirect(url_for("manage_students"))


# ── QR Code ──────────────────────────────────────────────────────────────── #

@app.route("/qr")
@login_required
def qr_code():
    """Generate QR for registration or scan URL. ?type=register|scan"""
    base_url = request.host_url.rstrip("/")
    qr_type  = request.args.get("type", "register")
    target   = f"{base_url}/register" if qr_type == "register" else f"{base_url}/"

    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(target)
    qr.make(fit=True)
    color = "#16a34a" if qr_type == "register" else "#1e40af"
    img = qr.make_image(fill_color=color, back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/admin/qr-page")
@login_required
def qr_page():
    base_url = request.host_url.rstrip("/")
    reg_url  = f"{base_url}/register"
    scan_url = f"{base_url}/"
    return render_template("qr_page.html", reg_url=reg_url, scan_url=scan_url)


# ── Error Handlers ───────────────────────────────────────────────────────── #

@app.errorhandler(HTTPException)
def handle_http(exc):
    return jsonify({"error": exc.name, "description": exc.description}), exc.code


@app.errorhandler(Exception)
def handle_exc(exc):
    logger.exception("Unhandled exception: %s", exc)
    return jsonify({"error": "Internal Server Error"}), 500


# ── Entry Point ──────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)