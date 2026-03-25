import sys, io, os
# Force UTF-8 stdout/stderr on Windows to prevent cp1252 UnicodeEncodeError
# from log lines printed by AI model libraries.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# Ensure backend directory is in sys.path for moved modules (enchance, facexlib, gfpgan)
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import uuid
import asyncio
import time

import enchance

# ─── LIFESPAN (replaces deprecated @app.on_event("startup")) ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start AI engine initialization in the background at server startup."""
    print(f"\n[{time.strftime('%H:%M:%S')}] --- SERVER STARTING ---")
    from enchance import AIImageEnhancer
    engine = AIImageEnhancer()
    import threading
    thread = threading.Thread(target=engine.initialize, daemon=True)
    thread.start()
    print(f"[{time.strftime('%H:%M:%S')}] Background initialization started. "
          "Server is ready for status checks.")
    yield
    # Shutdown: nothing to explicitly clean up for this app.

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = os.path.join("static", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FIX 5: Raised from 90s → 150s to match the internal 85s guard in enchance.py.
# asyncio.wait_for cannot interrupt a running thread on Windows, so the real
# hard stop is the 85s guard inside AIImageEnhancer.enhance(). This 150s limit
# is a safety net that fires only if something unexpected blocks even longer.
ENHANCE_TIMEOUT_S = 150.0


def _ts():
    return time.strftime("%H:%M:%S")


@app.get("/status")
async def get_status():
    """Check AI engine initialization status."""
    from enchance import AIImageEnhancer
    engine = AIImageEnhancer()
    return {"status": engine.get_status()}


@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    filter: str = Form("auto")
):
    print(f"[{_ts()}] Step 0: Received enhancement request for: {file.filename}")

    # ── 1. File Validation ────────────────────────────────────────────────────
    allowed_extensions = {"jpg", "jpeg", "png", "webp"}
    filename = file.filename or "uploaded_image.png"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format (.{ext}). Please upload JPG, PNG, or WebP."
        )

    MAX_UPLOAD = 30 * 1024 * 1024  # 30 MB
    image_data = await file.read()
    file_size = len(image_data)

    if file_size > MAX_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size / (1024*1024):.1f} MB). Max is 30 MB."
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    await file.seek(0)

    # ── 2. Path Setup ─────────────────────────────────────────────────────────
    unique_id = uuid.uuid4()
    upload_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{filename}")
    output_filename = f"enhanced_{unique_id}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # ── 3. Save + Process ────────────────────────────────────────────────────
    try:
        print(f"[{_ts()}] Step 1: Saving upload to disk...")
        with open(upload_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        from enchance import AIImageEnhancer
        engine = AIImageEnhancer()
        status = engine.get_status()

        if status != "ready":
            if "error" in status:
                raise HTTPException(
                    status_code=500,
                    detail=f"AI Engine initialization failed: {status}"
                )
            raise HTTPException(
                status_code=503,
                detail="AI Engine is still loading models. Please retry in a few seconds."
            )

        print(f"[{_ts()}] Step 2: Engine ready. Processing {unique_id} "
              f"(mode='{filter}', timeout={ENHANCE_TIMEOUT_S}s)...")

        try:
            await asyncio.wait_for(
                enchance.enhance_image_ai_mode(upload_path, output_path, mode=filter),
                timeout=ENHANCE_TIMEOUT_S
            )
            print(f"[{_ts()}] Step 3: Enhancement complete for {unique_id}.")
        except asyncio.TimeoutError:
            # This should be very rare now — the internal 85s guard fires first
            # and writes a fallback result. Check if a file was saved anyway.
            print(f"[{_ts()}] WARNING: asyncio timeout ({ENHANCE_TIMEOUT_S}s) for {unique_id}.")
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"[{_ts()}] Internal guard saved a fallback — returning it.")
            else:
                raise HTTPException(
                    status_code=504,
                    detail="Processing timed out. Please try a smaller image or retry."
                )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[{_ts()}] ERROR processing {unique_id}: {e}")
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {e}")

    # ── 4. Return Response ────────────────────────────────────────────────────
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print(f"[{_ts()}] ERROR: Output file missing/empty for {unique_id}.")
        raise HTTPException(status_code=500, detail="Enhanced image was not generated.")

    print(f"[{_ts()}] Step 4: Returning success response for {unique_id}.")
    return {
        "status": "success",
        "image_url": f"/static/outputs/{output_filename}"
    }


@app.get("/")
def read_root():
    return FileResponse(os.path.join("templates", "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
