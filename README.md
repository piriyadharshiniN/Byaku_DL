# BYAKU: AI Spatial Audio Assistant

**Individual contribution with project demo is in this ppt:** https://kumaragurudtsteam-my.sharepoint.com/:p:/g/personal/navithanjali_24ad_kct_ac_in/cQqoNh66HUG4RblS2YfqpM_bEgUCDFJA4fSxXlt6-OdBZ5vQrw

> **SAFETY DISCLAIMER**: BYAKU is an **assistive perception aid**, **NOT a mobility aid or guaranteed obstacle avoidance system**. It must never be relied upon for navigation, traffic safety, or avoidance of hazards. Depth estimates are strictly **relative qualitative rankings** (NEAR, MID-RANGE, FAR) within the current camera frame, **never absolute physical metres**.

---

## 1. Overview & Architecture

BYAKU is a voice-first assistive vision and spatial audio system designed for blind and low-vision users. A camera feed (or uploaded image/video) is analyzed to identify surrounding objects, their relative horizontal and vertical positions (LEFT / CENTER / RIGHT, TOP / MIDDLE / BOTTOM), and their relative proximity (NEAR / MID-RANGE / FAR). Spoken natural-language questions (e.g. *"Where is my bottle?"*) are answered rule-based strictly from active scene memory. A Human-in-the-Loop (HITL) review system allows reviewers to confirm, correct, or reject detections with an immutable, append-only SQLite audit trail.

```
┌────────────────────────────────────────────────────────┐
│                   BROWSER CLIENT                       │
│  - assistant.html (Voice-first Accessible UI)          │
│  - dashboard.html (Technical Pipeline & Review UI)     │
│  - Web Speech API + MediaDevices Camera Stream         │
└────────────────────────┬───────────────────────────────┘
                         │ HTTP / WebSocket (ws://)
                         ▼
┌────────────────────────────────────────────────────────┐
│               NODE.JS / EXPRESS RELAY                  │
│  - Port 3000 (server/http_websocket_relay.js)          │
│  - Static Asset Serving & Multipart Pass-Through       │
│  - Reverse Proxy to FastAPI Backend (/api/*, /health)  │
│  - WebSocket Upgrade & Relay (/ws)                     │
└────────────────────────┬───────────────────────────────┘
                         │ HTTP / WebSocket (ws://)
                         ▼
┌────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                      │
│  - Port 8765 (backend/main.py)                         │
│  - Computer Vision: YOLO11n + ByteTrack (Session-Safe) │
│  - Relative Depth: Depth Anything V2 Small             │
│  - Spatial Reasoning & Scene Graph (NetworkX)          │
│  - Scene Memory (RAM, Temporal Retention & Eviction)   │
│  - Voice Q&A Engine (Rule-based, Non-hallucinatory)   │
│  - HITL Review & Audit Engine (SQLite WAL Mode)        │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                   SQLITE PERSISTENCE                   │
│  - storage/byaku.db (sessions, reviews, audit_events)  │
│  - storage/review_cache/ (TTL=180s frame cache)        │
│  - storage/uploads/ (uploaded video/image retention)   │
└────────────────────────────────────────────────────────┘
```

## 2. Features, Components & How They Work

| Feature | Main components | How it works |
|---|---|---|
| Live camera assistant | Browser `MediaDevices` camera API, WebSocket client, FastAPI (`backend/main.py`) | The user starts the assistant before camera access is requested. The browser sends camera frames to the backend over `/ws`; processed results and status are returned to the page. The assistant can be stopped, and camera/connection errors are shown in the UI. |
| Object detection | Ultralytics YOLO11n or ONNX Runtime (`backend/models/detector.py`) | The detector identifies supported object classes and returns confidence scores and pixel bounding boxes. The default model uses the COCO-80 vocabulary. |
| Object tracking | Session-specific ByteTrack-style tracker (`backend/models/tracker.py`) | Detections are associated between frames using bounding-box overlap and confidence tiers. Stable track IDs help describe continuity and movement; each live session and uploaded video has isolated tracker state. |
| Relative depth | Depth Anything V2 Small, through Transformers/PyTorch or ONNX Runtime (`backend/models/depth.py`) | A frame is converted into a normalized relative depth map. The median value in the central part of each object box is classified as `NEAR`, `MID-RANGE`, or `FAR`. Flat, tiny, or unavailable depth results are reported as `UNKNOWN`; these are not metric distance measurements. |
| Spatial reasoning and scene graph | Bounding-box reasoning and NetworkX scene graph (`backend/spatial/reasoning.py`) | Box centers are mapped to left/center/right and top/middle/bottom regions. Pairwise image-space relationships are derived from object boxes and represented as graph nodes and edges for the current frame. |
| Session scene memory | In-memory `SceneMemory` (`backend/spatial/scene_memory.py`) | Tracked observations are kept as current or recently missing objects. The memory records changes and relationships for the active session, evicts objects after the configured retention period, and is cleared when that session ends. It is not the persistent HITL review database. |
| Scene descriptions and voice output | Template-based narration (`backend/voice/narration.py`), browser Web Speech APIs (`public/js/assistant_speech.js`) | The backend creates concise descriptions from detected labels, position, relative depth, and selected relationships, prioritizing nearer objects and limiting routine narration. The browser reads the returned text aloud and displays the response; speech errors are surfaced in the interface. |
| Voice and text questions | Deterministic `SceneQAEngine` (`backend/voice/qa.py`), browser speech recognition and text form | Spoken questions are transcribed by the browser where its speech-recognition API is available. The backend matches supported question patterns against the active session scene memory and returns a grounded answer rather than generating unseen objects. A text input is available as an alternative. |
| Image and video analysis | Upload form (`public/js/workflow_upload.js`), FastAPI upload endpoints, OpenCV, detector and depth model | An uploaded image is analyzed once. An uploaded video is decoded and sampled at configured intervals in a background job, with a video-specific tracker; job status and sampled results are returned separately from live-camera results. |
| Human review and retraining data | Review controls (`public/js/workflow_dashboard.js`), SQLite (`backend/hitl/`), COCO-style export | A reviewer can confirm, correct, or reject a detection and can edit its label and bounding box. SQLite stores review records and append-only audit events. The export includes active confirmed and corrected annotations; corrected annotations use the corrected label and corrected box when supplied. This creates review data for a later training workflow: it does **not** automatically train or update model weights. The export is annotation JSON with image references, not a packaged set of image files. |

### UI designed with low-vision use in mind

The assistant page keeps the main action—a large, central start/ask control—prominent and provides a visible camera preview with detection overlays. Object descriptions and answers are also provided as text and can be spoken aloud, so information is not conveyed by the overlay alone. Camera and microphone access are off until the user starts the assistant.

The interface uses large, responsive headings, clear section labels, DM Sans and Manrope type, and explicit status text alongside visual status indicators. Native buttons, links, labeled form controls, visible keyboard-focus outlines, and live status regions support keyboard and assistive-technology use. Layouts adapt to narrower screens, and the styles honor the system reduced-motion preference. The workflow dashboard separates uploads, live processing, scene memory, review, depth, and Q&A into named sections.

These are accessibility-oriented design choices, not a claim of WCAG certification or suitability for every user's needs. Browser support and permissions affect camera and speech features; the text question form remains available when speech recognition is unavailable. BYAKU is an assistive perception aid, not a navigation or hazard-avoidance system.

---

## 3. Setup & Installation

### Supported Python Versions
- **Verified Environment**: **Python 3.11.2** on Linux x86_64.
- **Compatible Range**: Python 3.10 to 3.12.
- **Node.js**: v18.20.4+ (Node 18+ LTS).

### Installation Commands

#### Option A: CPU Installation (Verified)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics>=8.3.0
```

#### Option B: CUDA 12.4 GPU Installation (*untested on this CPU machine*)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics>=8.3.0
```
*(Reference official PyTorch selector: https://pytorch.org/get-started/locally/)*

### Node.js Relay Setup
```bash
npm install
```

### Model Weights Setup
- **Expected Location**: `weights/yolo11n.pt` (overridable via `YOLO_MODEL_PATH`).
- **Download Command**:
  ```bash
  mkdir -p weights
  curl -L -o weights/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
  ```
- **Optional ONNX Export**:
  ```bash
  yolo export model=weights/yolo11n.pt format=onnx
  ```
- **Offline / Missing Weights Fallback**: If weights or deep learning frameworks are unavailable (e.g. offline sandbox or airgapped edge deployment), startup completes safely without blocking, `/health` reports `models_loaded: false`, and inference routes return a machine-readable `HTTP 503 (MODEL_UNAVAILABLE)` error documenting the manual download step.

---

## 4. Configuration & Environment Variables

All settings can be configured via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Backend host to bind |
| `PORT` | `8765` | FastAPI backend port |
| `RELAY_PORT` | `3000` | Node.js relay port |
| `PYTHON_BACKEND_URL` | `http://127.0.0.1:8765` | Backend URL for Node relay forwarding |
| `YOLO_MODEL_PATH` | `weights/yolo11n.pt` | Path to YOLO11n weights (`.pt` or `.onnx`) |
| `YOLO_CONF_THRESHOLD` | `0.25` | Object detection confidence floor |
| `YOLO_IOU_THRESHOLD` | `0.45` | NMS and tracking IoU threshold |
| `YOLO_IMGSZ` | `640` | Max side dimension for YOLO letterbox inference |
| `YOLO_DEVICE` | `auto` | Inference device: `auto`, `cpu`, or `cuda:0` |
| `YOLO_MAX_DET` | `100` | Maximum detections per frame |
| `YOLO_CONCURRENCY_LIMIT` | `1` | Max concurrent thread inferences |
| `DEPTH_MODEL_NAME` | `depth-anything/Depth-Anything-V2-Small-hf` | Hugging Face depth model identifier |
| `NEAR_THRESHOLD` | `0.66` | Normalized depth threshold for `NEAR` (>=0.66) |
| `MID_THRESHOLD` | `0.33` | Normalized depth threshold for `MID-RANGE` (0.33 to 0.66) |
| `FLAT_DEPTH_EPSILON` | `0.02` | Flat depth map threshold; below this returns `UNKNOWN` |
| `SCENE_RETENTION_FRAMES` | `5` | Consecutive missed frames before RAM memory eviction |
| `REVIEW_CACHE_TTL_SECONDS` | `180` | Lifetime of cached review crops before 410 Gone |
| `UPLOAD_RETENTION_HOURS` | `24` | Uploaded video/image disk retention |
| `DB_PATH` | `storage/byaku.db` | SQLite database file location |

---

## 5. How to Run & Test

### Run the Application
Start both the FastAPI backend (port 8765) and Node.js website/relay (port 3000) from the project directory:
   ```bash
   npm start
   ```
Open your browser at **`http://localhost:3000`** (or `http://127.0.0.1:3000`). Press **Ctrl+C** in the terminal to stop both services.

The launcher starts both services together and forwards their logs to the same terminal. To run only one service for development, use `npm run start:relay` for the Node.js relay, or run `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765` for the backend.

### Run Test Suites by Group
- **Run Mocked and Synthetic Backend Tests (38 tests)**:
  ```bash
  pytest -m "not realmodel"
  ```
- **Run Real Model Weight Tests (auto-skips if weights/ultralytics absent)**:
  ```bash
  pytest -m realmodel -rs
  ```
- **Run Frontend and Relay Integration Tests (41 tests)**:
  ```bash
  npm test
  ```

---

## 6. API Reference

All backend responses use the standard envelope:
```json
{
  "ok": true,
  "data": { ... },
  "error": null
}
```

### Endpoints Summary
- `GET /health` / `GET /api/health`: System health, model load status, versions, device, SQLite status.
- `GET /api/status`: Real-time pipeline status for the dashboard.
- `POST /api/sessions/live/start`: Create a new live camera session with an isolated ByteTrack instance and RAM scene memory.
- `POST /api/sessions/live/{id}/stop`: Terminate session and flush tracker and scene memory.
- `POST /api/live/{session_id}/frame`: Process a live camera frame. Returns detections, `bounding_box`, `center_x`, `center_y`, `inference_ms`, `device`, `model`.
- `GET /api/sessions/{id}/scene`: Get current and recent objects in session memory.
- `POST /api/sessions/{id}/ask`: Ask a voice/text question about the scene memory.
- `POST /api/image` (and `/api/upload/image`): Upload still image for detection and spatial reasoning (`track_id: null`).
- `POST /api/video` (and `/api/upload/video`): Upload video for background/polled sampling and tracking.
- `GET /api/video/{job_id}/status`: Poll video background job status and progress.
- `GET /api/video/{job_id}/result`: Retrieve sampled video analysis results.
- `POST /api/reviews` (and `/api/feedback`): Submit human review (`confirmed`, `corrected`, `rejected`).
- `GET /api/reviews`: List reviews with filters (`session_id`, `source`, `action`, `date_from`, `date_to`, `limit`, `offset`).
- `POST /api/reviews/{id}/undo`: Mark review undone and append an `undone` audit event (non-destructive).
- `GET /api/reviews/status`: SQLite review counts, active/undone stats, and cache stats.
- `GET /api/reviews/export`: Export active confirmed and corrected reviews in COCO-style JSON dataset format. Corrected annotations use the corrected label and, when provided, the corrected bounding box; otherwise they use the original box.
- `GET /api/audit`: Query append-only audit event log.
- `GET /api/review-frames/{frame_id}`: Retrieve cached review crop/frame. Returns `410 Gone` if expired (>180s).
- `WS /ws`: Real-time live camera streaming and voice Q&A socket.

---

## 7. Depth Convention & Spatial Rules

- **Relative Inverse Depth**: Depth Anything V2 outputs inverse depth where **larger values indicate closer objects**.
- **Normalization**: Values are min-max normalized per frame to $[0.0, 1.0]$ where $1.0 = \text{closest}$.
- **Object Sampling**: Depth is calculated as the **median depth within the central 50% of the bounding box**, making it robust to background bleed.
- **Classification Categories**:
  - `NEAR`: Depth $\ge 0.66$
  - `MID-RANGE`: $0.33 \le \text{Depth} < 0.66$
  - `FAR`: $\text{Depth} < 0.33$
- **Unknown Fallback**: If the bounding box is tiny ($<10$ px), the depth map is nearly flat ($\Delta < 0.02$), or the model is unavailable, the system outputs `depth_class: "UNKNOWN"` and `depth_confidence: "low"`.
- **Direction**:
  - Horizontal: `LEFT` ($<0.33$), `CENTER` ($0.33 - 0.67$), `RIGHT` ($>0.67$)
  - Vertical: `TOP` ($<0.33$), `MIDDLE` ($0.33 - 0.67$), `BOTTOM` ($>0.67$)

---

## 8. Known Limitations

1. **COCO-80 Class Vocabulary**: Default YOLO11n weights detect only the 80 standard COCO object categories. Custom objects require a drop-in fine-tuned YOLO11n weight file via `YOLO_MODEL_PATH`.
2. **CPU Inference Latency**: On standard dual-core CPUs without GPU acceleration, YOLO neural inference latency is approximately 15-40 ms per frame.
3. **Small or Distant Objects**: Objects occupying tiny pixel areas (e.g. $< 15$ pixels) or heavily occluded objects may be missed by the nano architecture.
4. **Relative Depth Only**: Depth estimates represent relative foreground/background ordering within the specific camera frame and cannot be used for metric distance measurement.
5. **Browser Constraints**: `getUserMedia` and `webkitSpeechRecognition` require HTTPS or `localhost`/`127.0.0.1`.
