# VisionSuite Web Platform

Production-ready web application for real-time computer vision. Converted from PySide6 desktop to a modern web stack with React/Next.js frontend and FastAPI backend.

## Architecture

```
Browser (Vercel Next.js)
    ↓ REST API / WebSocket
FastAPI Backend (Render/Railway/VPS)
    ↓ AI Inference
VisionSuite Core (YOLO, OpenCV, MediaPipe, OCR, ANPR)
```

## Project Structure

```
visionsuite/
├── frontend/                  # Next.js web frontend (deploy to Vercel)
│   ├── app/                   # App Router pages
│   │   ├── dashboard/         # Dashboard overview
│   │   ├── camera/            # Live camera detection
│   │   ├── image-detection/   # Image upload detection
│   │   ├── video-detection/   # Video upload detection
│   │   ├── anpr/              # License plate recognition
│   │   ├── ocr/               # OCR text extraction
│   │   ├── history/           # Detection history
│   │   ├── analytics/         # Analytics dashboard
│   │   └── settings/          # System settings
│   ├── components/            # Reusable UI components
│   ├── lib/                   # API client, types, hooks
│   ├── package.json
│   ├── tailwind.config.js
│   └── .env.example
│
├── backend/                   # FastAPI backend (deploy separately)
│   ├── app/
│   │   ├── main.py            # FastAPI entry point
│   │   ├── core/              # Config, logging
│   │   ├── api/               # REST endpoints + WebSocket
│   │   ├── ai/                # AI pipeline (reuses existing modules)
│   │   ├── models/            # Pydantic schemas + ORM
│   │   ├── services/          # Storage, database
│   │   └── websocket/         # Live detection WebSocket
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── modules/                   # EXISTING AI modules (preserved)
├── detector.py                # EXISTING detection backends
├── camera.py                  # EXISTING camera/media pipeline
├── tracker.py                 # EXISTING centroid tracker
├── processing.py              # EXISTING image filters
├── utils.py                   # EXISTING helpers
├── face_db.py                 # EXISTING face identity DB
├── database.py                # EXISTING PostgreSQL layer
├── repositories.py            # EXISTING data access
├── requirements.txt           # EXISTING desktop requirements
├── models/                    # YOLO weights (preserved)
└── .env.example               # EXISTING env config
```

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (optional, for history persistence)
- Tesseract OCR (optional, for OCR features)
- ZBar (optional, for barcode scanning)

### Backend Setup

```bash
cd backend

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with your backend URL

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Full Stack Setup

1. Start PostgreSQL (if using database)
2. Start backend: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
3. Start frontend: `cd frontend && npm run dev`
4. Open `http://localhost:3000`

## Environment Variables

### Backend (.env)

```env
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/visionsuite
REDIS_URL=redis://localhost:6379/0
STORAGE_URL=
STORAGE_KEY=
STORAGE_SECRET=
STORAGE_BUCKET=
UPLOAD_DIR=/tmp/visionsuite_uploads
OUTPUT_DIR=/tmp/visionsuite_output
CORS_ORIGINS=http://localhost:3000,https://your-app.vercel.app
YOLO_WEIGHTS_DIR=./models
PLATE_MODEL_DIR=./models/license_plate
DEVICE=cpu
CONFIDENCE_THRESHOLD=0.45
MAX_UPLOAD_SIZE_MB=50
MAX_VIDEO_DURATION_SECONDS=300
ENABLE_EASYOCR=true
ENABLE_TESSERACT=false
ENABLE_FACE_RECOGNITION=false
ENABLE_GPU=false
SECRET_KEY=your-secret-key
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect/image` | Upload image for detection |
| POST | `/api/detect/video` | Upload video for detection |
| POST | `/api/detect/frame` | Send base64 frame for detection |
| POST | `/api/anpr/image` | Upload image for ANPR |
| POST | `/api/anpr/video` | Upload video for ANPR |
| POST | `/api/ocr` | Upload image for OCR |
| GET | `/api/health` | Health check |
| GET | `/api/models` | List available models |
| GET | `/api/history/detections` | Detection history |
| GET | `/api/history/plates` | ANPR plate history |
| GET | `/api/history/dashboard` | Dashboard statistics |
| WS | `/ws/detection` | Live detection WebSocket |
| WS | `/ws/camera` | Camera stream WebSocket |

## AI Modules

### Currently Functional

| Module | Status | Model |
|--------|--------|-------|
| YOLO Object Detection | ✅ Available | yolov8n.pt / yolov11n.pt |
| Face Detection | ✅ Available | MediaPipe Tasks |
| Hand Tracking | ✅ Available | MediaPipe Tasks |
| Pose Estimation | ✅ Available | MediaPipe Tasks |
| QR/Barcode | ✅ Available | pyzbar + OpenCV fallback |
| ANPR | ✅ Available | Dedicated plate YOLO + EasyOCR/Tesseract |
| OCR | ✅ Available | Tesseract |
| Fire/Smoke | ✅ Available | Heuristic (no model needed) |
| Helmet Detection | ⚠️ Requires model download | YOLO/ONNX |
| PPE Detection | ⚠️ Requires model download | YOLO |
| Face Recognition | ⚠️ Requires face-recognition lib | MediaPipe + embeddings |
| Emotion Detection | ⚠️ Requires ONNX model | emotion-ferplus-8.onnx |
| Age/Gender | ⚠️ Requires ONNX model | age_gender.onnx |
| Sign Language | ✅ Available | MediaPipe Tasks heuristic |
| Vehicle Tracking | ✅ Available | YOLO COCO + centroid tracker |
| Animal Detection | ✅ Available | YOLO COCO classes |
| Currency Note | ⚠️ Requires ONNX model | currency.onnx |
| Plant Disease | ✅ Available | Heuristic classifier |
| Document Scanner | ✅ Available | OpenCV perspective + OCR |

## Deployment

### Vercel (Frontend)

1. Push frontend/ to a separate repository or deploy from the frontend folder
2. Connect to Vercel
3. Set environment variables:
   - `NEXT_PUBLIC_API_URL` = your backend URL
   - `NEXT_PUBLIC_WS_URL` = your backend WebSocket URL
4. Deploy

### Backend Deployment Options

#### Render
1. Connect your repository
2. Select "Docker" as runtime
3. Use the provided `Dockerfile`
4. Set environment variables
5. Deploy

#### Railway
1. Connect repository
2. Railway auto-detects Dockerfile
3. Set environment variables
4. Deploy

#### VPS / GPU Server
```bash
# Install dependencies
sudo apt update && sudo apt install -y python3.11 python3.11-venv libgl1 libglib2.0-0

# Clone and setup
git clone <repo>
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run with production ASGI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Database Setup

```sql
-- Create database
CREATE DATABASE visionsuite;

-- Create user
CREATE USER visionsuite_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE visionsuite TO visionsuite_user;
```

## Important Notes

1. **DO NOT run PySide6/QApplication in the web version** — the existing desktop code is preserved for desktop use only
2. **DO NOT use cv2.VideoCapture() for browser camera** — the frontend uses browser MediaDevices API
3. **Existing AI modules are reused** — no detection algorithms were rewritten
4. **Models auto-download** — YOLO weights and MediaPipe models download on first use
5. **Database is optional** — the backend works without PostgreSQL; history is lost between restarts
6. **Large video processing** — for production, implement Celery + Redis for background video jobs

## Missing Models

The following models may need to be downloaded:
- `models/yolov8n.pt` — auto-downloaded by ultralytics
- `models/yolov8s.pt` — auto-downloaded by ultralytics
- `models/yolov8m.pt` — auto-downloaded by ultralytics
- `models/face_detector.task` — auto-downloaded by mediapipe
- `models/hand_landmarker.task` — auto-downloaded by mediapipe
- `models/pose_landmarker.task` — auto-downloaded by mediapipe
- `models/license_plate/plate_detector_yolov8n.pt` — auto-downloaded from HuggingFace
- `models/emotion-ferplus-8.onnx` — downloaded on demand
- `models/age_gender.onnx` — downloaded on demand
- `models/mask_model.onnx` — downloaded on demand
- `models/helmet.onnx` — downloaded on demand
- `models/currency.onnx` — downloaded on demand

## Verification Steps

After setup:
1. Backend health: `curl http://localhost:8000/api/health`
2. API docs: Open `http://localhost:8000/docs`
3. Frontend: Open `http://localhost:3000`
4. Test image detection: Upload an image on the Image Detection page
5. Test camera: Go to Live Camera page, allow permission, start camera
6. Test ANPR: Upload a vehicle image on the ANPR page
7. Test OCR: Upload a document image on the OCR page

## License

MIT
