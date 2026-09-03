# EcoPulse — Planetary Climate Analytics Engine

> **Planetary climate analytics engine with NDVI vegetation tracking & carbon flux anomaly alerts**  
> Aggregates multi-spectral satellite imagery from Sentinel-2 and Landsat missions to monitor global deforestation, carbon emissions, and agricultural drought risk with deep learning segmentation.

```
ecopulse/
├── .env.example              # Root environment variables template (Mapbox & GEE)
├── .env                      # Local environment configuration
├── docker-compose.yml        # Orchestrated multi-container production deployment
├── README.md                 # System architecture, API docs & deployment guide
├── backend/
│   ├── .env.example          # Backend-specific environment template
│   ├── app.py                # FastAPI REST API (NDVI, Drought, Alerts, Inference, Tiles)
│   ├── gee_utils.py          # Google Earth Engine & multi-spectral telemetry pipeline
│   ├── model.py              # Spatio-Temporal U-Net with ConvLSTM2D bottleneck
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Container definition for FastAPI backend
│   └── tests/
│       └── test_api.py       # Automated endpoint & pipeline test suite
└── frontend/
    ├── index.html            # Clean semantic HTML5 dashboard
    ├── nginx.conf            # Production Nginx reverse-proxy & caching configuration
    ├── css/
    │   └── style.css         # Glassmorphic dark command center design system
    ├── js/
    │   └── app.js            # Open Satellite (Leaflet) & Mapbox 3D Globe telemetry logic
    └── Dockerfile            # Nginx container for static frontend
```

---

## 1. Technologies & Stack

- **Python 3.11+ & FastAPI**: High-throughput asynchronous REST API for multi-spectral telemetry streaming and inference.
- **TensorFlow 2.x & Deep Learning**: Spatio-temporal U-Net with ConvLSTM2D temporal bottleneck for multi-spectral burn-scar and deforestation segmentation.
- **Google Earth Engine (GEE)**: Surface Reflectance ingestion from `COPERNICUS/S2_SR_HARMONIZED` (10m) and `LANDSAT/LC08/C02/T1_L2` (30m).
- **Dual Map Engine Support**:
  - **Open Satellite Engine (Default)**: Powered by Leaflet + High-Resolution ESRI World Imagery + CartoDB Dark Matter labels. **Requires 0 API keys and works out of the box.**
  - **Mapbox 3D Globe Engine**: Interactive 3D globe projection with atmospheric space shaders.
- **Interactive VCI Drought Simulator**: Real-time Vegetation Condition Index threshold sensitivity slider.
- **Vanilla CSS3 & Modern JavaScript**: Decoupled, responsive, glassmorphic planetary command center dashboard.

---

## 2. API Keys & Environment Configuration

EcoPulse operates in two modes:
1. **Live Production Mode**: Connects directly to Google Earth Engine and Mapbox satellite basemap tiles using API keys.
2. **High-Fidelity Demo Fallback Mode**: If API credentials are not configured, the engine transparently serves deterministic synthetic telemetry, seasonal vegetation curves, and algorithmic inference out of the box.

Create a `.env` file in the root directory (or copy `.env.example`):

```bash
cp .env.example .env
```

### Environment Variables Reference

| Variable | Required | Description | Where to Obtain |
|---|:---:|---|---|
| `GEE_API_KEY` | Optional | Google Cloud API Key for Earth Engine REST endpoints | [Google Cloud Console](https://console.cloud.google.com/) |
| `GEE_SERVICE_ACCOUNT` | Optional | Google Cloud Service Account email for Earth Engine Python API | [Google Cloud Console](https://console.cloud.google.com/) |
| `GEE_CREDENTIALS_PATH` | Optional | Path to Service Account JSON key (`./secrets/gee_credentials.json`) | GCP Service Account Keys |
| `GEE_PROJECT` | Optional | Google Cloud Project ID registered with Earth Engine | GCP Project Settings |
| `MAPBOX_TOKEN` | Optional | Public Mapbox GL token for optional 3D globe basemap | Free at [mapbox.com](https://account.mapbox.com/) → Tokens |
| `MODEL_WEIGHTS_PATH` | Optional | Path to trained `.h5` / `.keras` U-Net weights file | Defaults to `./backend/weights/unet_burn.h5` |
| `PORT` | Optional | Backend server port (defaults to `8000`) | — |
| `CORS_ORIGINS` | Optional | Allowed CORS origins (defaults to `*`) | — |

---

## 3. Running Locally (Without Docker)

### Step 1: Start Backend API

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. Launch FastAPI server
uvicorn backend.app:app --reload --port 8000
```
- API is live at `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### Step 2: Serve the Frontend

```bash
cd frontend
python -m http.server 8080
```
- Open `http://localhost:8080` in your web browser.
- Open Satellite Engine is active by default. You can switch to Mapbox 3D Globe via the map engine toggle.

---

## 4. Production Deployment

### Option A: Docker Compose (Recommended Full-Stack)

```bash
# Build and run both backend and frontend containers
docker compose up --build -d
```
- **Frontend Dashboard**: `http://localhost:8080` (Nginx reverse-proxies `/api/` to `backend:8000`)
- **Backend API**: `http://localhost:8000`

### Option B: Cloud Deployment (Render / Railway / Fly.io / VPS)

1. **Backend Service**:
   - Set Build Command: `pip install -r backend/requirements.txt`
   - Set Start Command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - Add environment variables: `GEE_API_KEY`, `GEE_PROJECT`, `CORS_ORIGINS=*`
2. **Frontend Service**:
   - Deploy `frontend/` directory to static hosting (e.g. Vercel, Netlify, Cloudflare Pages).
   - In `frontend/index.html` or browser console, set `window.ECOPULSE_API_BASE = "https://your-backend-url.com"`.

---

## 5. API Endpoints Reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness & service readiness probe |
| `GET` | `/api/config` | Telemetry provider & credentials status |
| `GET` | `/api/metrics` | Headline metrics (Tile Stream Speed, Sensors, Carbon Flux, Anomaly Counts) |
| `GET` | `/api/ndvi` | Multi-spectral NDVI/NDWI/Carbon-flux time series & anomaly detection |
| `GET` | `/api/drought` | Agricultural drought vulnerability index (VCI & soil moisture) |
| `GET` | `/api/alerts` | Real-time global deforestation, carbon spike & wildfire alert feed |
| `POST` | `/api/inference/wildfire` | Spatio-temporal U-Net burn-scar & canopy loss segmentation |
| `GET` | `/api/tiles/{layer}/{z}/{x}/{y}.png` | Dynamic XYZ raster tiles (`ndvi`, `carbon`, `drought`, `burn`) |
| `GET` | `/api/export` | Summary telemetry report export |

### Example NDVI Query
```bash
curl "http://localhost:8000/api/ndvi?lon_min=-63.2&lat_min=-5.2&lon_max=-61.8&lat_max=-3.8&start_date=2025-01-01&end_date=2026-01-01"
```

---

## 6. Deep Learning Architecture (Spatio-Temporal U-Net)

- **Input Dimension**: `(Batch, Time=2, Height=256, Width=256, Channels=3)` representing pre-fire and post-fire multi-spectral observations.
- **TimeDistributed Encoder**: Shared 2D convolutional blocks with batch normalization capturing spatial features per timestamp.
- **ConvLSTM2D Bottleneck**: Attends directly to temporal transitions and spectral delta changes ($\Delta\text{NBR}$) between observations.
- **Decoder**: Transposed 2D convolutions with skip connections from post-event encoder activations.
- **Inference Latency**: Sub-35ms GPU/CPU inference processing 10m Sentinel-2 multi-spectral scene granules.

---

## 7. Testing

Run the automated backend test suite with pytest:

```bash
python -m pytest backend/tests -v
```
