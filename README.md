# EcoPulse

EcoPulse is an interactive planetary climate analytics engine and environmental monitoring platform. Monitor global deforestation, track multi-spectral vegetation indices, detect carbon flux anomalies, assess agricultural drought risk, and segment wildfire burn scars using spatio-temporal deep learning.

The project combines a Python FastAPI telemetry engine, a Spatio-Temporal U-Net with ConvLSTM2D bottlenecks, Google Earth Engine multi-spectral ingestion, and a modern glassmorphic dashboard powered by Leaflet and Mapbox 3D Globe visualizations.

---
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)&nbsp;
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)&nbsp;
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)&nbsp;
![Leaflet](https://img.shields.io/badge/Leaflet-199900?style=for-the-badge&logo=Leaflet&logoColor=white)&nbsp;
![Mapbox](https://img.shields.io/badge/Mapbox-000000?style=for-the-badge&logo=mapbox&logoColor=white)&nbsp;
[![License: MIT](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](https://github.com/xylium117/ecopulse/blob/main/LICENSE)


## What You Can Explore

- Multi-spectral NDVI, NDWI, and carbon flux time series analysis across global biomes
- Real-time global deforestation, wildfire, and carbon spike alert feeds with anomaly scoring
- Dual-engine map visualization: Open Satellite Engine (ESRI World Imagery + CartoDB Dark Matter, zero API keys required) and Mapbox 3D Globe with atmospheric shaders
- Spatio-temporal U-Net deep learning segmentation for multi-spectral burn-scar and canopy loss detection
- Interactive VCI (Vegetation Condition Index) agricultural drought simulator with real-time sensitivity controls
- Dynamic XYZ multi-spectral raster tile streaming for NDVI, carbon flux, drought index, and burn severity layers
- High-fidelity deterministic synthetic fallback mode for zero-configuration, out-of-the-box local operation
- Multi-sensor ingestion support for Sentinel-2 MSI (10m) and Landsat 8/9 OLI (30m) surface reflectance products
- Exportable planetary telemetry reports and regional environmental impact summaries
- Glassmorphic command center interface with responsive telemetry counters, charts, and interactive layer compositing

## Requirements

- Python 3.10 or newer and `pip`
- Modern web browser with WebGL support (for 3D globe and interactive map layers)
- Docker & Docker Compose (optional, for containerized multi-service deployment)
- Google Cloud / Earth Engine credentials (optional; fallback mode activates automatically if omitted)
- Mapbox GL public token (optional; Leaflet Open Satellite engine is enabled by default)

## Run the Interface

Serve the frontend directory with any static web server:

```powershell
cd frontend
python -m http.server 8080
```

Open [http://localhost:8080](http://localhost:8080).

The frontend runs out of the box with the default **Open Satellite Engine** using high-resolution ESRI satellite imagery. When the backend is offline or unconfigured, the interface automatically leverages synthetic telemetry curves and client-side fallbacks.

## Run the API

From the repository root, install dependencies and start the FastAPI server:

```powershell
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

The API listens on `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Available routes include:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service health and readiness |
| `GET` | `/api/config` | Inspect telemetry provider and credentials status |
| `GET` | `/api/metrics` | Stream global and viewport-scoped headline metrics |
| `GET` | `/api/ndvi` | Query multi-spectral NDVI/NDWI/Carbon-flux time series |
| `GET` | `/api/drought` | Fetch agricultural drought vulnerability (VCI & soil moisture) |
| `GET` | `/api/alerts` | Stream real-time global deforestation and wildfire alerts |
| `POST` | `/api/inference/wildfire` | Run spatio-temporal U-Net burn-scar segmentation |
| `GET` | `/api/tiles/{layer}/{z}/{x}/{y}.png` | Fetch dynamic XYZ raster tiles (`ndvi`, `carbon`, `drought`, `burn`) |
| `GET` | `/api/export` | Generate and export summary telemetry reports |

## Build and Test

Run the full container stack with Docker Compose:

```powershell
docker compose up --build -d
```

- **Frontend Dashboard**: `http://localhost:8080` (reverse-proxies `/api/` to the backend)
- **Backend API**: `http://localhost:8000`

Run the automated backend test suite:

```powershell
python -m pytest backend/tests -v
```

## Using the Planetary Command Center

Open the dashboard to interact with real-time planetary observations:

- **Satellite Engine Selector**: Toggle between the high-performance **Open Satellite Engine** (Leaflet + ESRI World Imagery) and the **Mapbox 3D Globe** with atmospheric shaders.
- **Spectral Overlay Layers**: Composite NDVI vegetation density, estimated carbon flux, agricultural drought indices, and wildfire burn severity directly onto the map.
- **Vegetation Time Series**: Select predefined biomes (e.g., Amazon Basin, Congo Rainforest, California Forests, Great Plains) or define custom geographic bounding boxes to inspect multi-year spectral curves.
- **VCI Drought Simulator**: Adjust the Vegetation Condition Index threshold slider in real time to simulate drought vulnerability, soil moisture depletion, and crop stress.
- **Deep Learning Wildfire Segmentation**: Upload paired pre- and post-fire multi-spectral granules or test sample scenes to segment active burn scars with the Spatio-Temporal U-Net.

## Multi-Spectral Ingestion & Fallback Mode

EcoPulse operates in two modes:

1. **Live Production Mode**: Connects directly to Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED` at 10m and `LANDSAT/LC08/C02/T1_L2` at 30m) and Mapbox satellite services when API credentials are provided.
2. **Deterministic Synthetic Fallback Mode**: When running without Earth Engine credentials or offline, the engine transparently computes high-fidelity mathematical approximations based on latitude, seasonal harmonic cycles, and biome baselines.

To configure live credentials, create a `.env` file from the provided template:

```powershell
cp .env.example .env
```

## GitHub Pages

The frontend dashboard is deployed automatically via [.github/workflows/deploy-gh-pages.yml](.github/workflows/deploy-gh-pages.yml).

To deploy your own repository:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/ecopulse.git
git branch -M main
git add .
git commit -m "Deploy EcoPulse"
git push -u origin main
```

In GitHub, open **Settings → Pages** and select **GitHub Actions** as the source.

For the `xylium117/ecopulse` repository, the expected Pages URL is:

```text
https://xylium117.github.io/ecopulse/
```

GitHub Pages hosts the static frontend dashboard. When deployed statically without a backend instance, the application operates in demo mode using client-side telemetry simulation.

## Development Notes

- The spatio-temporal segmentation model accepts input tensors of shape `(Batch, Time=2, Height=256, Width=256, Channels=3)` representing pre- and post-disturbance scenes.
- Dynamic XYZ raster tiles are computed using Mercator tile-to-lat/lon bounding box conversions with color mapping palettes.
- The default Open Satellite engine requires zero external API keys and runs purely on open GIS endpoints.
- Ensure all environment variables and secrets (such as GEE service account keys) remain excluded from version control.

## Roadmap

- [ ] Add real-time Sentinel-5P TROPOMI carbon monoxide (CO) and methane ($CH_4$) atmospheric trace gas layers
- [ ] Implement browser-side ONNX Runtime Web inference for edge segmentation without server round-trips
- [ ] Expand the deep learning pipeline to include multi-modal SAR (Sentinel-1 GRD) cloud-penetrating radar data
- [ ] Add automated Webhook and GeoJSON subscription endpoints for deforestation and wildfire alert dispatching
- [ ] Introduce custom polygon drawing tools for arbitrary multi-spectral area calculations

See the [open issues](https://github.com/xylium117/ecopulse/issues) for a full list of proposed features (and known issues).

## License

This repository is licensed under the [MIT License](LICENSE). Feel free to use and modify the code as you see fit.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes and add tests where practical.
4. Run `python -m pytest backend/tests` to verify test coverage.
5. Open a pull request with a concise description of the change.

---
