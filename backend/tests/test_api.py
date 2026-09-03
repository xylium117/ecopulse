import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app
from backend import gee_utils
from backend.model import WildfireSegmenter

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ecopulse-api"


def test_config_endpoint():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "earth_engine" in data
    assert "segmentation_model" in data
    assert "mapbox_token_configured" in data


def test_metrics_endpoint_global():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "spatial_resolution" in data
    assert any("Sentinel-2" in s for s in data["sensors"])
    assert "tile_stream_ms" in data
    assert "carbon_flux_rate" in data
    assert "active_anomalies_flagged" in data


def test_metrics_endpoint_with_viewport_bounds():
    params = {
        "lon_min": -65.0,
        "lat_min": -10.0,
        "lon_max": -50.0,
        "lat_max": 0.0,
    }
    response = client.get("/api/metrics", params=params)
    assert response.status_code == 200
    data = response.json()
    assert data["active_anomalies_flagged"] >= 0
    assert float(data["carbon_flux_rate"]) is not None


def test_ndvi_endpoint_valid():
    params = {
        "lon_min": -63.2,
        "lat_min": -5.2,
        "lon_max": -61.8,
        "lat_max": -3.8,
        "start_date": "2025-01-01",
        "end_date": "2026-01-01",
    }
    response = client.get("/api/ndvi", params=params)
    assert response.status_code == 200
    data = response.json()
    assert len(data["series"]) > 0
    assert "mean_ndvi" in data
    assert "carbon_flux_status" in data
    assert "source" in data
    assert data["anomaly_count"] >= 0

    first_pt = data["series"][0]
    assert "date" in first_pt
    assert "ndvi" in first_pt
    assert "ndwi" in first_pt
    assert "carbon_flux" in first_pt


def test_ndvi_endpoint_invalid_bbox():
    params = {
        "lon_min": -60.0,
        "lat_min": -5.0,
        "lon_max": -65.0,
        "lat_max": -3.0,
        "start_date": "2025-01-01",
        "end_date": "2026-01-01",
    }
    response = client.get("/api/ndvi", params=params)
    assert response.status_code == 400
    assert "Invalid bounding box" in response.json()["detail"]


def test_ndvi_endpoint_inverted_dates():
    params = {
        "lon_min": -63.0,
        "lat_min": -5.0,
        "lon_max": -60.0,
        "lat_max": -3.0,
        "start_date": "2026-01-01",
        "end_date": "2025-01-01",
    }
    response = client.get("/api/ndvi", params=params)
    assert response.status_code == 400
    assert "start_date must be before end_date" in response.json()["detail"]


def test_drought_endpoint_valid():
    params = {
        "lon_min": -63.2,
        "lat_min": -5.2,
        "lon_max": -61.8,
        "lat_max": -3.8,
    }
    response = client.get("/api/drought", params=params)
    assert response.status_code == 200
    data = response.json()
    assert "vci_percentage" in data
    assert "drought_class" in data
    assert "risk_level" in data
    assert "soil_moisture_proxy_kpa" in data
    assert "recommended_action" in data
    assert 0 <= data["vci_percentage"] <= 100


def test_drought_endpoint_invalid_coords():
    params = {
        "lon_min": 10.0,
        "lat_min": 20.0,
        "lon_max": 5.0,
        "lat_max": 25.0,
    }
    response = client.get("/api/drought", params=params)
    assert response.status_code == 400


def test_alerts_endpoint_global():
    response = client.get("/api/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    assert len(alerts) >= 5
    first = alerts[0]
    assert "id" in first
    assert "title" in first
    assert "severity" in first
    assert "loss_hectares" in first
    assert "co2_emissions_kt" in first
    assert "detected_at" in first
    assert "live_active" in first


def test_alerts_endpoint_with_viewport():
    params = {
        "lon_min": -60.0,
        "lat_min": -10.0,
        "lon_max": -50.0,
        "lat_max": 0.0,
    }
    response = client.get("/api/alerts", params=params)
    assert response.status_code == 200
    alerts = response.json()
    assert len(alerts) >= 6
    assert "Active AOI" in alerts[0]["title"] or "Live Viewport" in alerts[0]["region"]


@pytest.mark.parametrize("preset_name", ["california", "amazon", "borneo"])
def test_wildfire_inference_presets(preset_name):
    response = client.post(f"/api/inference/wildfire?preset={preset_name}")
    assert response.status_code == 200
    data = response.json()
    assert data["preset"] == preset_name
    assert "spatial_resolution" in data
    assert data["burned_area_hectares"] > 0
    assert "visuals" in data
    assert "pre_scene_b64" in data["visuals"]
    assert "post_scene_b64" in data["visuals"]
    assert "overlay_b64" in data["visuals"]
    assert "geojson" in data


def test_wildfire_inference_viewport_scan():
    params = {
        "preset": "viewport",
        "lon_min": -56.0,
        "lat_min": -8.0,
        "lon_max": -54.0,
        "lat_max": -6.0,
        "region_name": "Custom Viewport Zone",
    }
    response = client.post("/api/inference/wildfire", params=params)
    assert response.status_code == 200
    data = response.json()
    assert "Custom Viewport Zone" in data["title"] or "Amazon Rainforest" in data["title"]
    assert data["burned_area_hectares"] > 0


@pytest.mark.parametrize("layer", ["ndvi", "carbon", "drought", "burn", "heatmap"])
def test_tile_layers_valid(layer):
    response = client.get(f"/api/tiles/{layer}/4/4/7.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 100

    img = Image.open(io.BytesIO(response.content))
    assert img.size == (256, 256)
    assert img.mode == "RGBA"


def test_default_tile_route():
    response = client.get("/api/tiles/4/4/7.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_export_endpoint():
    response = client.get("/api/export?region=Amazon%20Basin")
    assert response.status_code == 200
    data = response.json()
    assert "report_title" in data
    assert "Amazon Basin" in data["report_title"]
    assert "drought_assessment" in data
    assert "total_observations" in data
    assert "total_anomalies" in data
    assert "timeseries_sample" in data


def test_is_land_region_classification():
    assert gee_utils.is_land_region(-3.4, -62.2) is True
    assert gee_utils.is_land_region(37.7, -119.5) is True
    assert gee_utils.is_land_region(48.8, 2.3) is True
    assert gee_utils.is_land_region(62.0, 129.7) is True

    assert gee_utils.is_land_region(0.0, -140.0) is False
    assert gee_utils.is_land_region(0.0, -30.0) is False


def test_flag_anomalies_zscore():
    sample_series = [
        {"date": f"2025-{i:02d}-01", "ndvi": 0.75 + (0.01 if i % 2 == 0 else -0.01), "ndwi": 0.40, "carbon_flux": -0.8}
        for i in range(1, 12)
    ]
    sample_series.append({"date": "2025-12-01", "ndvi": 0.10, "ndwi": -0.30, "carbon_flux": 5.2})

    flagged = gee_utils.flag_anomalies(sample_series, z_threshold=1.8)
    assert len(flagged) == 12
    assert flagged[-1]["anomaly"] is True
    assert flagged[0]["anomaly"] is False
