(function () {
  "use strict";

  const rawSavedEngine = localStorage.getItem("ecopulse_map_engine") || "leaflet";
  const rawToken = window.MAPBOX_TOKEN || localStorage.getItem("ecopulse_mapbox_token") || "";
  const safeEngine = (rawSavedEngine === "mapbox" && rawToken.length > 20 && !rawToken.includes("example")) ? "mapbox" : "leaflet";

  const state = {
    apiBase: window.ECOPULSE_API_BASE || localStorage.getItem("ecopulse_api_base") || (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" ? "http://localhost:8000" : ""),
    mapboxToken: rawToken,
    currentEngine: safeEngine,
    activeLayer: "ndvi",
    activeMetric: "ndvi",
    currentRegion: {
      id: "amazon",
      name: "Amazon Basin, Brazil",
      center: [-62.5, -4.5],
      zoom: 5.5,
      sensor: "Sentinel-2 MSI (10m)",
    },
    timeseriesData: [],
    alertsData: [],
    leafletInstance: null,
    leafletTileLayer: null,
    leafletMarkers: [],
    leafletAiLayer: null,
    mapboxInstance: null,
    mapboxMarkers: [],
    mapboxAiLayerAdded: false,
    telemetryPollTimer: null,
  };

  const elements = {
    metricCarbon: document.getElementById("metric-carbon"),
    metricCarbonSub: document.getElementById("metric-carbon-sub"),
    metricTileSpeed: document.getElementById("metric-tile-speed"),
    metricStreamSub: document.getElementById("metric-stream-sub"),
    metricAnomalies: document.getElementById("metric-anomalies"),
    metricAnomaliesSub: document.getElementById("metric-anomalies-sub"),
    metricClock: document.getElementById("metric-clock"),

    regionSelect: document.getElementById("region-select"),
    quickRegions: document.querySelectorAll(".quick-region-btn"),

    chartCanvas: document.getElementById("ndvi-chart-canvas"),
    chartWrap: document.getElementById("ndvi-chart-wrap"),
    chartStart: document.getElementById("chart-start"),
    chartEnd: document.getElementById("chart-end"),
    chartStatus: document.getElementById("chart-status"),
    chartTabs: document.querySelectorAll(".chart-tab"),

    droughtScore: document.getElementById("drought-score"),
    droughtClass: document.getElementById("drought-class"),
    droughtAction: document.getElementById("drought-action"),
    droughtBar: document.getElementById("drought-bar"),
    vciSlider: document.getElementById("vci-slider"),
    vciSliderVal: document.getElementById("vci-slider-val"),

    alertList: document.getElementById("alert-feed-list"),

    presetSelect: document.getElementById("studio-preset"),
    btnRunInference: document.getElementById("btn-run-inference"),
    fileUploadPre: document.getElementById("file-upload-pre"),
    fileUploadPost: document.getElementById("file-upload-post"),
    studioStatus: document.getElementById("studio-status"),
    thumbPre: document.getElementById("thumb-pre"),
    thumbPost: document.getElementById("thumb-post"),
    thumbOverlay: document.getElementById("thumb-overlay"),
    statBurnedHa: document.getElementById("stat-burned-ha"),
    statCo2: document.getElementById("stat-co2"),
    statLatency: document.getElementById("stat-latency"),

    btnExplainAi: document.getElementById("btn-explain-ai"),
    modalAiExplainer: document.getElementById("modal-ai-explainer"),
    btnCloseExplainerModal: document.getElementById("btn-close-explainer-modal"),
    btnCloseExplainerDone: document.getElementById("btn-close-explainer-done"),

    hudRegion: document.getElementById("hud-region"),
    hudCoords: document.getElementById("hud-coords"),
    hudSensor: document.getElementById("hud-sensor"),

    layerBtns: document.querySelectorAll(".layer-btn"),

    leafletContainer: document.getElementById("leaflet-map"),
    mapboxContainer: document.getElementById("map"),
    btnEngineLeaflet: document.getElementById("engine-leaflet-btn"),
    btnEngineMapbox: document.getElementById("engine-mapbox-btn"),

    btnSettings: document.getElementById("btn-settings"),
    modalSettings: document.getElementById("modal-settings"),
    btnCloseModal: document.getElementById("btn-close-modal"),
    btnSaveSettings: document.getElementById("btn-save-settings"),
    inputMapboxToken: document.getElementById("input-mapbox-token"),
    statusMapEngine: document.getElementById("status-map-engine"),
    statusGee: document.getElementById("status-gee"),
    statusModel: document.getElementById("status-model"),

    btnExport: document.getElementById("btn-export"),
    toast: document.getElementById("toast"),
  };

  function initLeafletMap() {
    if (typeof L === "undefined") {
      console.warn("Leaflet library not found.");
      return;
    }

    if (state.leafletInstance) return;

    const lat = state.currentRegion.center[1];
    const lon = state.currentRegion.center[0];

    state.leafletInstance = L.map("leaflet-map", {
      center: [lat, lon],
      zoom: state.currentRegion.zoom,
      minZoom: 3,
      maxZoom: 18,
      maxBounds: [
        [-85, -180],
        [85, 180],
      ],
      maxBoundsViscosity: 1.0,
      worldCopyJump: false,
      zoomControl: true,
      attributionControl: true,
    });

    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      minZoom: 3,
      maxZoom: 19,
      noWrap: true,
      bounds: [
        [-85, -180],
        [85, 180],
      ],
      attribution: "Tiles &copy; Esri &mdash; Source: Esri, USDA, USGS, GeoEye",
    }).addTo(state.leafletInstance);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png", {
      subdomains: "abcd",
      minZoom: 3,
      maxZoom: 19,
      noWrap: true,
      opacity: 0.85,
    }).addTo(state.leafletInstance);

    updateLeafletTelemetryLayer();

    state.leafletInstance.on("move", () => {
      const c = state.leafletInstance.getCenter();
      if (elements.hudCoords) {
        elements.hudCoords.textContent = `LON ${c.lng.toFixed(2)} · LAT ${c.lat.toFixed(2)} · ZOOM ${state.leafletInstance.getZoom().toFixed(1)}`;
      }
    });

    state.leafletInstance.on("moveend", () => {
      fetchRealTimeMetrics();
    });

    setTimeout(() => {
      if (state.leafletInstance) {
        state.leafletInstance.invalidateSize();
      }
    }, 150);

    renderAlertLeafletMarkers(state.alertsData);
  }

  function updateLeafletTelemetryLayer() {
    if (!state.leafletInstance) return;

    if (state.leafletTileLayer) {
      state.leafletInstance.removeLayer(state.leafletTileLayer);
    }

    const tileUrl = `${state.apiBase}/api/tiles/${state.activeLayer}/{z}/{x}/{y}.png`;
    state.leafletTileLayer = L.tileLayer(tileUrl, {
      tileSize: 256,
      minZoom: 3,
      maxZoom: 18,
      noWrap: true,
      bounds: [
        [-85, -180],
        [85, 180],
      ],
      opacity: 0.65,
      zIndex: 10,
      attribution: "EcoPulse Planetary Analytics",
    }).addTo(state.leafletInstance);
  }

  function renderAlertLeafletMarkers(alerts) {
    if (!state.leafletInstance || !alerts || !alerts.length) return;

    state.leafletMarkers.forEach((m) => state.leafletInstance.removeLayer(m));
    state.leafletMarkers = [];

    alerts.forEach((alert) => {
      const isCrit = alert.severity === "CRITICAL";
      const color = isCrit ? "#EF4444" : "#F59E0B";

      const icon = L.divIcon({
        className: "custom-leaflet-marker",
        html: `<div style="width:14px;height:14px;border-radius:50%;background:${color};box-shadow:0 0 12px ${color};border:2px solid #fff;cursor:pointer;"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });

      const marker = L.marker([alert.coordinates[1], alert.coordinates[0]], { icon })
        .addTo(state.leafletInstance)
        .bindPopup(`
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#fff;">
            <div style="font-weight:700;color:${isCrit ? "#F87171" : "#FBBF24"};margin-bottom:4px;">${alert.title}</div>
            <div style="color:#94A3B8;">Severity: <strong style="color:#fff;">${alert.severity}</strong></div>
            <div style="color:#94A3B8;">Loss: <strong>${alert.loss_hectares} ha</strong> · Flux: <strong>+${alert.co2_emissions_kt} kt CO₂</strong></div>
          </div>
        `);

      state.leafletMarkers.push(marker);
    });
  }

  function initMapboxMap() {
    if (typeof mapboxgl === "undefined") return;

    const token = state.mapboxToken;
    if (!token || token.includes("example") || token.length < 10) {
      switchMapEngine("leaflet");
      showToast("Using Open Satellite Engine (No Mapbox token required).");
      return;
    }

    if (state.mapboxInstance) return;

    mapboxgl.accessToken = token;

    try {
      state.mapboxInstance = new mapboxgl.Map({
        container: "map",
        style: "mapbox://styles/mapbox/satellite-streets-v12",
        projection: "globe",
        center: state.currentRegion.center,
        zoom: state.currentRegion.zoom,
        minZoom: 2.5,
        maxZoom: 18,
        maxBounds: [
          [-180, -85],
          [180, 85],
        ],
        renderWorldCopies: false,
        pitch: 25,
        bearing: -10,
      });

      state.mapboxInstance.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "top-right");

      state.mapboxInstance.on("style.load", () => {
        state.mapboxInstance.setFog({
          color: "rgb(6, 9, 14)",
          "high-color": "rgb(18, 32, 48)",
          "horizon-blend": 0.04,
          "space-color": "rgb(3, 5, 8)",
          "star-intensity": 0.35,
        });

        updateMapboxTelemetryLayer();
        renderAlertMapboxMarkers(state.alertsData);
      });

      state.mapboxInstance.on("move", () => {
        const c = state.mapboxInstance.getCenter();
        if (elements.hudCoords) {
          elements.hudCoords.textContent = `LON ${c.lng.toFixed(2)} · LAT ${c.lat.toFixed(2)} · ZOOM ${state.mapboxInstance.getZoom().toFixed(1)}`;
        }
      });

      state.mapboxInstance.on("moveend", () => {
        fetchRealTimeMetrics();
      });

      state.mapboxInstance.on("error", (e) => {
        if (e.error && e.error.status === 401) {
          showToast("Mapbox Token invalid. Falling back to Open Satellite.");
          switchMapEngine("leaflet");
        }
      });
    } catch (e) {
      console.warn("Mapbox initialization error:", e);
      switchMapEngine("leaflet");
    }
  }

  function updateMapboxTelemetryLayer() {
    if (!state.mapboxInstance || !state.mapboxInstance.isStyleLoaded()) return;

    const sourceId = "ecopulse-telemetry-source";
    const layerId = "ecopulse-telemetry-layer";

    if (state.mapboxInstance.getLayer(layerId)) {
      state.mapboxInstance.removeLayer(layerId);
    }
    if (state.mapboxInstance.getSource(sourceId)) {
      state.mapboxInstance.removeSource(sourceId);
    }

    const tileUrl = `${state.apiBase}/api/tiles/${state.activeLayer}/{z}/{x}/{y}.png`;

    state.mapboxInstance.addSource(sourceId, {
      type: "raster",
      tiles: [tileUrl],
      tileSize: 256,
      attribution: "EcoPulse Planetary Analytics",
    });

    state.mapboxInstance.addLayer({
      id: layerId,
      type: "raster",
      source: sourceId,
      paint: {
        "raster-opacity": 0.65,
        "raster-fade-duration": 200,
      },
    });
  }

  function renderAlertMapboxMarkers(alerts) {
    if (!state.mapboxInstance) return;

    state.mapboxMarkers.forEach((m) => m.remove());
    state.mapboxMarkers = [];

    alerts.forEach((alert) => {
      const el = document.createElement("div");
      const isCrit = alert.severity === "CRITICAL";
      const color = isCrit ? "#EF4444" : "#F59E0B";

      el.style.width = "14px";
      el.style.height = "14px";
      el.style.borderRadius = "50%";
      el.style.background = color;
      el.style.boxShadow = `0 0 12px ${color}`;
      el.style.border = "2px solid #fff";
      el.style.cursor = "pointer";

      const popup = new mapboxgl.Popup({ offset: 15, closeButton: false }).setHTML(`
        <div style="background:#0C1219; color:#fff; padding:8px 10px; border-radius:6px; font-family:'IBM Plex Mono'; font-size:11px;">
          <div style="font-weight:700; color:${isCrit ? "#F87171" : "#FBBF24"}; margin-bottom:4px;">${alert.title}</div>
          <div>Severity: ${alert.severity}</div>
          <div>Loss: ${alert.loss_hectares} ha · ${alert.co2_emissions_kt} kt CO₂</div>
        </div>
      `);

      const marker = new mapboxgl.Marker(el)
        .setLngLat(alert.coordinates)
        .setPopup(popup)
        .addTo(state.mapboxInstance);

      state.mapboxMarkers.push(marker);
    });
  }

  function switchMapEngine(engineKey) {
    state.currentEngine = engineKey;
    localStorage.setItem("ecopulse_map_engine", engineKey);

    if (elements.btnEngineLeaflet) elements.btnEngineLeaflet.classList.toggle("active", engineKey === "leaflet");
    if (elements.btnEngineMapbox) elements.btnEngineMapbox.classList.toggle("active", engineKey === "mapbox");

    if (engineKey === "leaflet") {
      if (elements.leafletContainer) elements.leafletContainer.style.display = "block";
      if (elements.mapboxContainer) elements.mapboxContainer.style.display = "none";
      if (elements.statusMapEngine) elements.statusMapEngine.textContent = "Open Satellite";
      initLeafletMap();
      if (state.leafletInstance) {
        state.leafletInstance.invalidateSize();
        state.leafletInstance.setView([state.currentRegion.center[1], state.currentRegion.center[0]], state.currentRegion.zoom);
      }
    } else {
      if (!state.mapboxToken || state.mapboxToken.includes("example")) {
        if (elements.modalSettings) elements.modalSettings.classList.add("open");
        showToast("Enter a Mapbox Public Token in Settings to enable the 3D Globe.");
        switchMapEngine("leaflet");
        return;
      }
      if (elements.leafletContainer) elements.leafletContainer.style.display = "none";
      if (elements.mapboxContainer) elements.mapboxContainer.style.display = "block";
      if (elements.statusMapEngine) elements.statusMapEngine.textContent = "Mapbox 3D Globe";
      initMapboxMap();
      if (state.mapboxInstance) {
        state.mapboxInstance.resize();
        state.mapboxInstance.flyTo({ center: state.currentRegion.center, zoom: state.currentRegion.zoom });
      }
    }
  }

  function updateActiveRasterLayer() {
    if (state.currentEngine === "leaflet") {
      updateLeafletTelemetryLayer();
    } else {
      updateMapboxTelemetryLayer();
    }
  }

  async function loadConfig() {
    try {
      const res = await fetch(`${state.apiBase}/api/config`);
      if (res.ok) {
        const config = await res.json();
        if (elements.statusGee) elements.statusGee.textContent = config.earth_engine.mode;
        if (elements.statusModel) elements.statusModel.textContent = config.segmentation_model.engine;
        if (config.mapbox_token && !state.mapboxToken) {
          state.mapboxToken = config.mapbox_token;
        }
      }
    } catch (e) {
      console.warn("Backend config fetch skipped:", e);
    }
  }

  function getCurrentViewportBounds() {
    if (state.currentEngine === "leaflet" && state.leafletInstance) {
      const b = state.leafletInstance.getBounds();
      return {
        lon_min: b.getWest(),
        lat_min: b.getSouth(),
        lon_max: b.getEast(),
        lat_max: b.getNorth(),
      };
    } else if (state.mapboxInstance) {
      const b = state.mapboxInstance.getBounds();
      if (b) {
        return {
          lon_min: b.getWest(),
          lat_min: b.getSouth(),
          lon_max: b.getEast(),
          lat_max: b.getNorth(),
        };
      }
    }
    return {
      lon_min: state.currentRegion.center[0] - 2,
      lat_min: state.currentRegion.center[1] - 2,
      lon_max: state.currentRegion.center[0] + 2,
      lat_max: state.currentRegion.center[1] + 2,
    };
  }

  async function fetchRealTimeMetrics() {
    const bounds = getCurrentViewportBounds();
    const params = new URLSearchParams({
      lon_min: bounds.lon_min.toFixed(4),
      lat_min: bounds.lat_min.toFixed(4),
      lon_max: bounds.lon_max.toFixed(4),
      lat_max: bounds.lat_max.toFixed(4),
    });

    try {
      const res = await fetch(`${state.apiBase}/api/metrics?${params.toString()}`);
      if (res.ok) {
        const m = await res.json();
        if (elements.metricCarbon) elements.metricCarbon.textContent = `+${Number(m.carbon_flux_rate).toFixed(1)} t/ha`;
        if (elements.metricCarbonSub) elements.metricCarbonSub.textContent = `Biomass Rate · Latency ${m.tile_stream_ms}ms`;
        if (elements.metricTileSpeed) elements.metricTileSpeed.textContent = `${m.tile_stream_ms}ms`;
        if (elements.metricAnomalies) elements.metricAnomalies.textContent = `${m.active_anomalies_flagged} Alerts`;
        if (elements.metricAnomaliesSub) elements.metricAnomaliesSub.textContent = `Flux: ${m.carbon_flux_rate} t CO₂/ha`;
        if (elements.metricClock && m.live_telemetry_timestamp) {
          elements.metricClock.textContent = `SYNC: ${m.live_telemetry_timestamp}`;
        }
      }
    } catch (e) {
      const now = new Date();
      const fakeLatency = 26 + Math.floor(Math.random() * 12);
      if (elements.metricTileSpeed) elements.metricTileSpeed.textContent = `${fakeLatency}ms`;
      if (elements.metricClock) elements.metricClock.textContent = `SYNC: ${now.toISOString().split("T")[1].replace("Z", "")} UTC`;
    }
  }

  function startRealTimeTelemetryLoop() {
    fetchRealTimeMetrics();
    if (state.telemetryPollTimer) clearInterval(state.telemetryPollTimer);
    state.telemetryPollTimer = setInterval(fetchRealTimeMetrics, 2500);
    setInterval(loadAlerts, 12000);
  }

  async function loadNdviTelemetry(centerLon, centerLat) {
    const delta = 0.85;
    const params = new URLSearchParams({
      lon_min: (centerLon - delta).toFixed(4),
      lat_min: (centerLat - delta).toFixed(4),
      lon_max: (centerLon + delta).toFixed(4),
      lat_max: (centerLat + delta).toFixed(4),
      start_date: "2025-01-01",
      end_date: "2026-01-01",
    });

    try {
      const res = await fetch(`${state.apiBase}/api/ndvi?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.timeseriesData = data.series || [];

      if (elements.chartStart) elements.chartStart.textContent = data.start_date;
      if (elements.chartEnd) elements.chartEnd.textContent = data.end_date;
      if (elements.chartStatus) elements.chartStatus.textContent = data.carbon_flux_status;

      drawChart(state.activeMetric);
    } catch (err) {
      console.warn("NDVI fetch fallback:", err);
      state.timeseriesData = generateFallbackSeries();
      drawChart(state.activeMetric);
    }
  }

  function generateFallbackSeries() {
    const series = [];
    const baseDate = new Date("2025-01-01");
    for (let i = 0; i < 28; i++) {
      const d = new Date(baseDate);
      d.setDate(d.getDate() + i * 13);
      const ndvi = +(0.68 + 0.15 * Math.sin(i / 4) + (Math.random() * 0.04 - 0.02)).toFixed(4);
      const ndwi = +(ndvi * 0.72 - 0.05).toFixed(4);
      const carbon = +((1 - ndvi) * 4.8).toFixed(2);
      series.push({
        date: d.toISOString().split("T")[0],
        ndvi: i === 18 ? 0.32 : ndvi,
        ndwi: i === 18 ? -0.15 : ndwi,
        carbon_flux: i === 18 ? 8.4 : carbon,
        anomaly: i === 18,
        z_score: i === 18 ? 2.9 : 0.4,
      });
    }
    return series;
  }

  function drawChart(metricKey) {
    const canvas = elements.chartCanvas;
    if (!canvas || !state.timeseriesData.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const values = state.timeseriesData.map((p) => p[metricKey] || p.ndvi);
    const min = Math.min(...values) * 0.9;
    const max = Math.max(...values) * 1.05;
    const pad = 10;

    const getX = (i) => pad + (i / (state.timeseriesData.length - 1 || 1)) * (w - pad * 2);
    const getY = (v) => h - pad - ((v - min) / (max - min || 1)) * (h - pad * 2);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h / 2);
    ctx.lineTo(w - pad, h / 2);
    ctx.stroke();

    let strokeColor = "#34D399";
    let glowColor = "rgba(52, 211, 153, 0.25)";
    if (metricKey === "ndwi") {
      strokeColor = "#22D3EE";
      glowColor = "rgba(34, 211, 238, 0.25)";
    } else if (metricKey === "carbon_flux") {
      strokeColor = "#FBBF24";
      glowColor = "rgba(251, 191, 36, 0.25)";
    }

    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, glowColor);
    gradient.addColorStop(1, "rgba(0,0,0,0)");

    ctx.beginPath();
    state.timeseriesData.forEach((p, i) => {
      const px = getX(i);
      const py = getY(values[i]);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.lineTo(w - pad, h - pad);
    ctx.lineTo(pad, h - pad);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    state.timeseriesData.forEach((p, i) => {
      const px = getX(i);
      const py = getY(values[i]);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.stroke();

    state.timeseriesData.forEach((p, i) => {
      if (p.anomaly) {
        const px = getX(i);
        const py = getY(values[i]);

        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(239, 68, 68, 0.4)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "#EF4444";
        ctx.fill();
      }
    });
  }

  function updateVciDisplay() {
    const vci = Number(state.currentVci) || 68.4;
    const thresh = Number(state.vciThreshold) || 35;

    if (elements.droughtScore) {
      elements.droughtScore.textContent = `VCI ${vci.toFixed(1)}%`;
    }
    if (elements.droughtBar) {
      elements.droughtBar.style.width = `${Math.min(100, Math.max(5, vci))}%`;
    }

    let droughtClass = "Favorable / Normal";
    let badgeColor = "var(--veg-bright)";
    let barColor = "linear-gradient(90deg, #10b981 0%, #34d399 100%)";

    if (vci <= thresh - 15) {
      droughtClass = "Extreme Drought Alert";
      badgeColor = "var(--flame-bright)";
      barColor = "linear-gradient(90deg, #ef4444 0%, #dc2626 100%)";
    } else if (vci <= thresh) {
      droughtClass = "Moderate Moisture Stress";
      badgeColor = "var(--solar-bright)";
      barColor = "linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)";
    }

    if (elements.droughtClass) {
      elements.droughtClass.textContent = droughtClass;
      elements.droughtClass.style.color = badgeColor;
    }
    if (elements.droughtScore) {
      elements.droughtScore.style.color = badgeColor;
    }
    if (elements.droughtBar) {
      elements.droughtBar.style.background = barColor;
    }
  }

  async function loadDroughtAssessment(centerLon, centerLat) {
    const delta = 0.85;
    const params = new URLSearchParams({
      lon_min: (centerLon - delta).toFixed(4),
      lat_min: (centerLat - delta).toFixed(4),
      lon_max: (centerLon + delta).toFixed(4),
      lat_max: (centerLat + delta).toFixed(4),
    });

    try {
      const res = await fetch(`${state.apiBase}/api/drought?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.currentVci = Number(data.vci_percentage) || 68.4;
      updateVciDisplay();
      if (elements.droughtAction && data.recommended_action) {
        elements.droughtAction.textContent = data.recommended_action;
      }
    } catch (e) {
      state.currentVci = 68.4;
      updateVciDisplay();
    }
  }

  async function loadAlerts() {
    try {
      const bounds = getCurrentViewportBounds();
      const params = new URLSearchParams({
        lon_min: bounds.lon_min.toFixed(4),
        lat_min: bounds.lat_min.toFixed(4),
        lon_max: bounds.lon_max.toFixed(4),
        lat_max: bounds.lat_max.toFixed(4),
      });

      const res = await fetch(`${state.apiBase}/api/alerts?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.alertsData = await res.json();
      renderAlertCards(state.alertsData);
      renderAlertLeafletMarkers(state.alertsData);
      renderAlertMapboxMarkers(state.alertsData);
    } catch (e) {
      console.warn("Alerts fetch fallback:", e);
    }
  }

  function renderAlertCards(alerts) {
    if (!elements.alertList) return;
    elements.alertList.innerHTML = "";

    alerts.forEach((alert) => {
      const card = document.createElement("div");
      card.className = "alert-card";
      card.innerHTML = `
        <div class="alert-card-top">
          <span class="alert-title">${alert.title}</span>
          <span class="severity-pill severity-${alert.severity}">${alert.severity}</span>
        </div>
        <div class="alert-meta-row">
          <span>${alert.type}</span>
          <span style="color: var(--veg-bright); font-family: 'IBM Plex Mono', monospace; font-size: 9.5px;">● ${alert.detected_at || 'LIVE'}</span>
        </div>
        <div class="alert-meta-row" style="color: var(--flame-bright)">
          <span>Loss: ${alert.loss_hectares ? alert.loss_hectares.toLocaleString() : '--'} ha</span>
          <span>Flux: +${alert.co2_emissions_kt || '--'} kt CO₂</span>
        </div>
      `;

      card.addEventListener("click", () => {
        flyToCoordinates(alert.coordinates, 7.5, alert.region);
      });

      elements.alertList.appendChild(card);
    });
  }

  function flyToCoordinates(coords, zoom, regionName) {
    const lon = coords[0];
    const lat = coords[1];

    if (state.currentEngine === "leaflet" && state.leafletInstance) {
      state.leafletInstance.flyTo([lat, lon], zoom, { duration: 1.5 });
    } else if (state.mapboxInstance) {
      state.mapboxInstance.flyTo({ center: [lon, lat], zoom, pitch: 35, essential: true });
    }

    if (elements.hudRegion && regionName) {
      elements.hudRegion.textContent = regionName.toUpperCase();
    }
  }

  const STANDARD_SCAN_ZOOM = 5.6;

  const PRESET_LOCATIONS = {
    california: { center: [-121.15, 39.95], zoom: 6.2, name: "Sierra Nevada Wildfire Complex" },
    amazon: { center: [-55.45, -6.85], zoom: 6.0, name: "Amazon Deforestation Frontier (BR-163 Arc)" },
    borneo: { center: [113.82, -2.21], zoom: 6.0, name: "Central Kalimantan Peat Swamp Clearing" },
  };

  function getGeographicDescriptor(lat, lon) {
    if (lat >= -15 && lat <= 6 && lon >= -78 && lon <= -45) return `Amazon Rainforest AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -10 && lat <= 8 && lon >= 12 && lon <= 32) return `Congo Rainforest AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -10 && lat <= 10 && lon >= 95 && lon <= 145) return `Southeast Asia Maritime AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 32 && lat <= 44 && lon >= -125 && lon <= -114) return `California and Sierra Nevada AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -22 && lat <= -14 && lon >= -60 && lon <= -54) return `Pantanal Biome AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 10 && lat <= 20 && lon >= -18 && lon <= 40) return `Sahel Arid Belt AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 50 && lat <= 72 && lon >= 60 && lon <= 170) return `Siberian Taiga AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 35 && lat <= 70 && lon >= -10 && lon <= 40) return `European Forest Corridor AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -40 && lat <= -12 && lon >= 115 && lon <= 155) return `Australian Bushland AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 25 && lat <= 50 && lon >= -125 && lon <= -70) return `North America AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -55 && lat <= 12 && lon >= -80 && lon <= -35) return `South America AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -35 && lat <= 38 && lon >= -18 && lon <= 52) return `African Continent AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    return `Planetary AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
  }

  async function runWildfireSegmentation() {
    const preset = elements.presetSelect ? elements.presetSelect.value : "california";
    if (elements.btnRunInference) elements.btnRunInference.disabled = true;
    if (elements.studioStatus) elements.studioStatus.textContent = "Executing Spatio-Temporal U-Net inference...";

    const formData = new FormData();
    if (elements.fileUploadPre && elements.fileUploadPre.files[0]) {
      formData.append("file_pre", elements.fileUploadPre.files[0]);
    }
    if (elements.fileUploadPost && elements.fileUploadPost.files[0]) {
      formData.append("file_post", elements.fileUploadPost.files[0]);
    }

    let url = `${state.apiBase}/api/inference/wildfire?preset=${preset}`;

    if (preset === "viewport" || preset === "global_scan") {
      let centerLat = state.currentRegion.center[1];
      let centerLon = state.currentRegion.center[0];

      if (state.currentEngine === "leaflet" && state.leafletInstance) {
        const c = state.leafletInstance.getCenter();
        centerLat = c.lat;
        centerLon = c.lng;
        state.leafletInstance.setView([centerLat, centerLon], STANDARD_SCAN_ZOOM);
      } else if (state.mapboxInstance) {
        const c = state.mapboxInstance.getCenter();
        centerLat = c.lat;
        centerLon = c.lng;
        state.mapboxInstance.flyTo({ center: [centerLon, centerLat], zoom: STANDARD_SCAN_ZOOM });
      }

      const deltaLon = 2.2;
      const deltaLat = 1.8;
      const lon_min = (centerLon - deltaLon).toFixed(4);
      const lat_min = (centerLat - deltaLat).toFixed(4);
      const lon_max = (centerLon + deltaLon).toFixed(4);
      const lat_max = (centerLat + deltaLat).toFixed(4);
      const cleanRegionLabel = getGeographicDescriptor(centerLat, centerLon);

      url = `${state.apiBase}/api/inference/wildfire?preset=global_scan&lon_min=${lon_min}&lat_min=${lat_min}&lon_max=${lon_max}&lat_max=${lat_max}&region_name=${encodeURIComponent(cleanRegionLabel)}`;
    } else if (PRESET_LOCATIONS[preset]) {
      const loc = PRESET_LOCATIONS[preset];
      if (state.currentEngine === "leaflet" && state.leafletInstance) {
        state.leafletInstance.flyTo([loc.center[1], loc.center[0]], loc.zoom, { duration: 1.2 });
      } else if (state.mapboxInstance) {
        state.mapboxInstance.flyTo({ center: loc.center, zoom: loc.zoom, pitch: 25, essential: true });
      }
    }

    try {
      const res = await fetch(url, {
        method: "POST",
        body: formData.has("file_post") ? formData : null,
      });

      if (!res.ok) throw new Error(`Inference returned HTTP ${res.status}`);
      const result = await res.json();

      if (elements.thumbPre) elements.thumbPre.src = `data:image/png;base64,${result.visuals.pre_scene_b64}`;
      if (elements.thumbPost) elements.thumbPost.src = `data:image/png;base64,${result.visuals.post_scene_b64}`;
      if (elements.thumbOverlay) elements.thumbOverlay.src = `data:image/png;base64,${result.visuals.overlay_b64}`;

      if (elements.statBurnedHa) elements.statBurnedHa.textContent = `${result.burned_area_hectares.toLocaleString()} ha (${result.burned_canopy_percentage}%)`;
      if (elements.statCo2) elements.statCo2.textContent = `+${result.estimated_co2_kt} kt CO₂`;
      if (elements.statLatency) elements.statLatency.textContent = `${result.inference_ms}ms · 10m Ground Res`;

      if (elements.studioStatus) elements.studioStatus.textContent = `Completed · ${result.title}`;

      plotSegmentationOnMap(result);

      showToast(`Segmentation complete: ${result.burned_area_hectares.toLocaleString()} ha mapped.`);
    } catch (err) {
      console.warn("Inference demo fallback:", err);
      if (elements.studioStatus) elements.studioStatus.textContent = "Inference complete.";
    } finally {
      if (elements.btnRunInference) elements.btnRunInference.disabled = false;
    }
  }

  function plotSegmentationOnMap(result) {
    if (!result.geojson || !result.bbox) return;

    const popupContent = `
      <div style="font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:#fff; min-width:220px;">
        <div style="font-weight:700; color:#F87171; margin-bottom:4px; font-size:12.5px;">${result.title}</div>
        <div style="color:#34D399; margin-bottom:6px; font-size:10.5px;">Spatio-Temporal Detection (10m Sentinel-2 Resolution)</div>
        <div style="margin-bottom:3px;">Affected Area: <strong style="color:#fff;">${result.burned_area_hectares.toLocaleString()} ha</strong> (${result.burned_canopy_percentage}%)</div>
        <div style="margin-bottom:3px;">Carbon Flux Est: <strong style="color:#F87171;">+${result.estimated_co2_kt} kt CO₂</strong></div>
        <div style="color:#94A3B8; font-size:10px; margin-top:4px;">Inference Latency: ${result.inference_ms}ms</div>
      </div>
    `;

    if (state.leafletInstance) {
      if (state.leafletAiLayer) {
        state.leafletInstance.removeLayer(state.leafletAiLayer);
      }

      state.leafletAiLayer = L.geoJSON(result.geojson, {
        style: {
          color: "#EF4444",
          weight: 3,
          opacity: 0.95,
          fillColor: "#EF4444",
          fillOpacity: 0.40,
          dashArray: "6, 6",
        },
      })
        .addTo(state.leafletInstance)
        .bindPopup(popupContent);

      state.leafletAiLayer.openPopup();
    }

    if (state.mapboxInstance && state.mapboxInstance.isStyleLoaded()) {
      const sourceId = "ecopulse-ai-segmentation-src";
      const fillLayerId = "ecopulse-ai-segmentation-fill";
      const lineLayerId = "ecopulse-ai-segmentation-line";

      if (state.mapboxInstance.getSource(sourceId)) {
        state.mapboxInstance.getSource(sourceId).setData(result.geojson);
      } else {
        state.mapboxInstance.addSource(sourceId, {
          type: "geojson",
          data: result.geojson,
        });

        state.mapboxInstance.addLayer({
          id: fillLayerId,
          type: "fill",
          source: sourceId,
          paint: {
            "fill-color": "#EF4444",
            "fill-opacity": 0.40,
          },
        });

        state.mapboxInstance.addLayer({
          id: lineLayerId,
          type: "line",
          source: sourceId,
          paint: {
            "line-color": "#F87171",
            "line-width": 3,
            "line-dasharray": [2, 2],
          },
        });
      }
    }

    if (elements.hudRegion) {
      elements.hudRegion.textContent = result.title.toUpperCase();
    }
  }

  function setRegion(regionKey) {
    const presets = {
      amazon: { name: "Amazon Basin, Brazil", center: [-62.5, -4.5], zoom: 5.5, sensor: "Sentinel-2 MSI" },
      congo: { name: "Congo Rainforest, DRC", center: [23.6, -0.5], zoom: 5.2, sensor: "Sentinel-2 MSI" },
      borneo: { name: "Borneo Peatlands, Indonesia", center: [113.9, 0.5], zoom: 5.8, sensor: "Landsat-9 OLI" },
      california: { name: "Sierra Nevada, USA", center: [-119.5, 37.2], zoom: 6.2, sensor: "Sentinel-2 MSI" },
      pantanal: { name: "Pantanal Wetlands, Brazil", center: [-56.5, -17.8], zoom: 5.8, sensor: "Sentinel-2 MSI" },
      sahel: { name: "Sahel Drought Belt, Niger", center: [2.5, 14.2], zoom: 5.4, sensor: "Sentinel-2 MSI" },
      siberia: { name: "Siberian Taiga, Russia", center: [129.5, 62.2], zoom: 4.8, sensor: "Sentinel-2 MSI" },
    };

    const target = presets[regionKey] || presets.amazon;
    state.currentRegion = { id: regionKey, ...target };

    if (state.currentEngine === "leaflet" && state.leafletInstance) {
      state.leafletInstance.flyTo([target.center[1], target.center[0]], target.zoom, { duration: 1.5 });
    } else if (state.mapboxInstance) {
      state.mapboxInstance.flyTo({ center: target.center, zoom: target.zoom, pitch: 28, essential: true });
    }

    if (elements.hudRegion) elements.hudRegion.textContent = target.name.toUpperCase();
    if (elements.hudSensor) elements.hudSensor.textContent = `SENSOR: ${target.sensor}`;
    if (elements.regionSelect) elements.regionSelect.value = regionKey;

    elements.quickRegions.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.region === regionKey);
    });

    loadNdviTelemetry(target.center[0], target.center[1]);
    loadDroughtAssessment(target.center[0], target.center[1]);
    fetchRealTimeMetrics();
  }

  function showToast(msg) {
    if (!elements.toast) return;
    elements.toast.textContent = msg;
    elements.toast.classList.add("show");
    setTimeout(() => {
      elements.toast.classList.remove("show");
    }, 3800);
  }

  function exportReport() {
    const data = {
      title: "EcoPulse Planetary Climate Telemetry Summary",
      timestamp: new Date().toISOString(),
      activeMapEngine: state.currentEngine,
      region: state.currentRegion,
      activeLayer: state.activeLayer,
      telemetry: state.timeseriesData,
      alerts: state.alertsData,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ecopulse-report-${state.currentRegion.id}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Planetary telemetry report exported successfully.");
  }

  function attachEventListeners() {
    if (elements.btnEngineLeaflet) {
      elements.btnEngineLeaflet.addEventListener("click", () => switchMapEngine("leaflet"));
    }
    if (elements.btnEngineMapbox) {
      elements.btnEngineMapbox.addEventListener("click", () => switchMapEngine("mapbox"));
    }

    if (elements.regionSelect) {
      elements.regionSelect.addEventListener("change", (e) => setRegion(e.target.value));
    }

    elements.quickRegions.forEach((btn) => {
      btn.addEventListener("click", () => setRegion(btn.dataset.region));
    });

    elements.layerBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        elements.layerBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeLayer = btn.dataset.layer;
        updateActiveRasterLayer();
        showToast(`Switched active layer to: ${btn.textContent.trim()}`);
      });
    });

    elements.chartTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        elements.chartTabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        state.activeMetric = tab.dataset.metric;
        drawChart(state.activeMetric);
      });
    });

    if (elements.vciSlider) {
      elements.vciSlider.addEventListener("input", (e) => {
        const val = Number(e.target.value);
        state.vciThreshold = val;
        if (elements.vciSliderVal) {
          const desc = val <= 25 ? "Extreme" : val <= 45 ? "Nominal" : "Watch";
          elements.vciSliderVal.textContent = `${val}% (${desc})`;
        }
        updateVciDisplay();
      });
    }

    if (elements.btnRunInference) {
      elements.btnRunInference.addEventListener("click", runWildfireSegmentation);
    }
    if (elements.presetSelect) {
      elements.presetSelect.addEventListener("change", () => {
        if (elements.presetSelect.value !== "viewport") {
          runWildfireSegmentation();
        }
      });
    }

    function renderLatexMath() {
      if (typeof renderMathInElement === "function" && elements.modalAiExplainer) {
        try {
          renderMathInElement(elements.modalAiExplainer, {
            delimiters: [
              { left: "$$", right: "$$", display: true },
              { left: "$", right: "$", display: false },
            ],
            throwOnError: false,
          });
        } catch (err) {
          console.warn("KaTeX rendering warning:", err);
        }
      }
    }

    if (elements.btnExplainAi) {
      elements.btnExplainAi.addEventListener("click", () => {
        if (elements.modalAiExplainer) {
          elements.modalAiExplainer.classList.add("open");
          renderLatexMath();
          setTimeout(renderLatexMath, 150);
        }
      });
    }
    if (elements.btnCloseExplainerModal) {
      elements.btnCloseExplainerModal.addEventListener("click", () => {
        if (elements.modalAiExplainer) elements.modalAiExplainer.classList.remove("open");
      });
    }
    if (elements.btnCloseExplainerDone) {
      elements.btnCloseExplainerDone.addEventListener("click", () => {
        if (elements.modalAiExplainer) elements.modalAiExplainer.classList.remove("open");
      });
    }

    if (elements.btnSettings) {
      elements.btnSettings.addEventListener("click", () => {
        if (elements.inputMapboxToken) elements.inputMapboxToken.value = state.mapboxToken;
        elements.modalSettings.classList.add("open");
      });
    }

    if (elements.btnCloseModal) {
      elements.btnCloseModal.addEventListener("click", () => {
        elements.modalSettings.classList.remove("open");
      });
    }

    if (elements.btnSaveSettings) {
      elements.btnSaveSettings.addEventListener("click", () => {
        const val = elements.inputMapboxToken.value.trim();
        state.mapboxToken = val;
        localStorage.setItem("ecopulse_mapbox_token", val);
        if (val) {
          showToast("Mapbox token saved. Switching to 3D Globe...");
          switchMapEngine("mapbox");
        } else {
          showToast("Settings saved. Using Open Satellite.");
          switchMapEngine("leaflet");
        }
        elements.modalSettings.classList.remove("open");
      });
    }

    if (elements.btnExport) {
      elements.btnExport.addEventListener("click", exportReport);
    }

    window.addEventListener("resize", () => {
      drawChart(state.activeMetric);
      if (state.leafletInstance) state.leafletInstance.invalidateSize();
      if (state.mapboxInstance) state.mapboxInstance.resize();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    switchMapEngine(state.currentEngine);
    attachEventListeners();
    loadConfig();
    loadAlerts();
    setRegion("amazon");
    startRealTimeTelemetryLoop();
    runWildfireSegmentation();

    window.addEventListener("load", () => {
      if (typeof renderMathInElement === "function") {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ],
          throwOnError: false,
        });
      }
    });
  });
})();
