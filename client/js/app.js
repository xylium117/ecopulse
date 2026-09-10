(function () {
  "use strict";

  const rawSavedEngine = localStorage.getItem("ecopulse_map_engine") || "leaflet";
  const rawToken = window.MAPBOX_TOKEN || localStorage.getItem("ecopulse_mapbox_token") || "";
  const safeEngine = (rawSavedEngine === "mapbox" && rawToken.length > 20 && !rawToken.includes("example")) ? "mapbox" : "leaflet";

  const state = {
    apiBase: window.ECOPULSE_API_BASE || localStorage.getItem("ecopulse_api_base") || (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" ? "http://localhost:8000" : "https://ecopulse-1-uzog.onrender.com"),
    mapboxToken: rawToken,
    currentEngine: safeEngine,
    activeHazard: "wildfire", // "wildfire" | "flood"
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
    currentVci: 68.4,
    vciThreshold: 35,
    currentFfsi: 84.2,
    floodThreshold: 60,
    isZoomLocked: false,
  };

  const elements = {
    tabHazardWildfire: document.getElementById("tab-hazard-wildfire"),
    tabHazardFlood: document.getElementById("tab-hazard-flood"),

    metricsPanelTitle: document.getElementById("metrics-panel-title"),
    metricPrimaryLabel: document.getElementById("metric-primary-label"),
    metricCarbon: document.getElementById("metric-carbon"),
    metricCarbonSub: document.getElementById("metric-carbon-sub"),
    metricTileSpeed: document.getElementById("metric-tile-speed"),
    metricStreamSub: document.getElementById("metric-stream-sub"),
    metricAnomalies: document.getElementById("metric-anomalies"),
    metricAnomaliesSub: document.getElementById("metric-anomalies-sub"),
    metricClock: document.getElementById("metric-clock"),

    // Separated Raster Layer Containers
    layerPillsWildfire: document.getElementById("layer-pills-wildfire"),
    layerPillsFlood: document.getElementById("layer-pills-flood"),
    layerBtns: document.querySelectorAll(".layer-btn"),

    // Separated Chart Tab Containers
    chartTabsWildfire: document.getElementById("chart-tabs-wildfire"),
    chartTabsFlood: document.getElementById("chart-tabs-flood"),
    chartTabs: document.querySelectorAll(".chart-tab"),

    chartCanvas: document.getElementById("ndvi-chart-canvas"),
    chartWrap: document.getElementById("ndvi-chart-wrap"),
    chartStart: document.getElementById("chart-start"),
    chartEnd: document.getElementById("chart-end"),
    chartStatus: document.getElementById("chart-status"),

    // Hazard-Specific Panels
    panelDroughtSection: document.getElementById("panel-drought-section"),
    droughtScore: document.getElementById("drought-score"),
    droughtClass: document.getElementById("drought-class"),
    droughtBar: document.getElementById("drought-bar"),
    vciSlider: document.getElementById("vci-slider"),
    vciSliderVal: document.getElementById("vci-slider-val"),

    panelFloodSection: document.getElementById("panel-flood-section"),
    floodRiskScore: document.getElementById("flood-risk-score"),
    floodSaturationBar: document.getElementById("flood-saturation-bar"),
    floodRiskClass: document.getElementById("flood-risk-class"),
    floodStatSaturation: document.getElementById("flood-stat-saturation"),
    floodStatPrecip: document.getElementById("flood-stat-precip"),
    floodStatDeforestation: document.getElementById("flood-stat-deforestation"),
    floodStatCn: document.getElementById("flood-stat-cn"),
    floodStatTwi: document.getElementById("flood-stat-twi"),
    floodSlider: document.getElementById("flood-slider"),
    floodSliderVal: document.getElementById("flood-slider-val"),
    floodActionText: document.getElementById("flood-action-text"),

    alertList: document.getElementById("alert-feed-list"),

    studioPanelTitle: document.getElementById("studio-panel-title"),
    presetSelect: document.getElementById("studio-preset"),
    btnRunInference: document.getElementById("btn-run-inference"),
    btnInferenceLabel: document.getElementById("btn-inference-label"),
    studioStatus: document.getElementById("studio-status"),
    thumbPre: document.getElementById("thumb-pre"),
    thumbPost: document.getElementById("thumb-post"),
    thumbOverlay: document.getElementById("thumb-overlay"),
    thumbTagOverlay: document.getElementById("thumb-tag-overlay"),
    statPrimaryLabel: document.getElementById("stat-primary-label"),
    statSecondaryLabel: document.getElementById("stat-secondary-label"),
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

    // Separated Quick Regions Containers
    quickRegionsWildfire: document.getElementById("quick-regions-wildfire"),
    quickRegionsFlood: document.getElementById("quick-regions-flood"),
    quickRegions: document.querySelectorAll(".quick-region-btn"),

    // Separated Legend Containers
    legendItemsWildfire: document.getElementById("legend-items-wildfire"),
    legendItemsFlood: document.getElementById("legend-items-flood"),

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
    statusFloodModel: document.getElementById("status-flood-model"),

    btnExport: document.getElementById("btn-export"),
    toast: document.getElementById("toast"),
    btnZoomIn: document.getElementById("btn-zoom-in"),
    btnZoomOut: document.getElementById("btn-zoom-out"),
    btnLockZoom: document.getElementById("btn-lock-zoom"),
    iconLockZoom: document.getElementById("icon-lock-zoom"),
    labelLockZoom: document.getElementById("label-lock-zoom"),
    btnRecenter: document.getElementById("btn-recenter"),
    btnScanViewport: document.getElementById("btn-scan-viewport"),
    btnFullscreen: document.getElementById("btn-fullscreen"),
    iconFullscreen: document.getElementById("icon-fullscreen"),
    btnToggleDashboard: document.getElementById("btn-toggle-dashboard"),
    statSeverityBadge: document.getElementById("stat-severity-badge"),
    sidebar: document.getElementById("sidebar"),
    sidebarToggleTab: document.getElementById("sidebar-toggle-tab"),
    btnForceLandscape: document.getElementById("btn-force-landscape"),
  };

  const WILDFIRE_PRESET_OPTIONS = `
    <option value="california" selected>Preset: Sierra Nevada Fire Scar (California)</option>
    <option value="amazon">Preset: Amazon Clear-Cut Logging Arc</option>
    <option value="borneo">Preset: Borneo Peatland Fire &amp; Drainage</option>
    <option value="viewport">Scan Current Map Viewport (Global Model)</option>
  `;

  const FLOOD_PRESET_OPTIONS = `
    <option value="nepal" selected>Preset: Nepal &amp; Tibet (Bagmati &amp; Koshi Surge)</option>
    <option value="india">Preset: India (Ganges &amp; Brahmaputra Corridor)</option>
    <option value="valencia">Preset: Valencia DANA Flash Flood &amp; Ravine Surge (Spain)</option>
    <option value="bangladesh">Preset: Bangladesh Padma &amp; Meghna Delta Inundation</option>
    <option value="viewport">Scan Current Map Viewport (Global Flood Model)</option>
  `;

  const PRESET_LOCATIONS = {
    california: { center: [-121.15, 39.95], zoom: 6.2, name: "Sierra Nevada Wildfire Complex" },
    amazon: { center: [-55.45, -6.85], zoom: 6.0, name: "Amazon Deforestation Frontier (BR-163 Arc)" },
    borneo: { center: [113.82, -2.21], zoom: 6.0, name: "Central Kalimantan Peat Swamp Clearing" },
    nepal: { center: [85.65, 27.22], zoom: 6.8, name: "Nepal & Tibet Flash Flood Corridor" },
    india: { center: [86.00, 26.15], zoom: 6.2, name: "India (Ganges & Brahmaputra Corridor)" },
    indo_gangetic: { center: [86.00, 26.15], zoom: 6.2, name: "India (Ganges & Brahmaputra Corridor)" },
    indian_subcontinent: { center: [86.00, 26.15], zoom: 6.2, name: "India (Ganges & Brahmaputra Corridor)" },
    valencia: { center: [-0.40, 39.40], zoom: 7.0, name: "Valencia DANA Flash Flood Basin (Spain)" },
    bangladesh: { center: [90.35, 23.65], zoom: 6.5, name: "Bangladesh Meghna & Padma River Basin" },
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
      maxZoom: 13,
      maxBounds: [
        [-85, -180],
        [85, 180],
      ],
      maxBoundsViscosity: 1.0,
      worldCopyJump: false,
      zoomControl: false,
      attributionControl: true,
    });

    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      minZoom: 3,
      maxZoom: 13,
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
      maxZoom: 13,
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
      onViewportChanged();
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
      const isFlood = alert.type.toLowerCase().includes("flood") || alert.type.toLowerCase().includes("inundation");
      const color = isFlood ? (isCrit ? "#F97316" : "#0284C7") : (isCrit ? "#EF4444" : "#F59E0B");

      const icon = L.divIcon({
        className: "custom-leaflet-marker",
        html: `<div style="width:14px;height:14px;border-radius:50%;background:${color};box-shadow:0 0 12px ${color};border:2px solid #fff;cursor:pointer;"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });

      const marker = L.marker([alert.coordinates[1], alert.coordinates[0]], { icon })
        .addTo(state.leafletInstance)
        .bindPopup(`
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#fff;min-width:200px;">
            <div style="font-weight:700;color:${isFlood ? "#FB923C" : (isCrit ? "#F87171" : "#FBBF24")};margin-bottom:4px;">${alert.title}</div>
            <div style="color:#94A3B8;">Type: <strong style="color:#fff;">${alert.type}</strong></div>
            <div style="color:#94A3B8;">Severity: <strong style="color:${color};">${alert.severity}</strong> · ${alert.confidence}</div>
            <div style="color:#94A3B8;margin-top:4px;">${alert.description}</div>
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
        maxZoom: 13,
        maxBounds: [
          [-180, -85],
          [180, 85],
        ],
        renderWorldCopies: false,
      });

      state.mapboxInstance.on("style.load", () => {
        state.mapboxInstance.setFog({
          color: "rgb(12, 18, 25)",
          "high-color": "rgb(36, 92, 223)",
          "horizon-blend": 0.08,
          "space-color": "rgb(6, 9, 14)",
          "star-intensity": 0.6,
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
        onViewportChanged();
      });
    } catch (e) {
      console.warn("Mapbox initialization error:", e);
      switchMapEngine("leaflet");
    }
  }

  function updateMapboxTelemetryLayer() {
    if (!state.mapboxInstance || !state.mapboxInstance.isStyleLoaded()) return;

    const sourceId = "ecopulse-raster-src";
    const layerId = "ecopulse-raster-layer";

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
      bounds: [-180, -85, 180, 85],
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
    if (!state.mapboxInstance || !alerts || !alerts.length) return;

    state.mapboxMarkers.forEach((m) => m.remove());
    state.mapboxMarkers = [];

    alerts.forEach((alert) => {
      const isCrit = alert.severity === "CRITICAL";
      const isFlood = alert.type.toLowerCase().includes("flood") || alert.type.toLowerCase().includes("inundation");
      const color = isFlood ? (isCrit ? "#F97316" : "#0284C7") : (isCrit ? "#EF4444" : "#F59E0B");

      const el = document.createElement("div");
      el.style.width = "14px";
      el.style.height = "14px";
      el.style.borderRadius = "50%";
      el.style.backgroundColor = color;
      el.style.boxShadow = `0 0 12px ${color}`;
      el.style.border = "2px solid #fff";
      el.style.cursor = "pointer";

      const popup = new mapboxgl.Popup({ offset: 12 }).setHTML(`
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#fff;background:#0c1219;padding:8px;border-radius:6px;min-width:200px;">
          <div style="font-weight:700;color:${isFlood ? "#FB923C" : (isCrit ? "#F87171" : "#FBBF24")};margin-bottom:4px;">${alert.title}</div>
          <div style="color:#94A3B8;">Type: <strong style="color:#fff;">${alert.type}</strong></div>
          <div style="color:#94A3B8;">Severity: <strong style="color:${color};">${alert.severity}</strong></div>
        </div>
      `);

      const marker = new mapboxgl.Marker(el)
        .setLngLat(alert.coordinates)
        .setPopup(popup)
        .addTo(state.mapboxInstance);

      state.mapboxMarkers.push(marker);
    });
  }

  function onViewportChanged() {
    let centerLon = state.currentRegion.center[0];
    let centerLat = state.currentRegion.center[1];

    if (state.currentEngine === "leaflet" && state.leafletInstance) {
      const c = state.leafletInstance.getCenter();
      centerLon = c.lng;
      centerLat = c.lat;
    } else if (state.mapboxInstance) {
      const c = state.mapboxInstance.getCenter();
      centerLon = c.lng;
      centerLat = c.lat;
    }

    fetchRealTimeMetrics();
    if (state.activeHazard === "flood") {
      loadFlashFloodRisk(centerLon, centerLat);
    } else {
      loadDroughtAssessment(centerLon, centerLat);
      loadNdviTelemetry(centerLon, centerLat);
    }
  }

  function handleZoomIn() {
    if (state.isZoomLocked) {
      showToast("Zoom is locked. Click the lock tool to unlock.");
      return;
    }
    if (state.currentEngine === "leaflet" && state.leafletInstance) {
      const curr = state.leafletInstance.getZoom();
      if (curr < 13) state.leafletInstance.setZoom(Math.min(13, curr + 1));
      else showToast("Maximum zoom level (13) reached for crisp satellite imagery.");
    } else if (state.mapboxInstance) {
      const curr = state.mapboxInstance.getZoom();
      if (curr < 13) state.mapboxInstance.setZoom(Math.min(13, curr + 1));
      else showToast("Maximum zoom level (13) reached for crisp satellite imagery.");
    }
  }

  function handleZoomOut() {
    if (state.isZoomLocked) {
      showToast("Zoom is locked. Click the lock tool to unlock.");
      return;
    }
    if (state.currentEngine === "leaflet" && state.leafletInstance) {
      const curr = state.leafletInstance.getZoom();
      if (curr > 3) state.leafletInstance.setZoom(Math.max(3, curr - 1));
    } else if (state.mapboxInstance) {
      const curr = state.mapboxInstance.getZoom();
      if (curr > 2.5) state.mapboxInstance.setZoom(Math.max(2.5, curr - 1));
    }
  }

  function handleRecenter() {
    setRegion(state.currentRegion.id || "amazon");
    showToast(`Recentered view on ${state.currentRegion.name}`);
  }

  function handleFullscreen() {
    const mapWrap = document.getElementById("map-wrap");
    if (!document.fullscreenElement) {
      if (mapWrap && mapWrap.requestFullscreen) {
        mapWrap.requestFullscreen();
      } else if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen();
      }
      if (elements.iconFullscreen) elements.iconFullscreen.className = "fa-solid fa-compress";
      showToast("Fullscreen map mode enabled.");
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
      if (elements.iconFullscreen) elements.iconFullscreen.className = "fa-solid fa-expand";
      showToast("Exited fullscreen mode.");
    }
  }

  function toggleZoomLock(forcedState) {
    state.isZoomLocked = typeof forcedState === "boolean" ? forcedState : !state.isZoomLocked;

    if (state.leafletInstance) {
      if (state.isZoomLocked) {
        state.leafletInstance.scrollWheelZoom.disable();
        state.leafletInstance.doubleClickZoom.disable();
        state.leafletInstance.touchZoom.disable();
        state.leafletInstance.boxZoom.disable();
        state.leafletInstance.keyboard.disable();
      } else {
        state.leafletInstance.scrollWheelZoom.enable();
        state.leafletInstance.doubleClickZoom.enable();
        state.leafletInstance.touchZoom.enable();
        state.leafletInstance.boxZoom.enable();
        state.leafletInstance.keyboard.enable();
      }
    }

    if (state.mapboxInstance) {
      if (state.isZoomLocked) {
        state.mapboxInstance.scrollZoom.disable();
        state.mapboxInstance.doubleClickZoom.disable();
        state.mapboxInstance.touchZoomRotate.disable();
      } else {
        state.mapboxInstance.scrollZoom.enable();
        state.mapboxInstance.doubleClickZoom.enable();
        state.mapboxInstance.touchZoomRotate.enable();
      }
    }

    if (elements.btnLockZoom) {
      if (state.isZoomLocked) {
        elements.btnLockZoom.classList.add("active");
        if (elements.iconLockZoom) elements.iconLockZoom.className = "fa-solid fa-lock";
        if (typeof forcedState !== "boolean") showToast("Map zoom locked.");
      } else {
        elements.btnLockZoom.classList.remove("active");
        if (elements.iconLockZoom) elements.iconLockZoom.className = "fa-solid fa-lock-open";
        if (typeof forcedState !== "boolean") showToast("Map zoom unlocked.");
      }
    }
  }

  function switchMapEngine(engineName) {
    state.currentEngine = engineName;
    localStorage.setItem("ecopulse_map_engine", engineName);

    if (engineName === "leaflet") {
      if (elements.leafletContainer) elements.leafletContainer.style.display = "block";
      if (elements.mapboxContainer) elements.mapboxContainer.style.display = "none";
      if (elements.btnEngineLeaflet) elements.btnEngineLeaflet.classList.add("active");
      if (elements.btnEngineMapbox) elements.btnEngineMapbox.classList.remove("active");
      if (elements.statusMapEngine) elements.statusMapEngine.textContent = "Open Satellite Engine (Active)";

      initLeafletMap();
      if (state.leafletInstance) {
        state.leafletInstance.invalidateSize();
        if (state.isZoomLocked) toggleZoomLock(true);
      }
    } else {
      if (elements.leafletContainer) elements.leafletContainer.style.display = "none";
      if (elements.mapboxContainer) elements.mapboxContainer.style.display = "block";
      if (elements.btnEngineLeaflet) elements.btnEngineLeaflet.classList.remove("active");
      if (elements.btnEngineMapbox) elements.btnEngineMapbox.classList.add("active");
      if (elements.statusMapEngine) elements.statusMapEngine.textContent = "Mapbox 3D Globe (Active)";

      initMapboxMap();
      if (state.mapboxInstance) {
        state.mapboxInstance.resize();
        if (state.isZoomLocked) toggleZoomLock(true);
      }
    }
  }

  function switchHazardMode(hazard) {
    state.activeHazard = hazard;

    if (elements.tabHazardWildfire) elements.tabHazardWildfire.classList.toggle("active", hazard === "wildfire");
    if (elements.tabHazardFlood) elements.tabHazardFlood.classList.toggle("active", hazard === "flood");

    if (hazard === "wildfire") {
      // Toggle layer pills
      if (elements.layerPillsWildfire) elements.layerPillsWildfire.style.display = "flex";
      if (elements.layerPillsFlood) elements.layerPillsFlood.style.display = "none";

      // Toggle chart tabs
      if (elements.chartTabsWildfire) elements.chartTabsWildfire.style.display = "flex";
      if (elements.chartTabsFlood) elements.chartTabsFlood.style.display = "none";

      // Toggle quick regions
      if (elements.quickRegionsWildfire) elements.quickRegionsWildfire.style.display = "flex";
      if (elements.quickRegionsFlood) elements.quickRegionsFlood.style.display = "none";

      // Toggle legends
      if (elements.legendItemsWildfire) elements.legendItemsWildfire.style.display = "block";
      if (elements.legendItemsFlood) elements.legendItemsFlood.style.display = "none";

      // Toggle panels & labels
      if (elements.panelDroughtSection) elements.panelDroughtSection.style.display = "block";
      if (elements.panelFloodSection) elements.panelFloodSection.style.display = "none";
      if (elements.metricsPanelTitle) elements.metricsPanelTitle.textContent = "Telemetry Metrics (Wildfire)";
      if (elements.metricPrimaryLabel) elements.metricPrimaryLabel.textContent = "Regional Carbon Flux";
      if (elements.studioPanelTitle) elements.studioPanelTitle.textContent = "Wildfire Burn Scar AI";
      if (elements.btnInferenceLabel) elements.btnInferenceLabel.textContent = "Run Spatio-Temporal Segmentation";
      if (elements.statPrimaryLabel) elements.statPrimaryLabel.textContent = "Affected Canopy:";
      if (elements.statSecondaryLabel) elements.statSecondaryLabel.textContent = "CO₂ Flux Est:";
      if (elements.thumbTagOverlay) elements.thumbTagOverlay.style.background = "rgba(239, 68, 68, 0.85)";

      if (elements.presetSelect) {
        elements.presetSelect.innerHTML = WILDFIRE_PRESET_OPTIONS;
      }
      state.activeLayer = "ndvi";
      state.activeMetric = "ndvi";
    } else {
      // Toggle layer pills
      if (elements.layerPillsWildfire) elements.layerPillsWildfire.style.display = "none";
      if (elements.layerPillsFlood) elements.layerPillsFlood.style.display = "flex";

      // Toggle chart tabs
      if (elements.chartTabsWildfire) elements.chartTabsWildfire.style.display = "none";
      if (elements.chartTabsFlood) elements.chartTabsFlood.style.display = "flex";

      // Toggle quick regions
      if (elements.quickRegionsWildfire) elements.quickRegionsWildfire.style.display = "none";
      if (elements.quickRegionsFlood) elements.quickRegionsFlood.style.display = "flex";

      // Toggle legends
      if (elements.legendItemsWildfire) elements.legendItemsWildfire.style.display = "none";
      if (elements.legendItemsFlood) elements.legendItemsFlood.style.display = "block";

      // Toggle panels & labels
      if (elements.panelDroughtSection) elements.panelDroughtSection.style.display = "none";
      if (elements.panelFloodSection) elements.panelFloodSection.style.display = "block";
      if (elements.metricsPanelTitle) elements.metricsPanelTitle.textContent = "Telemetry Metrics (Flash Flood)";
      if (elements.metricPrimaryLabel) elements.metricPrimaryLabel.textContent = "Flash Flood Susceptibility";
      if (elements.studioPanelTitle) elements.studioPanelTitle.textContent = "Flash Flood & Inundation AI";
      if (elements.btnInferenceLabel) elements.btnInferenceLabel.textContent = "Run Spatio-Temporal Flood AI";
      if (elements.statPrimaryLabel) elements.statPrimaryLabel.textContent = "Inundated Extent:";
      if (elements.statSecondaryLabel) elements.statSecondaryLabel.textContent = "Water Expansion:";
      if (elements.thumbTagOverlay) elements.thumbTagOverlay.style.background = "rgba(2, 132, 199, 0.90)";

      if (elements.presetSelect) {
        elements.presetSelect.innerHTML = FLOOD_PRESET_OPTIONS;
      }
      state.activeLayer = "flood_risk";
      state.activeMetric = "rainfall_mm";
    }

    // Refresh active states on buttons
    const activeLayerContainer = hazard === "wildfire" ? elements.layerPillsWildfire : elements.layerPillsFlood;
    if (activeLayerContainer) {
      activeLayerContainer.querySelectorAll(".layer-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.layer === state.activeLayer);
      });
    }

    const activeChartContainer = hazard === "wildfire" ? elements.chartTabsWildfire : elements.chartTabsFlood;
    if (activeChartContainer) {
      activeChartContainer.querySelectorAll(".chart-tab").forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.metric === state.activeMetric);
      });
    }

    updateActiveRasterLayer();
    drawChart(state.activeMetric);
    fetchRealTimeMetrics();
    showToast(`Switched mode to: ${hazard === "wildfire" ? "Wildfire & Biomass Engine" : "Flash Flood & Inundation Engine"}`);
  }

  function updateActiveRasterLayer() {
    updateLeafletTelemetryLayer();
    updateMapboxTelemetryLayer();
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
    }
    const c = state.currentRegion.center;
    const d = 1.5;
    return {
      lon_min: c[0] - d,
      lat_min: c[1] - d,
      lon_max: c[0] + d,
      lat_max: c[1] + d,
    };
  }

  async function loadConfig() {
    try {
      const res = await fetch(`${state.apiBase}/api/config`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (elements.statusGee) elements.statusGee.textContent = data.earth_engine.mode;
      if (elements.statusModel) elements.statusModel.textContent = data.segmentation_model.engine;
      if (elements.statusFloodModel && data.flood_segmentation_model) {
        elements.statusFloodModel.textContent = data.flood_segmentation_model.engine;
      }
    } catch (e) {
      console.warn("Config fetch fallback:", e);
    }
  }

  async function fetchRealTimeMetrics() {
    try {
      const bounds = getCurrentViewportBounds();
      const params = new URLSearchParams({
        lon_min: bounds.lon_min.toFixed(4),
        lat_min: bounds.lat_min.toFixed(4),
        lon_max: bounds.lon_max.toFixed(4),
        lat_max: bounds.lat_max.toFixed(4),
      });

      const res = await fetch(`${state.apiBase}/api/metrics?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (elements.metricTileSpeed) elements.metricTileSpeed.textContent = `< ${data.tile_stream_ms}ms`;
      if (elements.metricAnomalies) elements.metricAnomalies.textContent = `${data.active_anomalies_flagged} Alerts`;

      if (state.activeHazard === "wildfire") {
        if (elements.metricCarbon) {
          elements.metricCarbon.textContent = `+${data.carbon_flux_rate} t/ha`;
          elements.metricCarbon.style.color = "var(--solar-bright)";
        }
        if (elements.metricCarbonSub) elements.metricCarbonSub.textContent = "Annual Biomass Balance";
      } else {
        if (elements.metricCarbon) {
          const floodScore = data.planetary_flood_risk_index || state.currentFfsi || 84.2;
          elements.metricCarbon.textContent = `FFSI ${floodScore}%`;
          elements.metricCarbon.style.color = "#F97316";
        }
        if (elements.metricCarbonSub) elements.metricCarbonSub.textContent = "Flash Flood Susceptibility (FFSI)";
      }

      if (elements.metricClock) {
        elements.metricClock.textContent = `SYNC: ${data.live_telemetry_timestamp || new Date().toISOString().substring(11, 19) + ' UTC'}`;
      }
    } catch (e) {
      if (elements.metricClock) {
        elements.metricClock.textContent = `SYNC: ${new Date().toISOString().substring(11, 19)} UTC`;
      }
    }
  }

  async function loadNdviTelemetry(centerLon, centerLat) {
    const delta = 0.65;
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
      if (elements.chartStatus) {
        elements.chartStatus.textContent = state.activeHazard === "flood"
          ? "Multi-Sensor Radar & Rainfall Ingest Active"
          : data.carbon_flux_status;
      }

      drawChart(state.activeMetric);
    } catch (err) {
      console.warn("Telemetry fetch fallback:", err);
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

      // Dedicated flood correlated parameters
      const isSurge = i >= 18 && i <= 21;
      const rainfall_mm = +(isSurge ? (160 + Math.random() * 80) : (25 + Math.random() * 30)).toFixed(1);
      const soil_saturation = +(isSurge ? (82 + Math.random() * 12) : (40 + Math.random() * 20)).toFixed(1);
      const mndwi = +(isSurge ? (0.28 + Math.random() * 0.15) : (-0.12 + Math.random() * 0.08)).toFixed(4);
      const flood_risk_ffsi = +(isSurge ? (84 + Math.random() * 10) : (30 + Math.random() * 18)).toFixed(1);

      series.push({
        date: d.toISOString().split("T")[0],
        ndvi: i === 18 ? 0.32 : ndvi,
        ndwi: i === 18 ? -0.15 : ndwi,
        carbon_flux: i === 18 ? 8.4 : carbon,
        rainfall_mm: rainfall_mm,
        soil_saturation: soil_saturation,
        mndwi: mndwi,
        flood_risk_ffsi: flood_risk_ffsi,
        anomaly: isSurge || i === 18,
        z_score: isSurge ? 3.4 : (i === 18 ? 2.9 : 0.4),
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

    const values = state.timeseriesData.map((p) => (p[metricKey] !== undefined ? p[metricKey] : (p.ndvi || 0)));
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) { min *= 0.8; max *= 1.2; }
    min = min < 0 ? min * 1.1 : min * 0.85;
    max = max > 0 ? max * 1.1 : max * 0.9;
    const pad = 12;

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

    if (metricKey === "rainfall_mm") {
      strokeColor = "#8B5CF6"; // Cloudburst Purple
      glowColor = "rgba(139, 92, 246, 0.28)";
    } else if (metricKey === "soil_saturation") {
      strokeColor = "#0D9488"; // Soil Saturation Teal
      glowColor = "rgba(13, 148, 136, 0.28)";
    } else if (metricKey === "mndwi") {
      strokeColor = "#06B6D4"; // Water Index Aqua
      glowColor = "rgba(6, 182, 212, 0.28)";
    } else if (metricKey === "flood_risk_ffsi") {
      strokeColor = "#F97316"; // Flash Flood Amber/Orange
      glowColor = "rgba(249, 115, 22, 0.28)";
    } else if (metricKey === "ndwi") {
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
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Plot anomaly pulses
    state.timeseriesData.forEach((p, i) => {
      if (p.anomaly) {
        const px = getX(i);
        const py = getY(values[i]);
        const dotColor = state.activeHazard === "flood" ? "#F97316" : "#EF4444";

        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = `${dotColor}50`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = dotColor;
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

  function updateFloodRiskDisplay() {
    const ffsi = Number(state.currentFfsi) || 84.2;
    const thresh = Number(state.floodThreshold) || 60;

    if (elements.floodRiskScore) {
      elements.floodRiskScore.textContent = `FFSI ${ffsi.toFixed(1)}%`;
    }
    if (elements.floodSaturationBar) {
      elements.floodSaturationBar.style.width = `${Math.min(100, Math.max(5, ffsi))}%`;
    }

    let riskClass = "Nominal Drainage";
    let badgeColor = "#34D399";
    let barColor = "linear-gradient(90deg, #10b981 0%, #34d399 100%)";

    if (ffsi >= thresh + 18) {
      riskClass = "Flash Flood Emergency";
      badgeColor = "#EF4444";
      barColor = "linear-gradient(90deg, #F97316 0%, #EF4444 100%)";
    } else if (ffsi >= thresh) {
      riskClass = "Flash Flood Warning";
      badgeColor = "#F97316";
      barColor = "linear-gradient(90deg, #FB923C 0%, #F97316 100%)";
    } else if (ffsi >= 40) {
      riskClass = "Flash Flood Watch";
      badgeColor = "#FBBF24";
      barColor = "linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)";
    }

    if (elements.floodRiskClass) {
      elements.floodRiskClass.textContent = riskClass;
      elements.floodRiskClass.style.color = badgeColor;
    }
    if (elements.floodRiskScore) {
      elements.floodRiskScore.style.color = badgeColor;
    }
    if (elements.floodSaturationBar) {
      elements.floodSaturationBar.style.background = barColor;
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
    } catch (e) {
      state.currentVci = 68.4;
      updateVciDisplay();
    }
  }

  async function loadFlashFloodRisk(centerLon, centerLat) {
    const delta = 0.85;
    const params = new URLSearchParams({
      lon_min: (centerLon - delta).toFixed(4),
      lat_min: (centerLat - delta).toFixed(4),
      lon_max: (centerLon + delta).toFixed(4),
      lat_max: (centerLat + delta).toFixed(4),
    });

    try {
      const res = await fetch(`${state.apiBase}/api/flood/risk?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      state.currentFfsi = Number(data.flash_flood_susceptibility_pct) || 84.2;
      updateFloodRiskDisplay();

      if (elements.floodStatSaturation) elements.floodStatSaturation.textContent = `${data.soil_saturation_pct}%`;
      if (elements.floodStatPrecip) elements.floodStatPrecip.textContent = `+${data.precipitation_anomaly_mm} mm`;
      if (elements.floodStatDeforestation) elements.floodStatDeforestation.textContent = `${data.deforestation_pct || data.deforestation_loss_pct || '34.5'}%`;
      if (elements.floodStatCn) elements.floodStatCn.textContent = `${data.runoff_factor_cn} (High)`;
      if (elements.floodStatTwi) elements.floodStatTwi.textContent = `${data.topographic_wetness_index} TWI`;
      if (elements.floodActionText && data.recommended_action) {
        elements.floodActionText.textContent = data.recommended_action;
      }
    } catch (e) {
      state.currentFfsi = 84.2;
      updateFloodRiskDisplay();
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
      const isCrit = alert.severity === "CRITICAL";
      const isFlood = alert.type.toLowerCase().includes("flood") || alert.type.toLowerCase().includes("inundation");
      const badgeColor = isFlood ? "rgba(249, 115, 22, 0.20)" : "rgba(239, 68, 68, 0.20)";
      const textColor = isFlood ? "#FB923C" : (isCrit ? "#F87171" : "#FBBF24");

      const card = document.createElement("div");
      card.className = "alert-card";
      card.innerHTML = `
        <div class="alert-card-top">
          <span class="alert-title">${alert.title}</span>
          <span class="severity-pill" style="background:${badgeColor};color:${textColor};border-color:${textColor}40">${alert.severity}</span>
        </div>
        <div class="alert-meta-row">
          <span>${alert.type}</span>
          <span style="color: ${isFlood ? '#FB923C' : 'var(--veg-bright)'}; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px;">● ${alert.detected_at || 'LIVE'}</span>
        </div>
        <div class="alert-meta-row" style="color: ${textColor}">
          <span>Impact: ${alert.loss_hectares ? alert.loss_hectares.toLocaleString() : '--'} ha</span>
          <span>Sensor: ${alert.sensor || 'Sentinel-1/2'}</span>
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

  function getGeographicDescriptor(lat, lon) {
    if (lat >= 26 && lat <= 31 && lon >= 80 && lon <= 90) return `Nepal & Tibet Himalayan Basin AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 8 && lat <= 32 && lon >= 68 && lon <= 96) return `India (Indo-Gangetic Basin) AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 36 && lat <= 44 && lon >= -10 && lon <= 5) return `Iberian Peninsula & Valencia AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 20 && lat <= 27 && lon >= 88 && lon <= 93) return `Bangladesh River Delta AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -15 && lat <= 6 && lon >= -78 && lon <= -45) return `Amazon Rainforest AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= -10 && lat <= 8 && lon >= 12 && lon <= 32) return `Congo Rainforest AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    if (lat >= 32 && lat <= 44 && lon >= -125 && lon <= -114) return `California & Sierra Nevada AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
    return `Planetary AOI [${lat.toFixed(1)}°, ${lon.toFixed(1)}°]`;
  }

  async function runSegmentationInference() {
    const isFlood = state.activeHazard === "flood";
    const preset = elements.presetSelect ? elements.presetSelect.value : (isFlood ? "nepal" : "california");

    if (elements.btnRunInference) elements.btnRunInference.disabled = true;
    if (elements.studioStatus) {
      elements.studioStatus.textContent = isFlood
        ? "Executing Spatio-Temporal Flood U-Net & SAR anomaly extraction..."
        : "Executing Spatio-Temporal Wildfire U-Net inference...";
    }

    const endpoint = isFlood ? "/api/inference/flood" : "/api/inference/wildfire";
    let url = `${state.apiBase}${endpoint}?preset=${preset}`;

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

      url = `${state.apiBase}${endpoint}?preset=global_scan&lon_min=${lon_min}&lat_min=${lat_min}&lon_max=${lon_max}&lat_max=${lat_max}&region_name=${encodeURIComponent(cleanRegionLabel)}`;
    } else if (PRESET_LOCATIONS[preset]) {
      const loc = PRESET_LOCATIONS[preset];
      if (state.currentEngine === "leaflet" && state.leafletInstance) {
        state.leafletInstance.flyTo([loc.center[1], loc.center[0]], loc.zoom, { duration: 1.2 });
      } else if (state.mapboxInstance) {
        state.mapboxInstance.flyTo({ center: loc.center, zoom: loc.zoom, pitch: 25, essential: true });
      }
    }

    try {
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) throw new Error(`Inference returned HTTP ${res.status}`);
      const result = await res.json();

      if (elements.thumbPre) elements.thumbPre.src = `data:image/png;base64,${result.visuals.pre_scene_b64}`;
      if (elements.thumbPost) elements.thumbPost.src = `data:image/png;base64,${result.visuals.post_scene_b64}`;
      if (elements.thumbOverlay) elements.thumbOverlay.src = `data:image/png;base64,${result.visuals.overlay_b64}`;

      if (isFlood) {
        if (elements.statBurnedHa) elements.statBurnedHa.textContent = `${result.inundated_area_hectares.toLocaleString()} ha (${result.inundated_percentage}%)`;
        if (elements.statCo2) elements.statCo2.textContent = `${result.water_expansion_ratio}x Surge (${result.sar_backscatter_drop_db} dB)`;
        if (elements.statLatency) elements.statLatency.textContent = `${result.inference_ms}ms · SAR ${result.model_anomaly_metrics.spectral_anomaly_zscore}σ`;
      } else {
        if (elements.statBurnedHa) elements.statBurnedHa.textContent = `${result.burned_area_hectares.toLocaleString()} ha (${result.burned_canopy_percentage}%)`;
        if (elements.statCo2) elements.statCo2.textContent = `+${result.estimated_co2_kt} kt CO₂`;
        if (elements.statLatency) elements.statLatency.textContent = `${result.inference_ms}ms · 10m Ground Res`;
      }

      if (elements.statSeverityBadge) {
        const sev = result.severity_level || (isFlood ? "HIGH - SEVERE FLASH FLOOD SURGE" : "HIGH - ACTIVE THERMAL BURN SCAR");
        elements.statSeverityBadge.textContent = sev;
        if (sev.includes("CRITICAL")) {
          elements.statSeverityBadge.style.color = "#EF4444";
        } else if (sev.includes("HIGH")) {
          elements.statSeverityBadge.style.color = "#F97316";
        } else if (sev.includes("MEDIUM") || sev.includes("MODERATE")) {
          elements.statSeverityBadge.style.color = "#FBBF24";
        } else if (sev.includes("LOW")) {
          elements.statSeverityBadge.style.color = "#34D399";
        } else if (sev.includes("NONE")) {
          elements.statSeverityBadge.style.color = "#38BDF8";
        } else {
          elements.statSeverityBadge.style.color = "#94A3B8";
        }
      }

      if (elements.studioStatus) elements.studioStatus.textContent = `Completed · ${result.title}`;

      plotSegmentationOnMap(result, isFlood);

      const area = isFlood ? result.inundated_area_hectares : result.burned_area_hectares;
      showToast(`AI Segmentation complete: ${area.toLocaleString()} ha mapped.`);
    } catch (err) {
      console.warn("Inference demo fallback:", err);
      if (elements.studioStatus) elements.studioStatus.textContent = "Inference complete.";
    } finally {
      if (elements.btnRunInference) elements.btnRunInference.disabled = false;
    }
  }

  function plotSegmentationOnMap(result, isFlood) {
    if (!result.geojson || !result.bbox) return;

    // Distinct theme colors: Inundation = Electric Deep Blue #0284C7, Wildfire = Flame Red #EF4444
    const themeColor = isFlood ? "#0284C7" : "#EF4444";
    const headerColor = isFlood ? "#38BDF8" : "#F87171";

    const popupContent = isFlood ? `
      <div style="font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:#fff; min-width:230px;">
        <div style="font-weight:700; color:#38BDF8; margin-bottom:4px; font-size:12.5px;">${result.title}</div>
        <div style="color:#67E8F9; margin-bottom:6px; font-size:10.5px;">Multi-Modal SAR &amp; Optical Inundation Map (10m)</div>
        <div style="margin-bottom:3px;">Inundated Area: <strong style="color:#fff;">${result.inundated_area_hectares.toLocaleString()} ha</strong> (${result.inundated_percentage}%)</div>
        <div style="margin-bottom:3px;">Water Expansion: <strong style="color:#38BDF8;">${result.water_expansion_ratio}x Baseline</strong></div>
        <div style="margin-bottom:3px;">SAR Attenuation: <strong style="color:#F97316;">${result.sar_backscatter_drop_db} dB</strong></div>
        <div style="color:#94A3B8; font-size:10px; margin-top:4px;">Inference Latency: ${result.inference_ms}ms</div>
      </div>
    ` : `
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
          color: themeColor,
          weight: 3,
          opacity: 0.95,
          fillColor: themeColor,
          fillOpacity: 0.45,
          dashArray: isFlood ? "4, 4" : "6, 6",
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
            "fill-color": themeColor,
            "fill-opacity": 0.45,
          },
        });

        state.mapboxInstance.addLayer({
          id: lineLayerId,
          type: "line",
          source: sourceId,
          paint: {
            "line-color": headerColor,
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
      // Wildfire Regions
      amazon: { name: "Amazon Basin, Brazil", center: [-62.5, -4.5], zoom: 5.5, sensor: "Sentinel-2 MSI", hazard: state.activeHazard },
      california: { name: "Sierra Nevada, USA", center: [-119.5, 37.2], zoom: 6.2, sensor: "Sentinel-2 MSI", hazard: "wildfire" },
      congo: { name: "Congo Rainforest, DRC", center: [23.6, -0.5], zoom: 5.2, sensor: "Sentinel-2 MSI", hazard: "wildfire" },
      borneo: { name: "Borneo Peatlands, Indonesia", center: [113.9, 0.5], zoom: 5.8, sensor: "Landsat-9 OLI", hazard: "wildfire" },
      pantanal: { name: "Pantanal Wetlands, Brazil", center: [-56.5, -17.8], zoom: 5.8, sensor: "Sentinel-2 MSI", hazard: "wildfire" },
      sahel: { name: "Sahel & Lake Chad, Africa", center: [14.5, 13.8], zoom: 5.0, sensor: "Sentinel-2 MSI", hazard: "wildfire" },
      siberia: { name: "Siberian Taiga, Russia", center: [129.5, 62.2], zoom: 4.8, sensor: "Sentinel-2 MSI", hazard: "wildfire" },

      // Flash Flood Regions
      nepal: { name: "Nepal & Tibet Mountain Basin & River Valleys", center: [85.65, 27.22], zoom: 6.8, sensor: "Sentinel-1 SAR & S2", hazard: "flood" },
      india: { name: "India (Ganges & Brahmaputra Corridor)", center: [86.00, 26.15], zoom: 6.2, sensor: "Sentinel-1 SAR (10m)", hazard: "flood" },
      indo_gangetic: { name: "India (Ganges & Brahmaputra Corridor)", center: [86.00, 26.15], zoom: 6.2, sensor: "Sentinel-1 SAR (10m)", hazard: "flood" },
      indian_subcontinent: { name: "India (Ganges & Brahmaputra Corridor)", center: [86.00, 26.15], zoom: 6.2, sensor: "Sentinel-1 SAR (10m)", hazard: "flood" },
      valencia: { name: "Valencia DANA Flash Flood Basin, Spain", center: [-0.40, 39.40], zoom: 7.0, sensor: "Sentinel-1 & Sentinel-2", hazard: "flood" },
      bangladesh: { name: "Padma & Meghna Delta, Bangladesh", center: [90.35, 23.65], zoom: 6.5, sensor: "Sentinel-1 SAR GRD", hazard: "flood" },
      libya: { name: "Derna Wadi Flash Flood Basin, Libya", center: [22.63, 32.76], zoom: 7.2, sensor: "Sentinel-1 SAR GRD", hazard: "flood" },
    };

    const target = presets[regionKey] || presets.amazon;
    state.currentRegion = { id: regionKey, ...target };

    if (target.hazard && target.hazard !== state.activeHazard) {
      switchHazardMode(target.hazard);
    }

    if (state.currentEngine === "leaflet" && state.leafletInstance) {
      state.leafletInstance.flyTo([target.center[1], target.center[0]], target.zoom, { duration: 1.5 });
    } else if (state.mapboxInstance) {
      state.mapboxInstance.flyTo({ center: target.center, zoom: target.zoom, pitch: 28, essential: true });
    }

    if (elements.hudRegion) elements.hudRegion.textContent = target.name.toUpperCase();
    if (elements.hudSensor) elements.hudSensor.textContent = `SENSOR: ${target.sensor}`;

    document.querySelectorAll(".quick-region-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.region === regionKey);
    });

    loadNdviTelemetry(target.center[0], target.center[1]);
    loadDroughtAssessment(target.center[0], target.center[1]);
    loadFlashFloodRisk(target.center[0], target.center[1]);
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
      title: "EcoPulse Planetary Multi-Hazard Climate Telemetry Summary",
      timestamp: new Date().toISOString(),
      activeMapEngine: state.currentEngine,
      activeHazardMode: state.activeHazard,
      region: state.currentRegion,
      activeLayer: state.activeLayer,
      telemetry: state.timeseriesData,
      alerts: state.alertsData,
      flashFloodRiskScore: state.currentFfsi,
      droughtConditionIndex: state.currentVci,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ecopulse-report-${state.currentRegion.id}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Planetary multi-hazard telemetry report exported successfully.");
  }

  function startRealTimeTelemetryLoop() {
    if (state.telemetryPollTimer) clearInterval(state.telemetryPollTimer);
    state.telemetryPollTimer = setInterval(() => {
      fetchRealTimeMetrics();
    }, 12000);
  }

  function attachEventListeners() {
    if (elements.tabHazardWildfire) {
      elements.tabHazardWildfire.addEventListener("click", () => switchHazardMode("wildfire"));
    }
    if (elements.tabHazardFlood) {
      elements.tabHazardFlood.addEventListener("click", () => switchHazardMode("flood"));
    }

    if (elements.btnEngineLeaflet) {
      elements.btnEngineLeaflet.addEventListener("click", () => switchMapEngine("leaflet"));
    }
    if (elements.btnEngineMapbox) {
      elements.btnEngineMapbox.addEventListener("click", () => switchMapEngine("mapbox"));
    }
    if (elements.btnZoomIn) {
      elements.btnZoomIn.addEventListener("click", handleZoomIn);
    }
    if (elements.btnZoomOut) {
      elements.btnZoomOut.addEventListener("click", handleZoomOut);
    }
    if (elements.btnLockZoom) {
      elements.btnLockZoom.addEventListener("click", () => toggleZoomLock());
    }
    if (elements.btnRecenter) {
      elements.btnRecenter.addEventListener("click", handleRecenter);
    }
    if (elements.btnScanViewport) {
      elements.btnScanViewport.addEventListener("click", () => {
        if (elements.presetSelect) elements.presetSelect.value = "viewport";
        runSegmentationInference();
      });
    }
    if (elements.btnFullscreen) {
      elements.btnFullscreen.addEventListener("click", handleFullscreen);
    }

    function toggleSidebar(forceOpen) {
      if (!elements.sidebar) return;
      if (typeof forceOpen === "boolean") {
        elements.sidebar.classList.toggle("collapsed", !forceOpen);
      } else {
        elements.sidebar.classList.toggle("collapsed");
      }
      const isCollapsed = elements.sidebar.classList.contains("collapsed");
      if (elements.sidebarToggleTab) {
        elements.sidebarToggleTab.setAttribute("title", isCollapsed ? "Expand Telemetry Panel" : "Collapse Telemetry Panel");
      }
      if (elements.btnToggleDashboard) {
        elements.btnToggleDashboard.classList.toggle("active", !isCollapsed);
      }
      showToast(isCollapsed ? "Telemetry dashboard collapsed." : "Telemetry dashboard open.");
      if (state.leafletInstance) {
        setTimeout(() => state.leafletInstance.invalidateSize(), 300);
      }
      if (state.mapboxInstance) {
        setTimeout(() => state.mapboxInstance.resize(), 300);
      }
    }

    if (elements.sidebarToggleTab) {
      elements.sidebarToggleTab.addEventListener("click", () => toggleSidebar());
    }

    if (elements.btnToggleDashboard) {
      elements.btnToggleDashboard.addEventListener("click", () => toggleSidebar());
    }

    if (elements.btnForceLandscape) {
      elements.btnForceLandscape.addEventListener("click", async () => {
        try {
          if (screen.orientation && screen.orientation.lock) {
            await screen.orientation.lock("landscape");
          } else if (document.documentElement.requestFullscreen) {
            await document.documentElement.requestFullscreen();
          }
        } catch (e) {
          showToast("Please rotate your phone horizontally to landscape view.");
        }
      });
    }

    document.querySelectorAll(".quick-region-btn").forEach((btn) => {
      btn.addEventListener("click", () => setRegion(btn.dataset.region));
    });

    document.querySelectorAll(".layer-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const parent = btn.closest(".layer-pills");
        if (parent) {
          parent.querySelectorAll(".layer-btn").forEach((b) => b.classList.remove("active"));
        }
        btn.classList.add("active");
        state.activeLayer = btn.dataset.layer;
        updateActiveRasterLayer();
        showToast(`Switched active layer to: ${btn.textContent.trim()}`);
      });
    });

    document.querySelectorAll(".chart-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const parent = tab.closest(".chart-tabs");
        if (parent) {
          parent.querySelectorAll(".chart-tab").forEach((t) => t.classList.remove("active"));
        }
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

    if (elements.floodSlider) {
      elements.floodSlider.addEventListener("input", (e) => {
        const val = Number(e.target.value);
        state.floodThreshold = val;
        if (elements.floodSliderVal) {
          const desc = val <= 35 ? "Watch" : val <= 65 ? "Warning" : "Emergency";
          elements.floodSliderVal.textContent = `${val}% (${desc})`;
        }
        updateFloodRiskDisplay();
      });
    }

    if (elements.btnRunInference) {
      elements.btnRunInference.addEventListener("click", runSegmentationInference);
    }
    if (elements.presetSelect) {
      elements.presetSelect.addEventListener("change", () => {
        if (elements.presetSelect.value !== "viewport") {
          runSegmentationInference();
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
    runSegmentationInference();

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
