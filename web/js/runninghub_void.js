import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAME = "RunningHub Void Point Editor";
const MIN_NODE_WIDTH = 380;
const MIN_NODE_HEIGHT = 220;
const VIDEO_ACCEPT = ".mp4,.mov,.mkv,.webm,.avi,.gif,video/mp4,video/quicktime,video/webm,video/x-msvideo,video/x-matroska,image/gif";
const IMAGE_PREVIEW_EXTENSIONS = new Set(["gif", "webp", "avif"]);
const STATUS_BAR_MIN_HEIGHT = 168;
const NODE_BASE_HEIGHT = 248;
const MIN_PREVIEW_HEIGHT = 180;
const MAX_PREVIEW_HEIGHT = 420;

function hideWidgetForGood(node, widget, suffix = "") {
    if (!widget) {
        return;
    }

    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget" + suffix;
    widget.hidden = true;

    if (widget.element) {
        widget.element.style.display = "none";
    }

    if (widget.linkedWidgets) {
        for (const linkedWidget of widget.linkedWidgets) {
            hideWidgetForGood(node, linkedWidget, ":" + widget.name);
        }
    }
}

function addMenuHandler(nodeType, cb) {
    const getOpts = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function () {
        const r = getOpts?.apply(this, arguments);
        const maybeOptions = Array.isArray(arguments[1]) ? arguments[1] : (Array.isArray(r) ? r : null);
        cb.call(this, arguments[0], maybeOptions, r);
        return r;
    };
}

function parseFramePoints(rawValue) {
    if (!rawValue || typeof rawValue !== "string") {
        return {};
    }

    try {
        const parsed = JSON.parse(rawValue);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            return {};
        }

        const normalized = {};
        for (const [frameKey, points] of Object.entries(parsed)) {
            const frameIndex = Number.parseInt(frameKey, 10);
            if (!Number.isFinite(frameIndex) || frameIndex < 0 || !Array.isArray(points)) {
                continue;
            }

            const normalizedPoints = points
                .filter((point) => Array.isArray(point) && point.length === 2)
                .map((point) => [
                    Math.round(Number(point[0]) || 0),
                    Math.round(Number(point[1]) || 0),
                ]);

            if (normalizedPoints.length) {
                normalized[String(frameIndex)] = normalizedPoints;
            }
        }

        return normalized;
    } catch {
        return {};
    }
}

function stringifyFramePoints(pointsByFrame) {
    const sortedKeys = Object.keys(pointsByFrame || {})
        .map((key) => Number.parseInt(key, 10))
        .filter((value) => Number.isFinite(value) && value >= 0)
        .sort((a, b) => a - b);

    const normalized = {};
    for (const frameIndex of sortedKeys) {
        const points = pointsByFrame[String(frameIndex)] || [];
        const cleaned = points
            .filter((point) => Array.isArray(point) && point.length === 2)
            .map((point) => [
                Math.round(Number(point[0]) || 0),
                Math.round(Number(point[1]) || 0),
            ]);
        if (cleaned.length) {
            normalized[String(frameIndex)] = cleaned;
        }
    }

    return JSON.stringify(normalized);
}

function cloneFramePoints(pointsByFrame) {
    return JSON.parse(JSON.stringify(pointsByFrame || {}));
}

function getFileExtension(filename) {
    const parts = String(filename || "").split(".");
    return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : "";
}

function getPreviewParams(filename) {
    const extension = getFileExtension(filename);
    const isImagePreview = IMAGE_PREVIEW_EXTENSIONS.has(extension);
    return {
        filename,
        type: "input",
        format: `${isImagePreview ? "image" : "video"}/${extension || "mp4"}`,
        isImagePreview,
    };
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#39;");
}

function buildStatusMarkup(text) {
    return String(text || "")
        .split("\n")
        .filter(Boolean)
        .map((line) => {
            const separatorIndex = line.indexOf(":");
            if (separatorIndex === -1) {
                return `<div style="display:flex; align-items:flex-start; gap:8px; margin:0 0 4px 0; overflow-wrap:anywhere;"><span style="color: var(--rh-void-status-value, #e4e8ee);">${escapeHtml(line)}</span></div>`;
            }
            const label = line.slice(0, separatorIndex + 1);
            const value = line.slice(separatorIndex + 1).trimStart();
            return `<div style="display:flex; align-items:flex-start; gap:8px; margin:0 0 4px 0; overflow-wrap:anywhere;"><span style="flex:0 0 auto; color: var(--rh-void-status-label, #93a3b8); font-weight:600;">${escapeHtml(label)}</span><span style="flex:1 1 auto; color: var(--rh-void-status-value, #e4e8ee);">${escapeHtml(value)}</span></div>`;
        })
        .join("");
}

function getPreviewHeight(width, aspectRatio, visible) {
    if (!visible) {
        return 0;
    }
    const safeWidth = Math.max((width || MIN_NODE_WIDTH) - 20, MIN_NODE_WIDTH - 20);
    const ratio = Number.isFinite(aspectRatio) && aspectRatio > 0 ? aspectRatio : (16 / 9);
    return Math.max(MIN_PREVIEW_HEIGHT, Math.min(MAX_PREVIEW_HEIGHT, safeWidth / ratio));
}

function countPoints(pointsByFrame) {
    return Object.values(pointsByFrame || {}).reduce((sum, points) => sum + points.length, 0);
}

function summarizeFrames(pointsByFrame) {
    const keys = Object.keys(pointsByFrame || {}).sort((a, b) => Number(a) - Number(b));
    if (!keys.length) {
        return "none";
    }
    return keys.join(", ");
}

function parsePositiveInt(value, fallback = 1) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function findClosestSampleIndex(sourceFrameIndices, targetFrame) {
    if (!Array.isArray(sourceFrameIndices) || !sourceFrameIndices.length) {
        return 0;
    }

    let bestIndex = 0;
    let bestDistance = Math.abs((sourceFrameIndices[0] ?? 0) - targetFrame);
    for (let index = 1; index < sourceFrameIndices.length; index += 1) {
        const distance = Math.abs((sourceFrameIndices[index] ?? 0) - targetFrame);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
        }
    }
    return bestIndex;
}

async function uploadVideoFile(file) {
    const body = new FormData();
    const uploadFile = new File([file], file.name, {
        type: file.type,
        lastModified: file.lastModified,
    });
    body.append("image", uploadFile);

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });

    if (response.status !== 200) {
        throw new Error("Upload failed");
    }

    const data = await response.json();
    if (!data?.name) {
        throw new Error("Upload response missing filename");
    }

    return data.name;
}

async function fetchPreviewFrames(filename, frameStride) {
    const response = await api.fetchApi("/rh/void/preview_frames", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            filename,
            frame_stride: parsePositiveInt(frameStride, 1),
        }),
    });

    const payload = await response.json();
    if (response.status !== 200) {
        throw new Error(payload?.error || "Failed to extract video frames.");
    }

    return payload;
}

class VoidEditorDialog {
    static instance = null;

    static getInstance() {
        if (!VoidEditorDialog.instance) {
            VoidEditorDialog.instance = new VoidEditorDialog();
        }
        return VoidEditorDialog.instance;
    }

    constructor() {
        this.node = null;
        this.previewFrames = [];
        this.sourceFrameIndices = [];
        this.frameImages = new Map();
        this.currentSampleIndex = 0;
        this.workingPoints = {};

        this.overlay = document.createElement("div");
        this.overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.72);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 99999;
        `;

        this.panel = document.createElement("div");
        this.panel.style.cssText = `
            width: min(1080px, calc(100vw - 40px));
            height: min(820px, calc(100vh - 40px));
            background: #111315;
            color: #f4f4f4;
            border: 1px solid #3b3b3b;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
            overflow: hidden;
        `;

        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 14px 16px;
            border-bottom: 1px solid #2d2d2d;
            background: #17191c;
        `;

        const titleWrap = document.createElement("div");
        titleWrap.style.cssText = "display: flex; flex-direction: column; gap: 4px;";

        this.title = document.createElement("div");
        this.title.style.cssText = "font-size: 16px; font-weight: 600;";
        this.title.textContent = "Void Video Point Editor";

        this.subtitle = document.createElement("div");
        this.subtitle.style.cssText = "font-size: 12px; color: #a6a6a6;";
        this.subtitle.textContent = "Left click to add positive points on key frames. Save stores primary_points_by_frame.";

        titleWrap.appendChild(this.title);
        titleWrap.appendChild(this.subtitle);

        const actionWrap = document.createElement("div");
        actionWrap.style.cssText = "display: flex; gap: 8px; flex-wrap: wrap;";

        this.undoButton = this.createButton("Undo", () => this.undoPoint(), false);
        this.clearFrameButton = this.createButton("Clear Frame", () => this.clearFrame(), false);
        this.clearAllButton = this.createButton("Clear All", () => this.clearAll(), false);
        this.cancelButton = this.createButton("Cancel", () => this.close(), false);
        this.saveButton = this.createButton("Save", () => this.save(), true);

        actionWrap.appendChild(this.undoButton);
        actionWrap.appendChild(this.clearFrameButton);
        actionWrap.appendChild(this.clearAllButton);
        actionWrap.appendChild(this.cancelButton);
        actionWrap.appendChild(this.saveButton);

        header.appendChild(titleWrap);
        header.appendChild(actionWrap);

        const frameBar = document.createElement("div");
        frameBar.style.cssText = `
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 16px;
            border-bottom: 1px solid #262626;
            background: #141618;
        `;

        this.prevButton = this.createButton("<", () => this.jumpFrame(-1), false);
        this.nextButton = this.createButton(">", () => this.jumpFrame(1), false);
        this.frameLabel = document.createElement("div");
        this.frameLabel.style.cssText = "min-width: 96px; font-family: monospace; color: #d7d7d7;";
        this.frameLabel.textContent = "Frame 0/0";

        this.slider = document.createElement("input");
        this.slider.type = "range";
        this.slider.min = "0";
        this.slider.max = "0";
        this.slider.step = "1";
        this.slider.value = "0";
        this.slider.style.cssText = "flex: 1;";
        this.slider.addEventListener("input", async (event) => {
            await this.syncStrideFromNode();
            this.currentSampleIndex = Number.parseInt(event.target.value, 10) || 0;
            this.render();
        });

        this.framesSummary = document.createElement("div");
        this.framesSummary.style.cssText = "min-width: 180px; text-align: right; font-size: 12px; color: #a6a6a6;";
        this.framesSummary.textContent = "Frames: none";

        frameBar.appendChild(this.prevButton);
        frameBar.appendChild(this.frameLabel);
        frameBar.appendChild(this.slider);
        frameBar.appendChild(this.nextButton);
        frameBar.appendChild(this.framesSummary);

        this.canvasWrap = document.createElement("div");
        this.canvasWrap.style.cssText = `
            flex: 1;
            padding: 16px;
            background: #0f1012;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
        `;

        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `
            display: block;
            max-width: 100%;
            max-height: 100%;
            background: #1d1f22;
            border: 1px solid #333;
            border-radius: 8px;
            cursor: crosshair;
        `;
        this.ctx = this.canvas.getContext("2d");
        this.canvas.addEventListener("click", (event) => this.addPointFromClick(event));
        this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
        this.canvasWrap.appendChild(this.canvas);

        const footer = document.createElement("div");
        footer.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 16px;
            border-top: 1px solid #262626;
            background: #141618;
            font-size: 12px;
            color: #b1b1b1;
        `;

        this.status = document.createElement("div");
        this.status.textContent = "Upload a video first.";

        this.hint = document.createElement("div");
        this.hint.style.cssText = "color: #8e8e8e;";
        this.hint.textContent = "Coordinates are saved in original frame space.";

        footer.appendChild(this.status);
        footer.appendChild(this.hint);

        this.panel.appendChild(header);
        this.panel.appendChild(frameBar);
        this.panel.appendChild(this.canvasWrap);
        this.panel.appendChild(footer);
        this.overlay.appendChild(this.panel);
        document.body.appendChild(this.overlay);
    }

    createButton(label, onClick, primary) {
        const button = document.createElement("button");
        button.textContent = label;
        button.style.cssText = `
            border: 1px solid ${primary ? "#9f3434" : "#4a4a4a"};
            background: ${primary ? "#8c2626" : "#26292d"};
            color: #f7f7f7;
            border-radius: 6px;
            padding: 6px 10px;
            cursor: pointer;
            font-size: 12px;
            min-width: 64px;
        `;
        button.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            Promise.resolve(onClick()).catch((error) => {
                console.error(error);
            });
        });
        return button;
    }

    async show(node) {
        const state = node?._rhVoid;
        if (!state?.widgets?.upload?.value || state.widgets.uploadStatus?.value !== "success") {
            window.alert("Upload and preprocess a video successfully before opening the editor.");
            return;
        }

        if (!state.previewFrames?.length) {
            window.alert("This video has not finished preprocessing yet.");
            return;
        }

        const preserveFrame = state.sourceFrameIndices?.[state.lastFrame ?? 0] ?? 0;
        if (state.syncPreviewStride) {
            await state.syncPreviewStride(preserveFrame);
        }

        this.node = node;
        this.previewFrames = state.previewFrames;
        this.sourceFrameIndices = state.sourceFrameIndices?.length
            ? state.sourceFrameIndices
            : this.previewFrames.map((_, index) => index);
        this.frameImages = new Map();
        this.currentSampleIndex = Math.min(state.lastFrame ?? 0, Math.max(this.previewFrames.length - 1, 0));
        this.workingPoints = cloneFramePoints(state.pointsByFrame);
        this.slider.max = String(Math.max(this.previewFrames.length - 1, 0));
        this.slider.value = String(this.currentSampleIndex);
        this.overlay.style.display = "flex";
        this.title.textContent = `Void Video Point Editor: ${state.widgets.upload.value}`;
        await this.render();
    }

    close() {
        if (this.node?._rhVoid) {
            this.node._rhVoid.lastFrame = this.currentSampleIndex;
        }
        this.overlay.style.display = "none";
        this.node = null;
    }

    async loadFrameImage(frameIndex) {
        if (this.frameImages.has(frameIndex)) {
            return this.frameImages.get(frameIndex);
        }

        const image = new Image();
        image.src = `data:image/png;base64,${this.previewFrames[frameIndex]}`;
        await new Promise((resolve, reject) => {
            image.onload = resolve;
            image.onerror = reject;
        });
        this.frameImages.set(frameIndex, image);
        return image;
    }

    async render() {
        if (!this.previewFrames.length) {
            this.status.textContent = "No preview frames available.";
            return;
        }

        this.slider.value = String(this.currentSampleIndex);
        const image = await this.loadFrameImage(this.currentSampleIndex);
        this.canvas.width = image.width;
        this.canvas.height = image.height;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(image, 0, 0, this.canvas.width, this.canvas.height);

        const sourceFrameIndex = this.sourceFrameIndices[this.currentSampleIndex] ?? this.currentSampleIndex;
        const points = this.workingPoints[String(sourceFrameIndex)] || [];
        for (let index = 0; index < points.length; index += 1) {
            const [x, y] = points[index];
            this.ctx.beginPath();
            this.ctx.arc(x, y, 8, 0, Math.PI * 2);
            this.ctx.fillStyle = "#ff2e2e";
            this.ctx.fill();
            this.ctx.lineWidth = 2;
            this.ctx.strokeStyle = "#ffffff";
            this.ctx.stroke();
            this.ctx.fillStyle = "#ffffff";
            this.ctx.font = "13px sans-serif";
            this.ctx.fillText(String(index + 1), x + 10, y - 10);
        }

        this.frameLabel.textContent = `Frame ${sourceFrameIndex}/${this.sourceFrameIndices[this.sourceFrameIndices.length - 1] ?? sourceFrameIndex}`;
        this.framesSummary.textContent = `Frames: ${summarizeFrames(this.workingPoints)}`;
        this.status.textContent = `${countPoints(this.workingPoints)} points across ${Object.keys(this.workingPoints).length} frame(s).`;
    }

    addPointFromClick(event) {
        if (!this.previewFrames.length) {
            return;
        }

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const x = Math.max(0, Math.min(this.canvas.width - 1, Math.round((event.clientX - rect.left) * scaleX)));
        const y = Math.max(0, Math.min(this.canvas.height - 1, Math.round((event.clientY - rect.top) * scaleY)));

        const sourceFrameIndex = this.sourceFrameIndices[this.currentSampleIndex] ?? this.currentSampleIndex;
        const frameKey = String(sourceFrameIndex);
        if (!this.workingPoints[frameKey]) {
            this.workingPoints[frameKey] = [];
        }
        this.workingPoints[frameKey].push([x, y]);
        this.render();
    }

    undoPoint() {
        const sourceFrameIndex = this.sourceFrameIndices[this.currentSampleIndex] ?? this.currentSampleIndex;
        const frameKey = String(sourceFrameIndex);
        const points = this.workingPoints[frameKey];
        if (!points?.length) {
            return;
        }
        points.pop();
        if (!points.length) {
            delete this.workingPoints[frameKey];
        }
        this.render();
    }

    clearFrame() {
        const sourceFrameIndex = this.sourceFrameIndices[this.currentSampleIndex] ?? this.currentSampleIndex;
        const frameKey = String(sourceFrameIndex);
        if (this.workingPoints[frameKey]) {
            delete this.workingPoints[frameKey];
            this.render();
        }
    }

    clearAll() {
        this.workingPoints = {};
        this.render();
    }

    async syncStrideFromNode() {
        const state = this.node?._rhVoid;
        if (!state?.syncPreviewStride) {
            return;
        }

        const preserveFrame = this.sourceFrameIndices[this.currentSampleIndex] ?? this.currentSampleIndex;
        const changed = await state.syncPreviewStride(preserveFrame);
        if (!changed) {
            return;
        }

        this.previewFrames = state.previewFrames;
        this.sourceFrameIndices = state.sourceFrameIndices?.length
            ? state.sourceFrameIndices
            : this.previewFrames.map((_, index) => index);
        this.frameImages = new Map();
        this.currentSampleIndex = findClosestSampleIndex(this.sourceFrameIndices, preserveFrame);
        this.slider.max = String(Math.max(this.previewFrames.length - 1, 0));
        this.slider.value = String(this.currentSampleIndex);
    }

    async jumpFrame(delta) {
        await this.syncStrideFromNode();
        const nextFrame = Math.max(0, Math.min(this.previewFrames.length - 1, this.currentSampleIndex + delta));
        if (nextFrame !== this.currentSampleIndex) {
            this.currentSampleIndex = nextFrame;
            this.render();
        }
    }

    save() {
        if (!this.node?._rhVoid) {
            this.close();
            return;
        }

        const state = this.node._rhVoid;
        state.pointsByFrame = cloneFramePoints(this.workingPoints);
        state.confirmedPoints = cloneFramePoints(this.workingPoints);

        if (state.widgets.pointsStore) {
            state.widgets.pointsStore.value = stringifyFramePoints(state.pointsByFrame);
        }
        if (state.widgets.coordinates) {
            state.widgets.coordinates.value = stringifyFramePoints(state.confirmedPoints);
        }

        state.refreshSummary();
        this.node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        this.close();
    }
}

app.registerExtension({
    name: "RunningHub.Void.PointEditor",
    rh: {
        type: "nodes",
        nodes: [NODE_NAME],
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);

            const previewStrideWidget = this.widgets?.find((widget) => widget.name === "preview_stride");
            const uploadWidget = this.widgets?.find((widget) => widget.name === "upload");
            const uploadStatusWidget = this.widgets?.find((widget) => widget.name === "upload_status");
            const pointsStoreWidget = this.widgets?.find((widget) => widget.name === "points_store");
            const coordinatesWidget = this.widgets?.find((widget) => widget.name === "coordinates");

            if (previewStrideWidget && !previewStrideWidget.value) {
                previewStrideWidget.value = 1;
            }
            if (uploadStatusWidget && !uploadStatusWidget.value) {
                uploadStatusWidget.value = "idle";
            }
            if (pointsStoreWidget && !pointsStoreWidget.value) {
                pointsStoreWidget.value = "{}";
            }
            if (coordinatesWidget && !coordinatesWidget.value) {
                coordinatesWidget.value = "{}";
            }

            hideWidgetForGood(this, uploadWidget);
            hideWidgetForGood(this, uploadStatusWidget);
            hideWidgetForGood(this, pointsStoreWidget);
            hideWidgetForGood(this, coordinatesWidget);

            const wrapper = document.createElement("div");
            wrapper.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 8px;
                width: 100%;
                box-sizing: border-box;
                padding: 8px;
            `;

            const statusBar = document.createElement("div");
            statusBar.style.cssText = `
                min-height: ${STATUS_BAR_MIN_HEIGHT}px;
                padding: 8px 10px;
                border: 1px solid #353535;
                border-radius: 6px;
                background: #17191c;
                font-size: 12px;
                color: #c7c7c7;
                line-height: 1.45;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                overflow: auto;
            `;
            statusBar.innerHTML = buildStatusMarkup("Status: idle");

            const previewWrap = document.createElement("div");
            previewWrap.style.cssText = `
                width: 100%;
                height: ${MIN_PREVIEW_HEIGHT}px;
                background: #101214;
                border: 1px solid #333;
                border-radius: 6px;
                overflow: hidden;
                display: none;
                align-items: center;
                justify-content: center;
            `;

            const previewVideo = document.createElement("video");
            previewVideo.controls = false;
            previewVideo.loop = true;
            previewVideo.muted = true;
            previewVideo.autoplay = true;
            previewVideo.playsInline = true;
            previewVideo.style.cssText = `
                display: block;
                width: 100%;
                height: 100%;
                object-fit: contain;
                background: #000;
            `;

            const previewImage = document.createElement("img");
            previewImage.style.cssText = `
                display: none;
                width: 100%;
                height: 100%;
                object-fit: contain;
                background: #000;
            `;

            previewWrap.appendChild(previewVideo);
            previewWrap.appendChild(previewImage);

            const buttonRow = document.createElement("div");
            buttonRow.style.cssText = "display: flex; gap: 8px; flex-wrap: wrap;";

            const chooseButton = document.createElement("button");
            chooseButton.textContent = "Choose Video";
            chooseButton.style.cssText = `
                border: 1px solid #555;
                background: #2a2d31;
                color: #fff;
                border-radius: 6px;
                padding: 8px 10px;
                cursor: pointer;
                font-size: 12px;
                text-align: center;
            `;

            const openButton = document.createElement("button");
            openButton.textContent = "Open Editor";
            openButton.style.cssText = `
                border: 1px solid #555;
                background: #2a2d31;
                color: #fff;
                border-radius: 6px;
                padding: 8px 10px;
                cursor: pointer;
                font-size: 12px;
                text-align: center;
            `;

            buttonRow.appendChild(chooseButton);
            buttonRow.appendChild(openButton);
            wrapper.appendChild(statusBar);
            wrapper.appendChild(previewWrap);
            wrapper.appendChild(buttonRow);

            const domWidget = this.addDOMWidget("void_point_editor", "custom", wrapper, {
                serialize: false,
                hideOnZoom: false,
            });
            domWidget.computeSize = (width) => {
                const visible = previewWrap.style.display !== "none";
                const previewHeight = getPreviewHeight(width, this._rhVoid?.previewAspectRatio, visible);
                return [Math.max(width, MIN_NODE_WIDTH), Math.round(NODE_BASE_HEIGHT + previewHeight)];
            };

            const fileInput = document.createElement("input");
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                accept: VIDEO_ACCEPT,
                webkitdirectory: false,
            });
            document.body.append(fileInput);

            this._rhVoid = {
                widgets: {
                    previewStride: previewStrideWidget,
                    upload: uploadWidget,
                    uploadStatus: uploadStatusWidget,
                    pointsStore: pointsStoreWidget,
                    coordinates: coordinatesWidget,
                },
                fileInput,
                statusBar,
                chooseButton,
                openButton,
                previewWrap,
                previewVideo,
                previewImage,
                previewAspectRatio: null,
                previewFrames: [],
                sourceFrameIndices: [],
                pointsByFrame: parseFramePoints(pointsStoreWidget?.value),
                confirmedPoints: parseFramePoints(coordinatesWidget?.value),
                lastFrame: 0,
                frameStride: 1,
                totalFrames: 0,
                statusLabel: "idle",
                statusTone: "neutral",
                getRequestedStride: () => {
                    const state = this._rhVoid;
                    return parsePositiveInt(state.widgets.previewStride?.value, 1);
                },
                reloadPreviewFrames: async (preserveSourceFrame = 0) => {
                    const state = this._rhVoid;
                    const filename = state.widgets.upload?.value;
                    if (!filename) {
                        return false;
                    }

                    const previewPayload = await fetchPreviewFrames(filename, state.getRequestedStride());
                    state.previewFrames = Array.isArray(previewPayload.frames) ? previewPayload.frames : [];
                    state.sourceFrameIndices = Array.isArray(previewPayload.source_frame_indices)
                        ? previewPayload.source_frame_indices.map((value) => Number.parseInt(value, 10) || 0)
                        : state.previewFrames.map((_, index) => index);
                    state.totalFrames = Number.parseInt(previewPayload.total_frames, 10) || state.previewFrames.length;
                    state.frameStride = parsePositiveInt(previewPayload.frame_stride, state.getRequestedStride());
                    state.lastFrame = findClosestSampleIndex(state.sourceFrameIndices, preserveSourceFrame);
                    return true;
                },
                syncPreviewStride: async (preserveSourceFrame = null) => {
                    const state = this._rhVoid;
                    const requestedStride = state.getRequestedStride();
                    const targetFrame = preserveSourceFrame ?? (state.sourceFrameIndices[state.lastFrame] ?? 0);
                    if (requestedStride === state.frameStride && state.previewFrames?.length) {
                        return false;
                    }
                    await state.reloadPreviewFrames(targetFrame);
                    return true;
                },
                setBusy: (busy, label = "Choose Video") => {
                    const state = this._rhVoid;
                    state.chooseButton.disabled = busy;
                    state.chooseButton.textContent = busy ? label : "Choose Video";
                    state.chooseButton.style.opacity = busy ? "0.6" : "1";
                    state.chooseButton.style.cursor = busy ? "not-allowed" : "pointer";
                    state.openButton.disabled = busy || !(state.previewFrames?.length);
                    state.openButton.style.opacity = state.openButton.disabled ? "0.6" : "1";
                    state.openButton.style.cursor = state.openButton.disabled ? "not-allowed" : "pointer";
                },
                refreshLayout: () => {
                    const state = this._rhVoid;
                    const width = Math.max(this.size?.[0] || MIN_NODE_WIDTH, MIN_NODE_WIDTH);
                    const visible = state.previewWrap.style.display !== "none";
                    const previewHeight = getPreviewHeight(width, state.previewAspectRatio, visible);
                    state.previewWrap.style.height = visible ? `${Math.round(previewHeight)}px` : "0px";
                    this.setSize([
                        width,
                        Math.max(MIN_NODE_HEIGHT, Math.round(NODE_BASE_HEIGHT + previewHeight)),
                    ]);
                    this.setDirtyCanvas?.(true, true);
                    app.graph?.setDirtyCanvas?.(true, true);
                },
                updatePreview: (filename) => {
                    const state = this._rhVoid;
                    if (!filename) {
                        state.previewVideo.pause();
                        state.previewVideo.removeAttribute("src");
                        state.previewVideo.load();
                        state.previewImage.removeAttribute("src");
                        state.previewImage.style.display = "none";
                        state.previewVideo.style.display = "none";
                        state.previewWrap.style.display = "none";
                        state.previewAspectRatio = null;
                        state.refreshLayout();
                        return;
                    }

                    const params = getPreviewParams(filename);
                    const src = api.apiURL(`/view?${new URLSearchParams({
                        filename: params.filename,
                        type: "input",
                        rand: String(Date.now()),
                    })}`);

                    state.previewWrap.style.display = "flex";
                    if (params.isImagePreview) {
                        state.previewVideo.pause();
                        state.previewVideo.removeAttribute("src");
                        state.previewVideo.load();
                        state.previewVideo.style.display = "none";
                        state.previewImage.src = src;
                        state.previewImage.style.display = "block";
                    } else {
                        state.previewImage.removeAttribute("src");
                        state.previewImage.style.display = "none";
                        state.previewVideo.src = src;
                        state.previewVideo.style.display = "block";
                        state.previewVideo.play().catch(() => {});
                    }
                    state.refreshLayout();
                },
                setStatusText: (text, tone = "neutral") => {
                    const state = this._rhVoid;
                    const tones = {
                        neutral: { bg: "#17191c", border: "#353535", color: "#c7c7c7", label: "#93a3b8", value: "#e4e8ee" },
                        loading: { bg: "#1a1f29", border: "#35507a", color: "#d8e7ff", label: "#8db8ff", value: "#eef5ff" },
                        success: { bg: "#152117", border: "#2f6e39", color: "#d8ffd8", label: "#7fd58c", value: "#f1fff1" },
                        error: { bg: "#261717", border: "#7a3535", color: "#ffd8d8", label: "#ff9f9f", value: "#fff0f0" },
                    };
                    const style = tones[tone] || tones.neutral;
                    state.statusBar.innerHTML = buildStatusMarkup(text);
                    state.statusBar.style.background = style.bg;
                    state.statusBar.style.borderColor = style.border;
                    state.statusBar.style.color = style.color;
                    state.statusBar.style.setProperty("--rh-void-status-label", style.label);
                    state.statusBar.style.setProperty("--rh-void-status-value", style.value);
                },
                renderStatusBar: (statusLabel = null, tone = "neutral") => {
                    const state = this._rhVoid;
                    state.statusLabel = statusLabel ?? state.statusLabel ?? "idle";
                    state.statusTone = tone ?? state.statusTone ?? "neutral";
                    const lines = [];
                    lines.push(`Status: ${state.statusLabel}`);
                    lines.push(`Video: ${state.widgets.upload?.value || "none"}`);
                    lines.push(`Preview frames: ${state.previewFrames?.length || 0}`);
                    lines.push(`Sample stride: ${state.frameStride || 1}`);
                    lines.push(`Annotated frames: ${Object.keys(state.confirmedPoints || {}).length}`);
                    lines.push(`Points: ${countPoints(state.confirmedPoints)}`);
                    lines.push(`Frames: ${summarizeFrames(state.confirmedPoints)}`);
                    if (state.totalFrames) {
                        lines.push(`Total source frames: ${state.totalFrames}`);
                    }
                    state.setStatusText(lines.join("\n"), state.statusTone);
                },
                refreshSummary: () => {
                    const state = this._rhVoid;
                    const previewCount = state.previewFrames?.length || 0;
                    state.openButton.disabled = state.chooseButton.disabled || !previewCount;
                    state.openButton.style.opacity = state.openButton.disabled ? "0.6" : "1";
                    state.openButton.style.cursor = state.openButton.disabled ? "not-allowed" : "pointer";
                    state.renderStatusBar(state.statusLabel, state.statusTone);
                },
                resetPoints: () => {
                    const state = this._rhVoid;
                    state.pointsByFrame = {};
                    state.confirmedPoints = {};
                    state.lastFrame = 0;
                    state.previewFrames = [];
                    state.sourceFrameIndices = [];
                    state.totalFrames = 0;
                    state.frameStride = state.getRequestedStride();
                    state.statusLabel = "idle";
                    state.statusTone = "neutral";
                    state.updatePreview("");
                    if (state.widgets.pointsStore) {
                        state.widgets.pointsStore.value = "{}";
                    }
                    if (state.widgets.coordinates) {
                        state.widgets.coordinates.value = "{}";
                    }
                },
            };

            previewVideo.addEventListener("loadedmetadata", () => {
                if (!this._rhVoid) {
                    return;
                }
                this._rhVoid.previewAspectRatio = previewVideo.videoWidth > 0 && previewVideo.videoHeight > 0
                    ? previewVideo.videoWidth / previewVideo.videoHeight
                    : null;
                this._rhVoid.refreshLayout();
            });

            previewImage.addEventListener("load", () => {
                if (!this._rhVoid) {
                    return;
                }
                this._rhVoid.previewAspectRatio = previewImage.naturalWidth > 0 && previewImage.naturalHeight > 0
                    ? previewImage.naturalWidth / previewImage.naturalHeight
                    : null;
                this._rhVoid.refreshLayout();
            });

            previewVideo.addEventListener("error", () => {
                this._rhVoid?.refreshLayout();
            });

            previewImage.addEventListener("error", () => {
                this._rhVoid?.refreshLayout();
            });

            fileInput.onchange = async () => {
                const file = fileInput.files?.[0];
                if (!file) {
                    return;
                }

                try {
                    this._rhVoid.resetPoints();
                    uploadStatusWidget.value = "uploading...";
                    this._rhVoid.setBusy(true, "Uploading...");
                    this._rhVoid.renderStatusBar("uploading...", "loading");
                    this._rhVoid.refreshSummary();
                    const uploadedName = await uploadVideoFile(file);
                    uploadWidget.value = uploadedName;
                    this._rhVoid.updatePreview(uploadedName);
                    uploadStatusWidget.value = "processing...";
                    this._rhVoid.setBusy(true, "Processing...");
                    this._rhVoid.renderStatusBar("processing...", "loading");
                    this._rhVoid.refreshSummary();
                    await this._rhVoid.reloadPreviewFrames(0);
                    uploadStatusWidget.value = "success";
                    this._rhVoid.renderStatusBar("success", "success");
                } catch (error) {
                    console.error(error);
                    uploadWidget.value = "";
                    uploadStatusWidget.value = "failed";
                    this._rhVoid.resetPoints();
                    this._rhVoid.renderStatusBar(`failed (${error?.message || "please try again"})`, "error");
                } finally {
                    this._rhVoid.setBusy(false);
                    this._rhVoid.refreshSummary();
                    this.setDirtyCanvas?.(true, true);
                    app.graph?.setDirtyCanvas?.(true, true);
                    fileInput.value = "";
                }
            };

            chooseButton.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                app.canvas.node_widget = null;
                fileInput.click();
            });

            openButton.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                VoidEditorDialog.getInstance().show(this);
            });

            this.setSize([
                Math.max(this.size?.[0] || MIN_NODE_WIDTH, MIN_NODE_WIDTH),
                Math.max(this.size?.[1] || MIN_NODE_HEIGHT, MIN_NODE_HEIGHT),
            ]);

            this._rhVoid.setBusy(false);
            this._rhVoid.renderStatusBar("idle", "neutral");
            this._rhVoid.refreshSummary();
            this._rhVoid.refreshLayout();
            if (uploadWidget?.value && uploadStatusWidget?.value === "success") {
                this._rhVoid.updatePreview(uploadWidget.value);
            }
            return result;
        };

        addMenuHandler(nodeType, function (_, options) {
            if (!Array.isArray(options)) {
                return;
            }
            options.unshift({
                content: "Edit Image For Void",
                callback: () => VoidEditorDialog.getInstance().show(this),
            });
        });

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            if (this._rhVoid) {
                this._rhVoid.pointsByFrame = parseFramePoints(this._rhVoid.widgets.pointsStore?.value);
                this._rhVoid.confirmedPoints = parseFramePoints(this._rhVoid.widgets.coordinates?.value);
                this._rhVoid.setBusy(false);
                if (this._rhVoid.widgets.upload?.value && this._rhVoid.widgets.uploadStatus?.value === "success") {
                    this._rhVoid.updatePreview(this._rhVoid.widgets.upload.value);
                } else {
                    this._rhVoid.updatePreview("");
                }
                this._rhVoid.refreshSummary();
                this._rhVoid.refreshLayout();
            }
            return result;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted?.apply(this, arguments);
            if (!this._rhVoid) {
                return result;
            }

            if (message?.confirmed_coordinates?.[0]) {
                this._rhVoid.confirmedPoints = parseFramePoints(message.confirmed_coordinates[0]);
                this._rhVoid.pointsByFrame = parseFramePoints(this._rhVoid.widgets.pointsStore?.value || message.confirmed_coordinates[0]);
            }
            this._rhVoid.refreshSummary();
            return result;
        };

        const originalOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._rhVoid?.fileInput?.remove();
            return originalOnRemoved?.apply(this, arguments);
        };
    },
});
