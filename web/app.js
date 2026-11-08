import { Muxer, ArrayBufferTarget } from "https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.2/+esm";
import { BottleTracker, Detector, annotate, drawScene } from "./detect.js";

const detector = new Detector();
const tracker = new BottleTracker();

const liveModeButton = document.querySelector("#live-mode");
const mediaModeButton = document.querySelector("#media-mode");
const livePanel = document.querySelector("#live-panel");
const mediaPanel = document.querySelector("#media-panel");
const confidenceInput = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidence-value");
const statusLine = document.querySelector("#status");
const progressWrap = document.querySelector("#progress");
const progressBar = document.querySelector("#progress-bar");
const camera = document.querySelector("#camera");
const liveCanvas = document.querySelector("#live-canvas");
const liveContext = liveCanvas.getContext("2d");
const startButton = document.querySelector("#start-camera");
const flipButton = document.querySelector("#flip-camera");
const resetButton = document.querySelector("#reset-count");
const liveTotal = document.querySelector("#live-total");
const liveNow = document.querySelector("#live-now");
const fileInput = document.querySelector("#file-input");
const resultStats = document.querySelector("#result-stats");
const mediaCount = document.querySelector("#media-count");
const resultStage = document.querySelector("#result-stage");
const resultCanvas = document.querySelector("#result-canvas");
const resultContext = resultCanvas.getContext("2d");
const resultVideo = document.querySelector("#result-video");
const downloadLink = document.querySelector("#download");

let mode = null;
let facingMode = /Android|iPhone|iPad/i.test(navigator.userAgent) ? "environment" : "user";
let stream = null;
let liveTimer = 0;
let detectingLive = false;
let mediaToken = 0;
let downloadUrl = "";
let sourceUrl = "";

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = Number(confidenceInput.value).toFixed(2);
});

liveModeButton.addEventListener("click", () => setMode("live"));
mediaModeButton.addEventListener("click", () => {
  setMode("media");
  if (detector.session) fileInput.click();
});
startButton.addEventListener("click", () => {
  if (stream) stopCamera();
  else startCamera();
});
flipButton.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  if (stream) {
    stopCamera();
    await startCamera();
  }
});
resetButton.addEventListener("click", () => {
  tracker.reset();
  liveTotal.textContent = "0";
  liveNow.textContent = "0";
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (file) inspectFile(file);
});

window.addEventListener("pagehide", releaseSession);

loadDetector();

function setMode(nextMode) {
  mode = nextMode;
  liveModeButton.classList.toggle("active", mode === "live");
  mediaModeButton.classList.toggle("active", mode === "media");
  livePanel.hidden = mode !== "live";
  mediaPanel.hidden = mode !== "media";
  if (mode !== "live") stopCamera();
  if (mode !== "media") {
    mediaToken += 1;
    releaseResult();
    fileInput.value = "";
  }
  setStatus(mode === "live"
    ? "Start the camera. The count stays on this screen until you reset it or leave."
    : "Choose one photo or video. Detection starts as soon as it opens.");
}

async function loadDetector() {
  setStatus("Loading the detector for this visit…");
  showProgress(0);
  try {
    await detector.load((fraction) => showProgress(fraction || 0.05));
    hideProgress();
    setStatus("Choose live detection or a photo or video.");
  } catch (error) {
    hideProgress();
    setStatus(error.message || "The detector could not load.");
  }
}

async function startCamera() {
  if (!detector.session) {
    setStatus("The detector is still loading.");
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
  } catch (error) {
    setStatus("The browser could not open a camera. Allow camera access and use HTTPS or localhost.");
    return;
  }
  camera.srcObject = stream;
  await camera.play();
  startButton.textContent = "Stop camera";
  tracker.reset();
  scheduleLiveFrame();
}

function stopCamera() {
  window.clearTimeout(liveTimer);
  detectingLive = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  stream = null;
  camera.srcObject = null;
  startButton.textContent = "Start camera";
}

function scheduleLiveFrame() {
  window.clearTimeout(liveTimer);
  liveTimer = window.setTimeout(runLiveFrame, 60);
}

async function runLiveFrame() {
  if (!stream || detectingLive) return;
  if (camera.readyState < 2) {
    scheduleLiveFrame();
    return;
  }
  detectingLive = true;
  try {
    const boxes = await detector.detect(camera, currentConfidence());
    const tracks = tracker.update(boxes);
    drawScene(liveContext, camera, tracks, tracker.seen.size, tracks.length);
    liveTotal.textContent = String(tracker.seen.size);
    liveNow.textContent = String(tracks.length);
  } catch (error) {
    setStatus(error.message || "Live detection stopped.");
    stopCamera();
    return;
  } finally {
    detectingLive = false;
  }
  if (stream) scheduleLiveFrame();
}

async function inspectFile(file) {
  const token = ++mediaToken;
  releaseResult();
  const isImage = file.type.startsWith("image/");
  const isVideo = file.type.startsWith("video/");
  if (!isImage && !isVideo) {
    setStatus("Choose a photo or a video.");
    return;
  }
  if (!detector.session) {
    setStatus("The detector is still loading. Choose the file again in a moment.");
    return;
  }
  sourceUrl = URL.createObjectURL(file);
  try {
    if (isImage) await inspectImage(sourceUrl, file.name, token);
    else await inspectVideo(sourceUrl, file.name, token);
  } catch (error) {
    if (token === mediaToken) setStatus(error.message || "That file could not be read.");
  }
}

async function inspectImage(url, name, token) {
  setStatus("Detecting bottles…");
  const image = await loadImage(url);
  if (token !== mediaToken) return;
  const boxes = await detector.detect(image, currentConfidence());
  drawScene(resultContext, image, boxes, boxes.length, null);
  showImageResult(boxes.length);
  const blob = await new Promise((resolve) => resultCanvas.toBlob(resolve, "image/jpeg", 0.92));
  offerDownload(blob, `detected-${safeName(name)}.jpg`);
  setStatus(`${boxes.length} bottle${boxes.length === 1 ? "" : "s"} in this photo. The download stays on your device.`);
}

async function inspectVideo(url, name, token) {
  const video = document.createElement("video");
  video.src = url;
  video.muted = true;
  video.playsInline = true;
  await waitFor(video, "loadedmetadata");
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  if (!duration) throw new Error("That video has no readable frames.");
  const sampleFps = 5;
  const frameCount = Math.max(1, Math.min(240, Math.round(duration * sampleFps)));
  const step = duration / frameCount;
  const outputWidth = even(Math.min(video.videoWidth || 640, 960));
  const outputHeight = even(Math.round(outputWidth * ((video.videoHeight || 360) / (video.videoWidth || 640))));
  const scratch = document.createElement("canvas");
  scratch.width = outputWidth;
  scratch.height = outputHeight;
  const scratchContext = scratch.getContext("2d");
  tracker.reset();
  const encoder = createEncoder(outputWidth, outputHeight, sampleFps);
  showProgress(0);
  setStatus("Detecting bottles in the video…");

  for (let index = 0; index < frameCount; index += 1) {
    if (token !== mediaToken) {
      encoder.cancel();
      return;
    }
    await seek(video, Math.min(duration - 0.05, index * step));
    scratchContext.drawImage(video, 0, 0, outputWidth, outputHeight);
    const boxes = await detector.detect(scratch, currentConfidence());
    const tracks = tracker.update(boxes);
    annotate(scratchContext, tracks, tracker.seen.size, tracks.length);
    await encoder.add(scratch, index);
    showProgress((index + 1) / frameCount);
  }

  const blob = await encoder.finish();
  if (token !== mediaToken) return;
  hideProgress();
  const count = tracker.seen.size;
  showVideoResult(count);
  offerDownload(blob, `detected-${safeName(name)}.mp4`);
  setStatus(`${count} bottle${count === 1 ? "" : "s"} tracked in this video. Download saves the annotated file on your device. This site does not keep it.`);
}

function createEncoder(width, height, fps) {
  if (typeof VideoEncoder === "undefined") {
    throw new Error("This browser cannot build a downloadable video. The count can still be shown on Chrome, Edge, or Safari.");
  }
  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: "avc", width, height },
    fastStart: "in-memory",
  });
  let failed = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (error) => { failed = error; },
  });
  encoder.configure({
    codec: "avc1.42001f",
    width,
    height,
    bitrate: 1_800_000,
    framerate: fps,
    avc: { format: "avc" },
  });
  const frameDuration = Math.round(1_000_000 / fps);
  return {
    async add(canvas, index) {
      if (failed) throw failed;
      if (encoder.encodeQueueSize > 4) {
        await new Promise((resolve) => encoder.addEventListener("dequeue", resolve, { once: true }));
      }
      const frame = new VideoFrame(canvas, { timestamp: index * frameDuration, duration: frameDuration });
      encoder.encode(frame, { keyFrame: index % 15 === 0 });
      frame.close();
    },
    cancel() {
      if (encoder.state !== "closed") encoder.close();
    },
    async finish() {
      await encoder.flush();
      if (failed) throw failed;
      muxer.finalize();
      encoder.close();
      return new Blob([target.buffer], { type: "video/mp4" });
    },
  };
}

function showImageResult(count) {
  resultStats.hidden = false;
  mediaCount.textContent = String(count);
  resultStage.hidden = false;
  resultCanvas.hidden = false;
  resultVideo.hidden = true;
  resultVideo.removeAttribute("src");
}

function showVideoResult(count) {
  resultStats.hidden = false;
  mediaCount.textContent = String(count);
  resultStage.hidden = false;
  resultCanvas.hidden = true;
  resultVideo.hidden = false;
}

function offerDownload(blob, filename) {
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(blob);
  downloadLink.href = downloadUrl;
  downloadLink.download = filename;
  downloadLink.textContent = filename.endsWith(".mp4") ? "Download detected video" : "Download detected photo";
  downloadLink.hidden = false;
  if (!resultVideo.hidden) resultVideo.src = downloadUrl;
}

function releaseResult() {
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  downloadUrl = "";
  sourceUrl = "";
  downloadLink.hidden = true;
  downloadLink.removeAttribute("href");
  resultVideo.removeAttribute("src");
  resultVideo.hidden = true;
  resultCanvas.hidden = true;
  resultStage.hidden = true;
  resultStats.hidden = true;
  hideProgress();
}

function releaseSession() {
  mediaToken += 1;
  stopCamera();
  releaseResult();
  fileInput.value = "";
}

function currentConfidence() {
  return Number(confidenceInput.value);
}

function setStatus(text) {
  statusLine.textContent = text;
}

function showProgress(fraction) {
  progressWrap.hidden = false;
  progressBar.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
}

function hideProgress() {
  progressWrap.hidden = true;
  progressBar.style.width = "0";
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("That photo could not be opened."));
    image.src = url;
  });
}

function waitFor(element, eventName) {
  return new Promise((resolve, reject) => {
    const ok = () => {
      element.removeEventListener(eventName, ok);
      element.removeEventListener("error", fail);
      resolve();
    };
    const fail = () => {
      element.removeEventListener(eventName, ok);
      element.removeEventListener("error", fail);
      reject(new Error("That video could not be opened."));
    };
    element.addEventListener(eventName, ok);
    element.addEventListener("error", fail);
  });
}

function seek(video, time) {
  return new Promise((resolve) => {
    const finish = () => {
      video.removeEventListener("seeked", finish);
      resolve();
    };
    video.addEventListener("seeked", finish);
    if (Math.abs(video.currentTime - time) < 0.001) {
      finish();
      return;
    }
    video.currentTime = Math.max(0, time);
  });
}

function even(value) {
  const rounded = Math.max(2, Math.round(value));
  return rounded % 2 === 0 ? rounded : rounded - 1;
}

function safeName(name) {
  const stem = name.replace(/\.[^.]+$/, "");
  const cleaned = stem.replace(/[^a-z0-9-_]+/gi, "_").slice(0, 40);
  return cleaned || "bottle";
}
