const INPUT_SIZE = 640;

export class BottleTracker {
  constructor() {
    this.reset();
  }

  reset() {
    this.nextId = 1;
    this.tracks = [];
    this.seen = new Set();
  }

  update(boxes) {
    const assigned = new Set();
    const matched = [];

    for (const box of boxes) {
      let bestIndex = -1;
      let bestScore = 0.35;
      for (let index = 0; index < this.tracks.length; index += 1) {
        if (assigned.has(index)) continue;
        const score = iou(this.tracks[index], box);
        if (score > bestScore) {
          bestScore = score;
          bestIndex = index;
        }
      }

      if (bestIndex >= 0) {
        assigned.add(bestIndex);
        const track = this.tracks[bestIndex];
        Object.assign(track, box, { missed: 0 });
        matched.push(track);
      } else {
        const track = { ...box, id: this.nextId, missed: 0 };
        this.nextId += 1;
        this.tracks.push(track);
        matched.push(track);
      }
    }

    for (let index = 0; index < this.tracks.length; index += 1) {
      const track = this.tracks[index];
      if (!assigned.has(index) && !matched.includes(track)) {
        track.missed += 1;
      }
    }
    this.tracks = this.tracks.filter((track) => track.missed < 6);
    for (const track of matched) this.seen.add(track.id);
    return matched;
  }
}

export class Detector {
  constructor() {
    this.session = null;
    this.inputName = "images";
    this.outputName = "output0";
  }

  async load(onProgress) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
    const response = await fetch("models/best.onnx?v=2");
    if (!response.ok) throw new Error("The detector model could not be downloaded.");
    const total = Number(response.headers.get("content-length")) || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (onProgress) onProgress(total ? received / total : 0);
    }
    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    this.session = await ort.InferenceSession.create(buffer.buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
  }

  async detect(source, confidence) {
    if (!this.session) throw new Error("The detector is still loading.");
    const prepared = letterbox(source, INPUT_SIZE);
    const tensor = new ort.Tensor("float32", prepared.input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const outputs = await this.session.run({ [this.inputName]: tensor });
    const raw = outputs[this.outputName];
    return decode(raw, confidence, prepared, source.width || source.videoWidth, source.height || source.videoHeight);
  }
}

export function drawScene(ctx, source, boxes, total, inFrame) {
  const width = source.videoWidth || source.width;
  const height = source.videoHeight || source.height;
  ctx.canvas.width = width;
  ctx.canvas.height = height;
  ctx.drawImage(source, 0, 0, width, height);
  annotate(ctx, boxes, total, inFrame);
}

export function annotate(ctx, boxes, total, inFrame) {
  const width = ctx.canvas.width;
  ctx.lineWidth = Math.max(2, Math.round(width / 320));
  ctx.font = `${Math.max(14, Math.round(width / 46))}px Segoe UI, sans-serif`;
  for (const box of boxes) {
    ctx.strokeStyle = "#39d98a";
    ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    const label = `${Math.round(box.score * 100)}%`;
    const textWidth = ctx.measureText(label).width;
    const top = Math.max(box.y1 - 22, 0);
    ctx.fillStyle = "#10231b";
    ctx.fillRect(box.x1, top, textWidth + 12, 22);
    ctx.fillStyle = "#f4fff8";
    ctx.fillText(label, box.x1 + 6, top + 16);
  }
  drawBadge(ctx, total, inFrame);
}

function drawBadge(ctx, total, inFrame) {
  const lines = [`Detected: ${total}`];
  if (inFrame !== undefined && inFrame !== null) lines.push(`In frame: ${inFrame}`);
  ctx.save();
  ctx.font = "600 16px Segoe UI, sans-serif";
  const width = Math.max(...lines.map((line) => ctx.measureText(line).width)) + 28;
  const height = 18 + lines.length * 22;
  const x = Math.max(ctx.canvas.width - width - 12, 8);
  const y = 12;
  ctx.fillStyle = "rgba(16, 35, 27, 0.82)";
  roundRect(ctx, x, y, width, height, 10);
  ctx.fill();
  ctx.strokeStyle = "#8ee0b5";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = "#f7fffb";
  lines.forEach((line, index) => ctx.fillText(line, x + 14, y + 24 + index * 22));
  ctx.restore();
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

function letterbox(source, size) {
  const width = source.videoWidth || source.width;
  const height = source.videoHeight || source.height;
  const scale = Math.min(size / width, size / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const dx = Math.floor((size - resizedWidth) / 2);
  const dy = Math.floor((size - resizedHeight) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, dx, dy, resizedWidth, resizedHeight);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const input = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let index = 0; index < plane; index += 1) {
    input[index] = pixels[index * 4] / 255;
    input[plane + index] = pixels[index * 4 + 1] / 255;
    input[plane * 2 + index] = pixels[index * 4 + 2] / 255;
  }
  return { input, scale, dx, dy };
}

function decode(output, confidence, prepared, imageWidth, imageHeight) {
  const data = output.data;
  const boxesPerRow = output.dims[2];
  const candidates = [];
  for (let index = 0; index < boxesPerRow; index += 1) {
    const score = data[4 * boxesPerRow + index];
    if (score < confidence) continue;
    const centerX = data[index];
    const centerY = data[boxesPerRow + index];
    const boxWidth = data[2 * boxesPerRow + index];
    const boxHeight = data[3 * boxesPerRow + index];
    const box = {
      x1: clamp((centerX - boxWidth / 2 - prepared.dx) / prepared.scale, 0, imageWidth),
      y1: clamp((centerY - boxHeight / 2 - prepared.dy) / prepared.scale, 0, imageHeight),
      x2: clamp((centerX + boxWidth / 2 - prepared.dx) / prepared.scale, 0, imageWidth),
      y2: clamp((centerY + boxHeight / 2 - prepared.dy) / prepared.scale, 0, imageHeight),
      score,
    };
    if (box.x2 - box.x1 > 2 && box.y2 - box.y1 > 2) candidates.push(box);
  }
  return nms(candidates, 0.45);
}

function nms(boxes, threshold) {
  const ordered = [...boxes].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of ordered) {
    if (kept.some((other) => iou(box, other) > threshold)) continue;
    kept.push(box);
  }
  return kept;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
