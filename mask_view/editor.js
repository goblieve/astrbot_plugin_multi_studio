(() => {
  const token = location.hash.slice(1);
  const byId = (id) => document.getElementById(id);
  const overlay = byId("overlay");
  const display = overlay.getContext("2d");
  const selection = document.createElement("canvas");
  const paint = selection.getContext("2d");
  const status = byId("status");
  const edit = byId("edit");
  const originalUpload = byId("original-upload");
  const originalProgress = byId("original-progress");
  const originalInput = byId("original-file");
  const originalDrop = byId("original-drop");
  const replaceOriginalButton = byId("replace-original");
  const restoreOriginalButton = byId("restore-original");
  let originalFile = null;
  let originalRevision = 0;
  let replacingOriginal = false;
  const brushSize = byId("size");
  const brushPreview = byId("brush-preview");
  const fillHoles = byId("fill-holes");
  const referenceInput = byId("reference");
  const referencePreview = byId("reference-preview");
  const referenceDrop = byId("reference-drop");
  let referenceFile = null;
  let referenceUrl = "";
  let maxReferenceBytes = 0;
  let maxOriginalBytes = 0;
  let originalChunkBytes = 0;
  let originalChunkParallelism = 1;
  let originalCacheEnabled = false;
  let originalUploadId = "";
  let strokes = [];
  let initialMask = null;
  let maskHistory = [];
  let current = null;
  let erase = false;
  let panMode = false;
  let panDrag = null;
  const activeTouches = new Map();
  let pinchState = null;
  let zoomPercent = 100;
  let busy = false;
  let finished = false;
  let alternateModes = false;
  let fill = null;
  const edgeCanvas = document.createElement("canvas");
  const edgeCtx = edgeCanvas.getContext("2d");

  function show(message, error = false) {
    status.textContent = message;
    status.dataset.error = String(error);
  }

  function lock(value) {
    busy = value;
    for (const id of ["auto", "auto-append", "submit", "clear", "undo", "invert", "pan-tool", "fill-holes",
      "reference", "reference-drop", "reference-clear", "replace-original", "grow", "feather"]) {
      const node = byId(id);
      if (node) node.disabled = value || finished;
    }
    if (busy || finished) {
      brushPreview.hidden = true;
      delete referenceDrop.dataset.drag;
    }
  }

  async function api(path, options = {}) {
    const response = await fetch("/api/" + path, {
      ...options, headers: {Authorization: "Bearer " + token, ...options.headers},
      cache: "no-store", referrerPolicy: "no-referrer",
    });
    if (!response.ok) {
      const message = response.status >= 500 ? "连接服务器失败（HTTP " + response.status + "）。"
        : (await response.text() || "当前操作失败");
      throw Object.assign(new Error(message), {status: response.status});
    }
    return response;
  }

  function originalHeaders(revision = originalRevision) {
    return revision > 0 ? {"X-Original-Revision": String(revision)} : {};
  }

  function render() {
    display.clearRect(0, 0, overlay.width, overlay.height);
    drawMask(display);
    display.globalCompositeOperation = "source-in";
    display.fillStyle = "rgba(255,52,70,0.55)";
    display.fillRect(0, 0, overlay.width, overlay.height);
    display.globalCompositeOperation = "source-over";
    const grow = Number(byId("grow")?.value) || 0;
    const feather = Number(byId("feather")?.value) || 0;
    if ((grow > 0 || feather > 0) && !current && (initialMask || strokes.length)) {
      if (edgeCanvas.width !== overlay.width || edgeCanvas.height !== overlay.height) {
        edgeCanvas.width = overlay.width;
        edgeCanvas.height = overlay.height;
      } else {
        edgeCtx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);
      }
      drawMask(edgeCtx);
      edgeCtx.globalCompositeOperation = "source-in";
      edgeCtx.fillStyle = "rgba(255,156,52,0.42)";
      edgeCtx.fillRect(0, 0, edgeCanvas.width, edgeCanvas.height);
      edgeCtx.globalCompositeOperation = "source-over";
      const radius = Math.max(1, Math.min(24, Math.round(Math.max(0, grow) + feather * 0.6)));
      display.save();
      display.globalCompositeOperation = "destination-over";
      for (const [dx, dy] of [
        [radius, 0], [-radius, 0], [0, radius], [0, -radius],
        [Math.round(radius * 0.7), Math.round(radius * 0.7)],
        [-Math.round(radius * 0.7), Math.round(radius * 0.7)],
        [Math.round(radius * 0.7), -Math.round(radius * 0.7)],
        [-Math.round(radius * 0.7), -Math.round(radius * 0.7)],
      ]) {
        display.drawImage(edgeCanvas, dx, dy);
      }
      display.restore();
    }
  }

  function drawMask(context) {
    context.drawImage(selection, 0, 0);
    if (fill) context.drawImage(fill.canvas, fill.x, fill.y);
  }

  function paintPart(context, stroke, previous, currentPoint) {
    context.globalCompositeOperation = stroke.erase ? "destination-out" : "source-over";
    context.lineWidth = stroke.size;
    context.lineCap = "round";
    context.strokeStyle = "#fff";
    context.fillStyle = "#fff";
    context.beginPath();
    if (previous) {
      context.moveTo(previous[0], previous[1]);
      context.lineTo(currentPoint[0], currentPoint[1]);
      context.stroke();
    } else {
      context.arc(currentPoint[0], currentPoint[1], stroke.size / 2, 0, Math.PI * 2);
      context.fill();
    }
    context.globalCompositeOperation = "source-over";
  }

  function paintStroke(context, stroke) {
    stroke.points.forEach((point, index) => paintPart(context, stroke, stroke.points[index - 1], point));
  }

  function paintOnSelection(stroke, previous, point) {
    paintPart(paint, stroke, previous, point);
    if (stroke.erase && fill) paintPart(fill.context, stroke, previous, point);
  }

  function markExterior(frame) {
    const {data, width, height} = frame;
    for (let i = 3; i < data.length; i += 4) data[i] = data[i] >= 128 ? 255 : 0;
    const spans = [];
    const alpha = (x, y) => 4 * (y * width + x) + 3;
    function visit(x, y) {
      let left = x;
      let right = x;
      while (left > 0 && !data[alpha(left - 1, y)]) left--;
      while (right + 1 < width && !data[alpha(right + 1, y)]) right++;
      for (let i = alpha(left, y), end = alpha(right, y); i <= end; i += 4) data[i] = 1;
      spans.push(y, left, right);
      return right;
    }
    visit(0, 0);
    while (spans.length) {
      const right = spans.pop();
      const left = spans.pop();
      const y = spans.pop();
      for (const next of [y - 1, y + 1]) {
        if (next < 0 || next >= height) continue;
        for (let x = left; x <= right; x++) {
          if (data[alpha(x, next)] === 0) x = visit(x, next);
        }
      }
    }
  }

  function updateFill() {
    fill = null;
    if (!fillHoles.checked || !strokes.some(stroke => !stroke.erase)) return;
    let left = selection.width;
    let top = selection.height;
    let right = 0;
    let bottom = 0;
    for (const stroke of strokes) {
      if (stroke.erase) continue;
      const radius = stroke.size / 2 + 2;
      for (const [x, y] of stroke.points) {
        left = Math.min(left, Math.floor(x - radius));
        top = Math.min(top, Math.floor(y - radius));
        right = Math.max(right, Math.ceil(x + radius));
        bottom = Math.max(bottom, Math.ceil(y + radius));
      }
    }
    left = Math.max(-1, left);
    top = Math.max(-1, top);
    right = Math.min(selection.width + 1, right);
    bottom = Math.min(selection.height + 1, bottom);
    const canvas = document.createElement("canvas");
    canvas.width = right - left;
    canvas.height = bottom - top;
    const context = canvas.getContext("2d", {willReadFrequently: true});
    context.translate(-left, -top);
    // Keep an empty border around the bitmap so the outside is always reachable.
    context.beginPath();
    context.rect(left + 1, top + 1, canvas.width - 2, canvas.height - 2);
    context.clip();
    // Find closed brush outlines first. Eraser strokes cut the result afterwards.
    for (const stroke of strokes) {
      if (!stroke.erase) paintStroke(context, stroke);
    }
    const frame = context.getImageData(0, 0, canvas.width, canvas.height);
    markExterior(frame);
    const {data} = frame;
    let hasHoles = false;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] === 0) {
        data[i] = data[i + 1] = data[i + 2] = data[i + 3] = 255;
        hasHoles = true;
      } else {
        data[i + 3] = 0;
      }
    }
    if (!hasHoles) return;
    context.putImageData(frame, 0, 0);
    // The original selection already contains any ink repainted after an erasure.
    for (const stroke of strokes) {
      if (stroke.erase) paintStroke(context, stroke);
    }
    fill = {canvas, context, x: left, y: top};
  }

  function replay() {
    paint.clearRect(0, 0, selection.width, selection.height);
    if (initialMask) paint.drawImage(initialMask, 0, 0);
    for (const stroke of strokes) paintStroke(paint, stroke);
    updateFill();
    render();
  }
  fillHoles.addEventListener("change", () => {
    updateFill();
    render();
  });

  function showBrush(x, y, diameter = Number(brushSize.value), isEraser = erase) {
    brushPreview.style.left = x + "px";
    brushPreview.style.top = y + "px";
    brushPreview.style.width = brushPreview.style.height = diameter + "px";
    brushPreview.dataset.tool = isEraser ? "eraser" : "brush";
    brushPreview.hidden = false;
  }

  function previewSize() {
    const diameter = Number(brushSize.value);
    byId("size-value").textContent = diameter + " px";
    if (busy || finished) return;
    const rect = overlay.getBoundingClientRect();
    const viewport = byId("viewport").getBoundingClientRect();
    const left = Math.max(rect.left, viewport.left);
    const right = Math.min(rect.right, viewport.right);
    const top = Math.max(rect.top, 0);
    const bottom = Math.min(rect.bottom, window.innerHeight);
    const y = top + Math.min((bottom - top) / 2, diameter / 2 + 16);
    if (bottom > top && right > left) showBrush((left + right) / 2 - rect.left, y - rect.top, diameter);
  }

  brushSize.addEventListener("input", previewSize);
  brushSize.addEventListener("focus", previewSize);
  brushSize.addEventListener("blur", () => {
    if (!current) brushPreview.hidden = true;
  });

  function trackBrush(event, rect) {
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (busy || finished || x < 0 || y < 0 || x > rect.width || y > rect.height
        || (event.pointerType === "touch" && !current)) {
      brushPreview.hidden = true;
      return;
    }
    const diameter = current ? current.size * rect.width / overlay.width : Number(brushSize.value);
    showBrush(x, y, diameter, current?.erase ?? erase);
  }

  function position(event, rect) {
    return [
      (event.clientX - rect.left) * overlay.width / rect.width,
      (event.clientY - rect.top) * overlay.height / rect.height,
    ];
  }

  function snapshotMask() {
    const canvas = document.createElement("canvas");
    canvas.width = selection.width;
    canvas.height = selection.height;
    drawMask(canvas.getContext("2d"));
    return canvas;
  }

  function pushMaskHistory() {
    if (maskHistory.length >= 20) maskHistory.shift();
    maskHistory.push({
      initialMask,
      strokes: strokes.map(s => ({...s, points: s.points.slice()})),
    });
  }

  function setZoom(percent) {
    zoomPercent = Math.max(100, Math.min(350, Math.round(percent)));
    const stage = byId("stage");
    stage.style.width = zoomPercent + "%";
    const enlarged = zoomPercent > 100;
    byId("zoom").setAttribute("aria-pressed", String(enlarged));
    byId("zoom").textContent = enlarged ? `缩放 ${zoomPercent}%（点击还原）` : "放大预览";
    brushPreview.hidden = true;
  }

  overlay.addEventListener("pointerdown", (event) => {
    if (busy || finished) return;
    const viewport = byId("viewport");
    if (event.pointerType === "touch") {
      activeTouches.set(event.pointerId, {x: event.clientX, y: event.clientY});
      if (activeTouches.size >= 2) {
        if (current) {
          strokes.pop();
          current = null;
          replay();
        }
        const pts = [...activeTouches.values()];
        const dx = pts[0].x - pts[1].x;
        const dy = pts[0].y - pts[1].y;
        pinchState = {
          dist: Math.hypot(dx, dy) || 1,
          zoom: zoomPercent,
          midX: (pts[0].x + pts[1].x) / 2,
          midY: (pts[0].y + pts[1].y) / 2,
          scrollLeft: viewport.scrollLeft,
          scrollTop: viewport.scrollTop,
        };
        brushPreview.hidden = true;
        return;
      }
    }
    if (panMode || event.button === 1) {
      overlay.setPointerCapture(event.pointerId);
      panDrag = {
        pointer: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        scrollLeft: viewport.scrollLeft,
        scrollTop: viewport.scrollTop,
      };
      overlay.style.cursor = "grabbing";
      brushPreview.hidden = true;
      return;
    }
    if (current || event.button !== 0 || pinchState) return;
    if (strokes.length >= 200) { show("笔迹达到上限，请撤销或清空后继续。", true); return; }
    overlay.setPointerCapture(event.pointerId);
    const rect = overlay.getBoundingClientRect();
    current = {pointer: event.pointerId, erase, size: Number(brushSize.value) * overlay.width / rect.width, points: []};
    strokes.push(current);
    trackBrush(event, rect);
    const point = position(event, rect);
    current.points.push(point);
    paintOnSelection(current, null, point);
    render();
  });
  overlay.addEventListener("pointerenter", (event) => {
    if (!panMode) trackBrush(event, overlay.getBoundingClientRect());
  });
  overlay.addEventListener("pointerleave", () => { brushPreview.hidden = true; });
  overlay.addEventListener("pointermove", (event) => {
    const viewport = byId("viewport");
    if (activeTouches.has(event.pointerId)) {
      activeTouches.set(event.pointerId, {x: event.clientX, y: event.clientY});
      if (pinchState && activeTouches.size >= 2) {
        const pts = [...activeTouches.values()];
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
        setZoom(pinchState.zoom * (dist / pinchState.dist));
        const midX = (pts[0].x + pts[1].x) / 2;
        const midY = (pts[0].y + pts[1].y) / 2;
        viewport.scrollLeft = pinchState.scrollLeft - (midX - pinchState.midX);
        viewport.scrollTop = pinchState.scrollTop - (midY - pinchState.midY);
        return;
      }
    }
    if (panDrag && panDrag.pointer === event.pointerId) {
      viewport.scrollLeft = panDrag.scrollLeft - (event.clientX - panDrag.startX);
      viewport.scrollTop = panDrag.scrollTop - (event.clientY - panDrag.startY);
      return;
    }
    if (panMode || pinchState) return;
    if (current && current.pointer !== event.pointerId) return;
    const rect = overlay.getBoundingClientRect();
    trackBrush(event, rect);
    if (!current || current.points.length >= 4000) return;
    const point = position(event, rect);
    paintOnSelection(current, current.points.at(-1), point);
    current.points.push(point);
    render();
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
    overlay.addEventListener(name, (event) => {
      activeTouches.delete(event.pointerId);
      if (activeTouches.size < 2) pinchState = null;
      if (panDrag?.pointer === event.pointerId) {
        panDrag = null;
        overlay.style.cursor = panMode ? "grab" : "crosshair";
      }
      if (current?.pointer === event.pointerId) {
        const wasEraser = current.erase;
        current = null;
        if (fillHoles.checked && !wasEraser) {
          updateFill();
        }
        render();
      }
      if (event.pointerType === "touch" || name === "pointercancel") brushPreview.hidden = true;
    });
  }

  function updateToolButtons() {
    byId("brush").setAttribute("aria-pressed", String(!panMode && !erase));
    byId("eraser").setAttribute("aria-pressed", String(!panMode && erase));
    const panBtn = byId("pan-tool");
    if (panBtn) panBtn.setAttribute("aria-pressed", String(panMode));
    overlay.style.cursor = panMode ? "grab" : "crosshair";
    brushPreview.hidden = true;
  }
  for (const id of ["brush", "eraser"]) {
    byId(id).addEventListener("click", () => {
      panMode = false;
      erase = id === "eraser";
      updateToolButtons();
    });
  }
  byId("pan-tool")?.addEventListener("click", () => {
    panMode = !panMode;
    updateToolButtons();
  });
  byId("undo").addEventListener("click", () => {
    if (current) return;
    if (strokes.length) {
      strokes.pop();
    } else if (maskHistory.length) {
      const prev = maskHistory.pop();
      initialMask = prev.initialMask;
      strokes = prev.strokes;
    } else {
      initialMask = null;
    }
    replay();
  });
  byId("invert")?.addEventListener("click", () => {
    if (current || busy || finished || !selection.width) return;
    pushMaskHistory();
    const cur = snapshotMask();
    const canvas = document.createElement("canvas");
    canvas.width = selection.width;
    canvas.height = selection.height;
    const ctx = canvas.getContext("2d", {willReadFrequently: true});
    ctx.drawImage(cur, 0, 0);
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const {data} = frame;
    for (let i = 0; i < data.length; i += 4) {
      const inv = 255 - data[i + 3];
      data[i] = data[i + 1] = data[i + 2] = 255;
      data[i + 3] = inv;
    }
    ctx.putImageData(frame, 0, 0);
    initialMask = canvas;
    strokes = [];
    replay();
    show("已反选当前遮罩（可点撤销恢复）。");
  });
  byId("clear").addEventListener("click", () => {
    if (current) return;
    if (initialMask || strokes.length) pushMaskHistory();
    initialMask = null;
    strokes = [];
    replay();
  });
  byId("zoom").addEventListener("click", () => {
    setZoom(zoomPercent > 100 ? 100 : 200);
  });

  async function waitForAutoMask(job) {
    const until = Date.now() + 15 * 60 * 1000;
    while (Date.now() < until) {
      if (job.status === "queued") {
        const ahead = Number(job.ahead) || 0;
        show(ahead > 0 ? `正在排队等待自动识别（前面还有 ${ahead} 个绘图任务）…` : "自动遮罩已排队，即将开始…");
      } else {
        show("正在使用 SAM3 识别目标…");
      }
      await pause(1500);
      let response;
      try {
        response = await api("auto?job=" + encodeURIComponent(job.id), {
          headers: originalHeaders(), signal: timedSignal(8000),
        });
      } catch (error) {
        if (error.status && error.status < 500) throw error;
        continue;
      }
      if (response.status !== 202) return response;
      job = await response.json();
    }
    throw new Error("等待自动遮罩超时，请重新获取编辑链接。");
  }

  async function showAutoMask(target, job) {
    lock(true);
    show("正在自动识别目标…");
    try {
      let result;
      if (job) {
        result = await waitForAutoMask(job);
      } else {
        result = await api("auto", {method: "POST", body: target, headers: originalHeaders()});
        if (result.status === 202) result = await waitForAutoMask(await result.json());
      }
      const bitmap = await createImageBitmap(await result.blob());
      const canvas = document.createElement("canvas");
      canvas.width = selection.width;
      canvas.height = selection.height;
      const mask = canvas.getContext("2d", {willReadFrequently: true});
      mask.drawImage(bitmap, 0, 0);
      bitmap.close();
      const frame = mask.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < frame.data.length; i += 4) {
        frame.data[i + 3] = frame.data[i];
        frame.data[i] = frame.data[i + 1] = frame.data[i + 2] = 255;
      }
      mask.putImageData(frame, 0, 0);
      const append = byId("auto-append")?.checked !== false && (initialMask || strokes.length);
      if (initialMask || strokes.length) pushMaskHistory();
      if (append) {
        const merged = snapshotMask();
        merged.getContext("2d").drawImage(canvas, 0, 0);
        initialMask = merged;
      } else {
        initialMask = canvas;
      }
      strokes = [];
      replay();
      show(append ? "已叠加自动遮罩到当前选区（可点撤销回退）。" : "检查红色区域，必要时手动修正后提交。");
    } catch (error) {
      show(error.message, true);
    } finally {
      lock(false);
    }
  }
  byId("auto").addEventListener("click", () => {
    const target = byId("target").value.trim();
    if (!target || current || busy || finished) {
      if (!target) show("请先填写要识别的目标。", true);
      return;
    }
    const append = byId("auto-append")?.checked !== false;
    if (!append && (initialMask || strokes.length) && !window.confirm("自动遮罩会替换当前选区。继续吗？")) return;
    showAutoMask(target);
  });

  function setReference(file) {
    referenceFile = file;
    if (referenceUrl) URL.revokeObjectURL(referenceUrl);
    referenceUrl = file ? URL.createObjectURL(file) : "";
    if (referenceUrl) referencePreview.src = referenceUrl;
    else referencePreview.removeAttribute("src");
    referencePreview.hidden = !referenceUrl;
    byId("reference-clear").hidden = !referenceUrl;
    referenceDrop.textContent = file ? "拖入或点击更换图片" : "拖入图片，或点击选择文件";
  }

  function selectReference(files) {
    if (!files?.length || busy || finished || !maxReferenceBytes) return;
    if (files.length !== 1) {
      show("请一次选择一张参考图。", true);
      return;
    }
    const file = files[0];
    if (file.size > maxReferenceBytes) {
      show("参考图 2 最大 8 MiB。", true);
      return;
    }
    const supported = file.type ? /^image\/(jpeg|png|webp)$/i.test(file.type) : /\.(jpe?g|png|webp)$/i.test(file.name);
    if (!file.size || !supported) {
      show("请选择 JPG、PNG 或 WebP 图片。", true);
      return;
    }
    setReference(file);
    show("");
  }

  referenceDrop.addEventListener("click", () => referenceInput.click());
  referenceInput.addEventListener("change", () => {
    selectReference(referenceInput.files);
    referenceInput.value = "";
  });
  byId("reference-clear").addEventListener("click", () => {
    setReference(null);
  });
  function fileDrag(event) {
    return Array.from(event.dataTransfer?.types || []).includes("Files");
  }
  function bindImageDrop(target, select) {
    target.addEventListener("dragover", (event) => {
      if (!fileDrag(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = target.disabled ? "none" : "copy";
      if (!target.disabled) target.dataset.drag = "true";
    });
    target.addEventListener("dragleave", () => { delete target.dataset.drag; });
    target.addEventListener("drop", (event) => {
      if (!fileDrag(event)) return;
      event.preventDefault();
      delete target.dataset.drag;
      if (!target.disabled) select(event.dataTransfer.files);
    });
  }
  bindImageDrop(referenceDrop, selectReference);
  for (const type of ["dragover", "drop"]) {
    window.addEventListener(type, (event) => {
      if (fileDrag(event)) event.preventDefault();
    });
  }

  for (const [id, outputId] of [["grow", "grow-value"], ["feather", "feather-value"]]) {
    const slider = byId(id);
    const output = byId(outputId);
    const update = () => {
      output.textContent = slider.value + " px";
      if (selection.width) render();
    };
    slider.addEventListener("input", update);
    update();
  }

  byId("submit").addEventListener("click", async () => {
    if (current || busy || finished) return;
    const prompt = byId("prompt").value.trim();
    const reference = referenceFile;
    if (reference && !prompt) { show("上传参考图 2 后，请填写修改要求。", true); return; }
    if (!prompt && !alternateModes) { show("请输入修改要求。", true); return; }
    lock(true);
    show("正在提交…");
    try {
      const output = document.createElement("canvas");
      output.width = selection.width;
      output.height = selection.height;
      const ctx = output.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, output.width, output.height);
      drawMask(ctx);
      const mask = await new Promise(resolve => output.toBlob(resolve, "image/png"));
      if (!mask) throw new Error("无法导出遮罩");
      const grow = Number(byId("grow").value) || 0;
      const feather = Number(byId("feather").value) || 0;
      const edgeOptions = grow || feather;
      const header = (reference || edgeOptions)
        ? {prompt, grow, feather, ...(reference ? {mask_bytes: mask.size} : {})} : prompt;
      const parts = [JSON.stringify(header) + "\n", mask];
      if (reference) parts.push(reference);
      const body = new Blob(parts, {type: "application/octet-stream"});
      await api("submit", {method: "POST", body, headers: originalHeaders()});
      finished = true;
      setReference(null);
      edit.hidden = true;
      show("已提交。完成后，结果会发回原 QQ 会话。");
    } catch (error) {
      show(error.message, true);
    } finally {
      lock(false);
    }
  });

  async function openImage(state, {keepPrompt = false, keepMask = false} = {}) {
    const signal = typeof AbortSignal.timeout === "function" ? AbortSignal.timeout(60000) : undefined;
    const headers = originalHeaders(state.original_revision || 0);
    const url = URL.createObjectURL(await (await api("image", {signal, headers})).blob());
    const photo = byId("photo");
    photo.src = url;
    try {
      await photo.decode();
    } finally {
      URL.revokeObjectURL(url);
    }
    originalRevision = state.original_revision || 0;
    overlay.width = selection.width = state.width;
    overlay.height = selection.height = state.height;
    byId("stage").style.setProperty("--photo-width", state.width + "px");
    if (keepMask) {
      replay();
    } else {
      strokes = [];
      maskHistory = [];
      initialMask = fill = null;
      brushPreview.hidden = true;
    }
    if (!keepPrompt) byId("prompt").value = state.prompt || "";
    alternateModes = state.alternate_modes === true;
    maxReferenceBytes = state.max_reference_bytes || 0;
    byId("reference-panel").hidden = !maxReferenceBytes;
    byId("mode-help").hidden = !alternateModes;
    const dimensions = byId("preview-size");
    dimensions.hidden = !Array.isArray(state.original_size)
      || (state.original_size[0] === state.width && state.original_size[1] === state.height);
    if (!dimensions.hidden) {
      dimensions.textContent = "预览 " + state.width + " × " + state.height + "，原图 "
        + state.original_size.join(" × ") + "。带遮罩的任务将保留原图尺寸。";
    }
    originalUpload.hidden = true;
    replacingOriginal = false;
    replaceOriginalButton.hidden = state.can_replace_original !== true;
    edit.hidden = false;
    show(alternateModes ? "涂抹选区，或输入修改要求。" : "涂出要修改的区域，也可以先让系统自动识别。");
  }

  function lockOriginalUpload(value) {
    for (const id of ["original-file", "original-drop", "upload-original", "restore-original"]) {
      byId(id).disabled = value;
    }
  }

  function showOriginalUpload(state) {
    originalRevision = state.original_revision || 0;
    maxOriginalBytes = state.max_original_bytes;
    originalChunkBytes = state.original_chunk_bytes || 0;
    originalChunkParallelism = state.original_chunk_parallelism === 2 ? 2 : 1;
    originalCacheEnabled = state.original_cache_enabled === true;
    replacingOriginal = state.can_restore_original === true;
    originalFile = null;
    originalUploadId = "";
    originalInput.value = "";
    originalDrop.textContent = "拖入原图，或点击选择文件";
    restoreOriginalButton.hidden = !replacingOriginal;
    edit.hidden = true;
    originalUpload.hidden = false;
    const message = replacingOriginal ? "请选择新原图。更换成功后会清空当前遮罩。" : "请上传要编辑的原图。";
    show(state.original_upload?.error || message, Boolean(state.original_upload?.error));
  }

  replaceOriginalButton.addEventListener("click", async () => {
    if (busy || current || finished) return;
    lock(true);
    try {
      const state = await (await api("original/replace", {
        method: "POST", headers: originalHeaders(), signal: timedSignal(10000),
      })).json();
      showOriginalUpload(state);
    } catch (error) {
      show(error.message, true);
    } finally {
      lock(false);
    }
  });

  restoreOriginalButton.addEventListener("click", async () => {
    lockOriginalUpload(true);
    try {
      const state = await (await api("original/restore", {
        method: "POST", headers: originalHeaders(), signal: timedSignal(10000),
      })).json();
      await openImage(state, {keepPrompt: true, keepMask: true});
      show("已恢复原图，可以继续编辑。");
    } catch (error) {
      show(error.message, true);
    } finally {
      lockOriginalUpload(false);
    }
  });

  function uploadOriginal(file) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/original");
      request.setRequestHeader("Authorization", "Bearer " + token);
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.timeout = 150000;
      request.upload.addEventListener("progress", (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.min(100, Math.round(100 * event.loaded / event.total));
        originalProgress.value = percent;
        show("正在上传原图… " + percent + "%");
      });
      request.upload.addEventListener("load", () => {
        originalProgress.removeAttribute("value");
        show("原图已发送，正在等待服务器处理…");
      });
      request.addEventListener("load", () => {
        if (request.status >= 200 && request.status < 300) {
          try { resolve(JSON.parse(request.responseText)); }
          catch { reject(new Error("服务器返回的原图状态无法读取。")); }
        } else {
          const text = request.status >= 500 ? "服务器未能及时处理原图（HTTP " + request.status + "）。"
            : request.responseText;
          reject(new Error(text || "原图上传失败。"));
        }
      });
      request.addEventListener("error", () => reject(new Error("上传连接中断，请重试。")));
      request.addEventListener("timeout", () => reject(new Error("等待原图处理超时，请重试。")));
      request.addEventListener("abort", () => reject(new Error("原图上传已中止。")));
      request.send(file);
    });
  }

  const pause = (milliseconds) => new Promise(resolve => setTimeout(resolve, milliseconds));
  const timedSignal = (milliseconds) => typeof AbortSignal.timeout === "function"
    ? AbortSignal.timeout(milliseconds) : undefined;

  async function findCachedOriginal(file) {
    if (!originalCacheEnabled || !globalThis.crypto?.subtle) return null;
    show("正在校验原图…");
    let digest;
    try {
      const hash = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
      digest = Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, "0")).join("");
    } catch {
      return null;
    }
    try {
      const response = await api("original/cache", {method: "POST", body: digest, signal: timedSignal(8000)});
      const state = await response.json();
      return state.hit ? state : null;
    } catch (error) {
      if (error.status === 400 || error.status === 404) throw error;
      return null;
    }
  }

  async function sendOriginalChunk(file, id, offset) {
    const chunk = file.slice(offset, offset + originalChunkBytes);
    for (let attempt = 0; ; attempt++) {
      try {
        const response = await api("original/chunk", {
          method: "POST", body: chunk, signal: timedSignal(40000),
          headers: {
            "Content-Type": "application/octet-stream", "X-Original-Upload": id,
            "X-Original-Offset": String(offset), "X-Original-Size": String(file.size),
          },
        });
        return await response.json();
      } catch (error) {
        const retryable = !error.status || [408, 409, 429].includes(error.status) || error.status >= 500;
        if (!retryable || attempt >= 2) throw error;
        show("上传遇到网络波动，正在重试当前片段…");
        await pause(500 * (attempt + 1));
      }
    }
  }

  async function waitForOriginal(id) {
    originalProgress.removeAttribute("value");
    show("群晖已收齐原图，正在生成预览…");
    const until = Date.now() + 95000;
    while (Date.now() < until) {
      let state;
      try {
        state = await (await api("draft", {signal: timedSignal(8000)})).json();
      } catch (error) {
        if (error.status && error.status < 500) throw error;
      }
      if (state && !state.needs_original) return state;
      if (state?.original_upload?.id === id && state.original_upload.status === "failed") {
        originalUploadId = "";
        throw new Error(state.original_upload.error || "生成预览失败。");
      }
      await pause(1000);
    }
    throw new Error("原图已收到，等待预览超时。请刷新页面查看处理结果。");
  }

  async function uploadInChunks(file) {
    if (!originalUploadId) {
      originalUploadId = Array.from(crypto.getRandomValues(new Uint8Array(16)),
        byte => byte.toString(16).padStart(2, "0")).join("");
    }
    const id = originalUploadId;
    let nextOffset = 0;
    let confirmed = 0;
    let failure;
    async function worker() {
      while (!failure && nextOffset < file.size) {
        const offset = nextOffset;
        const end = Math.min(file.size, offset + originalChunkBytes);
        nextOffset = end;
        try {
          const upload = await sendOriginalChunk(file, id, offset);
          const received = upload.received_total ?? upload.received;
          const acknowledged = originalChunkParallelism === 1
            ? upload.received >= end : upload.acknowledged_offset === offset;
          if (upload.id !== id || !acknowledged || !Number.isInteger(upload.received)
              || !Number.isInteger(received) || upload.received < 0 || upload.received > received
              || received < end - offset || received > file.size) {
            throw new Error("群晖返回的上传进度无效，请重试。");
          }
          nextOffset = Math.max(nextOffset, upload.received);
          confirmed = Math.max(confirmed, received);
          const percent = Math.floor(100 * confirmed / file.size);
          originalProgress.value = percent;
          show("正在上传原图… 群晖已接收 " + percent + "%");
        } catch (error) {
          failure ||= error;
        }
      }
    }
    const workers = Math.min(originalChunkParallelism, Math.ceil(file.size / originalChunkBytes));
    await Promise.all(Array.from({length: workers}, () => worker()));
    if (failure) throw failure;
    if (confirmed !== file.size) throw new Error("群晖尚未确认收齐原图，请重试。");
    return await waitForOriginal(id);
  }

  function selectOriginal(files) {
    if (!files?.length) return;
    if (files.length !== 1) { show("请一次选择一张原图。", true); return; }
    const file = files[0];
    if (file.size > maxOriginalBytes) { show("原图最大 48 MiB。", true); return; }
    if (!file.size || !/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name)) {
      show("请选择 JPG、PNG、WebP、BMP 或 TIFF 图片。", true);
      return;
    }
    originalFile = file;
    originalUploadId = "";
    originalDrop.textContent = file.name + " · 点击或拖入以更换";
    show("原图已选择，点击「上传原图」继续。");
    return true;
  }
  bindImageDrop(originalDrop, (files) => {
    if (selectOriginal(files)) uploadSelectedOriginal();
  });
  originalDrop.addEventListener("click", () => originalInput.click());
  originalInput.addEventListener("change", () => {
    selectOriginal(originalInput.files);
    originalInput.value = "";
  });

  async function uploadSelectedOriginal() {
    const button = byId("upload-original");
    if (button.disabled) return;
    const file = originalFile;
    if (!file) { show("请先选择一张原图。", true); return; }
    if (file.size > maxOriginalBytes) { show("原图最大 48 MiB。", true); return; }
    lockOriginalUpload(true);
    const keepPrompt = replacingOriginal;
    originalProgress.value = 0;
    originalProgress.hidden = false;
    show("正在上传原图… 0%");
    try {
      const cached = await findCachedOriginal(file);
      if (!cached) show("正在上传原图… 0%");
      const state = cached || await (originalChunkBytes ? uploadInChunks(file) : uploadOriginal(file));
      show("原图处理完成，正在载入画板…");
      await openImage(state, {keepPrompt});
      if (cached) show("已复用相同原图。可以涂抹选区或输入修改要求。");
      originalFile = null;
    } catch (error) {
      try {
        // The server may have stored the image even if the response was lost.
        const signal = typeof AbortSignal.timeout === "function" ? AbortSignal.timeout(8000) : undefined;
        const state = await (await api("draft", {signal})).json();
        if (!state.needs_original) {
          show("原图处理完成，正在载入画板…");
          await openImage(state, {keepPrompt});
          originalFile = null;
        } else {
          show(error.message, true);
        }
      } catch {
        show(error.message, true);
      }
    } finally {
      originalProgress.hidden = true;
      lockOriginalUpload(false);
    }
  }
  byId("upload-original").addEventListener("click", uploadSelectedOriginal);

  async function open() {
    if (!/^[A-Za-z0-9_-]{43}$/.test(token)) {
      show("请使用 QQ 机器人私发的有效编辑链接。", true);
      return;
    }
    try {
      const state = await (await api("draft")).json();
      if (state.needs_original) {
        showOriginalUpload(state);
        if (state.original_upload?.status === "processing") {
          lockOriginalUpload(true);
          originalProgress.hidden = false;
          try { await openImage(await waitForOriginal(state.original_upload.id)); }
          finally {
            originalProgress.hidden = true;
            lockOriginalUpload(false);
          }
        }
      } else {
        await openImage(state);
        if (["queued", "processing", "ready"].includes(state.auto_job?.status)) {
          showAutoMask(null, state.auto_job);
        }
      }
    } catch (error) {
      show(error.message, true);
    }
  }
  open();
})();
