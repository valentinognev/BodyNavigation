import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import * as THREE from "three";
import { axisLabel } from "./plot";
import {
  axisTickMarks,
  centeredCloud,
  dataAxisLines,
  dragOrbit,
  legendHudRows,
  markerSize,
  opaqueClearColor,
  orbitCamera,
  shouldStopRecorder,
  startFramePump,
  wheelOrbit,
  type Orbit,
  type TrajectoryScene,
  type Vec3,
} from "./plot3d";

const buttonClass =
  "w-auto rounded bg-slate-800 px-3 py-1 text-sm text-white disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900";

const AXIS_COLOR = "#64748b";
const START_COLOR = "#15803d";
const END_COLOR = "#b91c1c";
const LABEL_FILL = "#1e293b";
const VIEW_W = 360;
const VIEW_H = 280;

function openCaptureStream(canvas: HTMLCanvasElement): {
  stream: MediaStream;
  requestFrame: (() => void) | undefined;
} {
  const manual = canvas.captureStream(0);
  const track = manual.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined;
  if (track != null && typeof track.requestFrame === "function") {
    return { stream: manual, requestFrame: () => track.requestFrame() };
  }
  manual.getTracks().forEach((item) => item.stop());
  return { stream: canvas.captureStream(30), requestFrame: undefined };
}

function offsetBy(point: Vec3, center: Vec3): Vec3 {
  return { x: point.x - center.x, y: point.y - center.y, z: point.z - center.z };
}

function vector3(point: Vec3): THREE.Vector3 {
  return new THREE.Vector3(point.x, point.y, point.z);
}

function makeMarkerTexture(kind: "start" | "end"): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  if (ctx == null) return texture;
  if (kind === "start") {
    ctx.beginPath();
    ctx.arc(32, 32, 28, 0, Math.PI * 2);
    ctx.fillStyle = START_COLOR;
    ctx.fill();
  } else {
    ctx.fillStyle = END_COLOR;
    ctx.fillRect(8, 8, 48, 48);
  }
  texture.needsUpdate = true;
  return texture;
}

function makeTextSprite(text: string, worldHeight: number, sizeAttenuation = true): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const probe = canvas.getContext("2d");
  const font = "24px sans-serif";
  let width = 64;
  let height = 36;
  if (probe != null) {
    probe.font = font;
    width = Math.max(8, Math.ceil(probe.measureText(text).width) + 12);
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (ctx != null) {
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#ffffff";
    ctx.strokeText(text, width / 2, height / 2);
    ctx.fillStyle = LABEL_FILL;
    ctx.fillText(text, width / 2, height / 2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, sizeAttenuation }),
  );
  sprite.scale.set(worldHeight * (width / height), worldHeight, 1);
  return sprite;
}

function addLine(
  scene: THREE.Scene,
  from: Vec3,
  to: Vec3,
  color: string,
  bin: { dispose(): void }[],
): void {
  const geometry = new THREE.BufferGeometry().setFromPoints([vector3(from), vector3(to)]);
  const material = new THREE.LineBasicMaterial({ color });
  scene.add(new THREE.Line(geometry, material));
  bin.push(geometry, material);
}

export function TrajectoryView({ scene, label }: { scene: TrajectoryScene; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const orbitRef = useRef<Orbit>({ azimuth: 45, elevation: 25, distance: 1 });
  const renderRef = useRef<(() => void) | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const stopPumpRef = useRef<(() => void) | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordError, setRecordError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas == null) return;
    const probe = document.createElement("canvas");
    if (probe.getContext("webgl2") == null && probe.getContext("webgl") == null) return;
    const pathPoints = scene.series.flatMap((item) => item.points);
    const { center, radius } = centeredCloud(pathPoints);
    const size = markerSize(radius);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
        preserveDrawingBuffer: true,
      });
    } catch {
      return;
    }
    const width = canvas.clientWidth || VIEW_W;
    const height = canvas.clientHeight || VIEW_H;
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height, false);
    renderer.autoClear = false;

    const world = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, radius / 100, radius * 200);
    camera.up.set(0, 0, 1);
    const bin: { dispose(): void }[] = [];

    for (const series of scene.series) {
      const cloud = series.points.map((point) => offsetBy(point, center));
      const positions = new Float32Array(cloud.length * 3);
      cloud.forEach((point, index) => {
        positions[index * 3] = point.x;
        positions[index * 3 + 1] = point.y;
        positions[index * 3 + 2] = point.z;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({ color: series.color });
      world.add(new THREE.Line(geometry, material));
      bin.push(geometry, material);
    }

    for (const marker of scene.markers) {
      const texture = makeMarkerTexture(marker.kind);
      const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
      const sprite = new THREE.Sprite(material);
      const at = offsetBy(marker.point, center);
      sprite.position.set(at.x, at.y, at.z);
      sprite.scale.set(size, size, 1);
      world.add(sprite);
      bin.push(texture, material);
    }

    const axisHeight = radius * 0.1;
    const tickHeight = radius * 0.07;
    dataAxisLines(scene.axes).forEach((line, index) => {
      const from = offsetBy(line.from, center);
      const to = offsetBy(line.to, center);
      addLine(world, from, to, AXIS_COLOR, bin);
      const axisIndex = index as 0 | 1 | 2;
      const marks = axisTickMarks(scene.axes[axisIndex], from, to, axisIndex, radius);
      for (const mark of marks) {
        for (const cross of mark.crosses) addLine(world, cross.from, cross.to, AXIS_COLOR, bin);
        const tickSprite = makeTextSprite(mark.label, tickHeight);
        const nudge = mark.crosses[0];
        tickSprite.position.set(
          mark.at.x + (nudge.to.x - mark.at.x) * 3,
          mark.at.y + (nudge.to.y - mark.at.y) * 3,
          mark.at.z + (nudge.to.z - mark.at.z) * 3,
        );
        world.add(tickSprite);
        const tickMap = tickSprite.material.map;
        if (tickMap != null) bin.push(tickMap);
        bin.push(tickSprite.material);
      }
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const dz = to.z - from.z;
      const length = Math.hypot(dx, dy, dz) || 1;
      const axisSprite = makeTextSprite(axisLabel(line.name), axisHeight);
      axisSprite.position.set(
        to.x + (dx / length) * radius * 0.08,
        to.y + (dy / length) * radius * 0.08,
        to.z + (dz / length) * radius * 0.08,
      );
      world.add(axisSprite);
      const axisMap = axisSprite.material.map;
      if (axisMap != null) bin.push(axisMap);
      bin.push(axisSprite.material);
    });

    const hud = new THREE.Scene();
    const hudCamera = new THREE.OrthographicCamera(0, width, height, 0, -10, 10);
    const hudRows = legendHudRows(scene.legend, width, height);
    for (const row of hudRows) {
      const chipGeom = new THREE.PlaneGeometry(row.chip.w, row.chip.h);
      const chipMat = new THREE.MeshBasicMaterial({
        color: row.color,
        side: THREE.DoubleSide,
      });
      const chip = new THREE.Mesh(chipGeom, chipMat);
      chip.position.set(row.chip.x + row.chip.w / 2, height - (row.chip.y + row.chip.h / 2), 0);
      hud.add(chip);
      bin.push(chipGeom, chipMat);
      const text = makeTextSprite(row.label, 14, false);
      text.position.set(row.text.x + text.scale.x / 2, height - row.text.y, 0);
      hud.add(text);
      const textMap = text.material.map;
      if (textMap != null) bin.push(textMap);
      bin.push(text.material);
    }

    orbitRef.current = { azimuth: 45, elevation: 25, distance: Math.max(radius * 3.5, 1) };

    const draw = () => {
      const orbit = orbitRef.current;
      const cameraPoint = orbitCamera(orbit.azimuth, orbit.elevation, orbit.distance);
      camera.position.set(cameraPoint.x, cameraPoint.y, cameraPoint.z);
      camera.lookAt(0, 0, 0);
      const css = typeof getComputedStyle === "function" ? getComputedStyle(canvas).backgroundColor : "";
      renderer.setClearColor(opaqueClearColor(css), 1);
      renderer.clear();
      renderer.render(world, camera);
      renderer.clearDepth();
      renderer.render(hud, hudCamera);
    };
    renderRef.current = draw;
    draw();

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      orbitRef.current = wheelOrbit(orbitRef.current, event.deltaY);
      draw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      renderRef.current = null;
      for (const item of bin) item.dispose();
      renderer.dispose();
      stopPumpRef.current?.();
      stopPumpRef.current = null;
      const recorder = recorderRef.current;
      if (recorder != null && shouldStopRecorder(recorder.state)) recorder.stop();
      recorderRef.current = null;
    };
  }, [scene]);

  function onPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    dragRef.current = { x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (drag == null) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    dragRef.current = { x: event.clientX, y: event.clientY };
    orbitRef.current = dragOrbit(orbitRef.current, dx, dy);
    renderRef.current?.();
  }

  function onPointerUp() {
    dragRef.current = null;
  }

  function startRecording() {
    const canvas = canvasRef.current;
    if (
      canvas == null ||
      typeof MediaRecorder === "undefined" ||
      typeof canvas.captureStream !== "function"
    ) {
      setRecordError("Recording is unavailable");
      return;
    }
    let stream: MediaStream;
    let requestFrame: (() => void) | undefined;
    try {
      const opened = openCaptureStream(canvas);
      stream = opened.stream;
      requestFrame = opened.requestFrame;
    } catch {
      setRecordError("Recording is unavailable");
      return;
    }
    let recorder: MediaRecorder;
    try {
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";
      recorder = new MediaRecorder(stream, { mimeType: mime });
    } catch {
      try {
        recorder = new MediaRecorder(stream);
      } catch {
        setRecordError("Recording is unavailable");
        return;
      }
    }
    const chunks: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = () => {
      recorderRef.current = null;
      if (chunks.length > 0) {
        const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "trajectory.webm";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
      }
      setRecording(false);
    };
    recorder.start();
    recorderRef.current = recorder;
    stopPumpRef.current?.();
    stopPumpRef.current = startFramePump(() => renderRef.current?.(), requestFrame);
    setRecording(true);
    setRecordError(null);
  }

  function stopRecording() {
    stopPumpRef.current?.();
    stopPumpRef.current = null;
    const recorder = recorderRef.current;
    if (recorder != null && shouldStopRecorder(recorder.state)) recorder.stop();
  }

  return (
    <div className="mb-3">
      <canvas
        ref={canvasRef}
        width={VIEW_W}
        height={VIEW_H}
        className="w-full rounded border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900"
        aria-label={`${label}, drag to orbit`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      />
      <div className="mt-2 flex items-center gap-2">
        {recording ? (
          <button type="button" className={buttonClass} onClick={stopRecording}>
            Stop
          </button>
        ) : (
          <button type="button" className={buttonClass} onClick={startRecording}>
            Record
          </button>
        )}
        <span className="text-xs text-slate-500 dark:text-slate-400">Drag to orbit. Scroll to zoom.</span>
        {recordError != null ? (
          <span className="text-xs text-red-700 dark:text-red-300">{recordError}</span>
        ) : null}
      </div>
    </div>
  );
}
