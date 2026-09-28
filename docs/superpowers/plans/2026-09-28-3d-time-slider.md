# 3D Time Slider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a time slider below the 3D orbit view that scrubs the run: it shows the partial trajectory up to the selected moment plus one position dot per vehicle at that moment.

**Architecture:** Times ride on the existing `Trajectory` points as a parallel array. A shared sorted-unique time domain feeds one range input in `ResultsPane`; per-tick truncation plus linear interpolation produce a view scene that reuses the full-run axes so the camera frame never jumps. Position dots render as colored spheres in `TrajectoryView`, whose orbit state survives scrubbing and resets only on a new run.

**Tech Stack:** TypeScript, React 18, three.js, Vitest (all in `workbench/web`; no Python changes).

**Spec:** Approved in chat on 2026-09-28 (bounded change; no separate spec file). User answers: slider scrubs the 3D orbit view only (time charts and ground track untouched); one shared time across vehicles.

## Global Constraints

- 3D orbit view only. Do not touch time charts, ground track, `plot.csv`, or the checklist.
- Shared domain = sorted unique finite `time` values across all trajectory tracks. Slider defaults to the last tick (full trajectory).
- Axes stay fixed at the full-run scale while scrubbing; only series points and position dots change.
- Prefix = points with `time <= t`, plus a linearly interpolated point at exactly `t` (clamped to endpoints). Every vehicle with a non-empty trajectory gets exactly one position dot in its own track color.
- `start` markers stay; full-run `end` markers are dropped in the scrubbed view (the dots are the ends now). Legend unchanged (no position entries).
- Slider hidden when there is no scene or fewer than 2 distinct times.
- Backend already guarantees `time` on every track row (`Python/src/cadac/cli.py::make_plot_on_step` injects `{"time": ctx.sim_time}` when missing). Missing/non-finite `row.time` falls back to the row index so fixtures without time still work.
- TDD: failing test, watch it fail, minimal code, watch it pass. Grok non-fast implementers and reviewers only.
- Do not edit `CADAC_Simulations/`, `Python/`, or `README.md`. Bump `UPDATES.md` to `0.189.0` (feature → subver) in Task 3.
- Commit steps run only when the user has asked for commits in that session.
- Build on top of the current working tree (it holds uncommitted 0.188.x Results changes); do not stash or revert.
- Task 1 first. Tasks 2 and 3 after Task 1 (Task 3 touches files Task 2 touches; run sequentially).

## File structure

- `workbench/web/src/plot3d.ts` — `Trajectory.times`, `sliderTimes`, `trackPositionAt`, `truncateTrajectory`, `sceneOfTracksUpTo`, `shouldResetOrbit`, `position` marker kind. No React.
- `workbench/web/src/plot3d.test.ts` — unit proofs for the above plus updates to existing `trajectoryOf`/`sceneOf` literals.
- `workbench/web/src/ResultsPane.tsx` — slider state, domain memo, scrubbed scene memo, range input below `TrajectoryView`.
- `workbench/web/src/ResultsPane.test.ts` — slider DOM proofs.
- `workbench/web/src/TrajectoryView.tsx` — `resetKey` prop, orbit preservation, position spheres.
- `UPDATES.md` — `0.189.0` entry on top.

---

### Task 1: Timed trajectories and scrubbed scenes

**Files:**
- Modify: `workbench/web/src/plot3d.ts`
- Modify: `workbench/web/src/plot3d.test.ts`

**Interfaces:**
- Consumes: existing `Trajectory`, `trajectoryOf`, `trajectoriesFromPlot`, `sceneOfTracks`, `TRACK_COLORS` (unchanged signatures except `Trajectory` gains a field).
- Produces:
  - `Trajectory = { kind: "geographic" | "local"; xName: string; yName: string; zName: string; points: Vec3[]; times: number[] }` (`times` parallel to `points`)
  - `sliderTimes(tracks: { name: string; trajectory: Trajectory }[]): number[]`
  - `trackPositionAt(trajectory: Trajectory, t: number): Vec3 | null`
  - `truncateTrajectory(trajectory: Trajectory, t: number): { points: Vec3[]; position: Vec3 | null }`
  - `sceneOfTracksUpTo(tracks: { name: string; trajectory: Trajectory }[], t: number): TrajectoryScene | null`
  - `shouldResetOrbit(prevKey: string, nextKey: string): boolean`
  - Marker kind becomes `"start" | "end" | "position"`; `position` markers carry `color: string` (series color) and `label` = vehicle name. `start`/`end` markers unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `workbench/web/src/plot3d.test.ts`:

```typescript
import {
  sceneOfTracksUpTo,
  shouldResetOrbit,
  sliderTimes,
  trackPositionAt,
  truncateTrajectory,
} from "./plot3d";

const TIMED = {
  kind: "geographic" as const,
  xName: "lonx",
  yName: "latx",
  zName: "alt",
  points: [
    { x: 0, y: 0, z: 0 },
    { x: 10, y: 10, z: 100 },
    { x: 20, y: 20, z: 200 },
  ],
  times: [0, 1, 2],
};

it("collects one sorted unique time domain across vehicles", () => {
  const other = { ...TIMED, times: [1, 3] };
  expect(
    sliderTimes([
      { name: "a", trajectory: TIMED },
      { name: "b", trajectory: other },
    ]),
  ).toEqual([0, 1, 2, 3]);
});

it("interpolates the vehicle position at an exact moment", () => {
  expect(trackPositionAt(TIMED, 0.5)).toEqual({ x: 5, y: 5, z: 50 });
  expect(trackPositionAt(TIMED, -1)).toEqual({ x: 0, y: 0, z: 0 });
  expect(trackPositionAt(TIMED, 9)).toEqual({ x: 20, y: 20, z: 200 });
  expect(trackPositionAt({ ...TIMED, points: [], times: [] }, 1)).toBeNull();
});

it("truncates to the prefix plus the interpolated moment", () => {
  const cut = truncateTrajectory(TIMED, 0.5);
  expect(cut.points).toEqual([
    { x: 0, y: 0, z: 0 },
    { x: 5, y: 5, z: 50 },
  ]);
  expect(cut.position).toEqual({ x: 5, y: 5, z: 50 });
});

it("builds a scrubbed scene on full-run axes with one dot per vehicle", () => {
  const scene = sceneOfTracksUpTo([{ name: "vehicle", trajectory: TIMED }], 1);
  expect(scene).not.toBeNull();
  expect(scene!.series[0].points).toHaveLength(2);
  expect(scene!.axes).toEqual(sceneOfTracks([{ name: "vehicle", trajectory: TIMED }])!.axes);
  expect(scene!.markers.filter((m) => m.kind === "position")).toHaveLength(1);
  expect(scene!.markers.some((m) => m.kind === "end")).toBe(false);
  expect(scene!.markers.find((m) => m.kind === "position")).toMatchObject({
    label: "vehicle",
    color: "#0f766e",
  });
});

it("resets the orbit only when the run key changes", () => {
  expect(shouldResetOrbit("a", "a")).toBe(false);
  expect(shouldResetOrbit("a", "b")).toBe(true);
});
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd workbench/web && npm test -- plot3d.test.ts`
Expected: FAIL — `sceneOfTracksUpTo`, `shouldResetOrbit`, `sliderTimes`, `trackPositionAt`, `truncateTrajectory` are not exported.

- [ ] **Step 3: Write the minimal implementation**

In `workbench/web/src/plot3d.ts`:

1. Add `times: number[]` to the `Trajectory` type.
2. In `trajectoryOf`, iterate with the row index and push the time alongside each kept point:

```typescript
const rows Kept: for (let index = 0; index < rows.length; index += 1) {
  const row = rows[index];
  // ... existing point computation ...
  if (finitePoint(point)) {
    points.push(point);
    const t = Number(row.time);
    times.push(Number.isFinite(t) ? t : index);
  }
}
```

3. Fix the three `return { kind, xName, yName, zName, points }` sites to include `times`.
4. Widen the marker type and add the helpers:

```typescript
export type TrajectoryMarker = {
  kind: "start" | "end" | "position";
  label: string;
  point: Vec3;
  color?: string;
};

export function sliderTimes(tracks: { name: string; trajectory: Trajectory }[]): number[] {
  const seen = new Set<number>();
  for (const track of tracks) {
    for (const t of track.trajectory.times) {
      if (Number.isFinite(t)) seen.add(t);
    }
  }
  return [...seen].sort((a, b) => a - b);
}

export function trackPositionAt(trajectory: Trajectory, t: number): Vec3 | null {
  const { points, times } = trajectory;
  if (points.length === 0) return null;
  if (t <= times[0]) return points[0];
  for (let i = 1; i < points.length; i += 1) {
    if (t <= times[i]) {
      const span = times[i] - times[i - 1] || 1;
      const f = (t - times[i - 1]) / span;
      const a = points[i - 1];
      const b = points[i];
      return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, z: a.z + (b.z - a.z) * f };
    }
  }
  return points[points.length - 1];
}
```

5. `truncateTrajectory` returns prefix points with `times[i] <= t`, then appends the interpolated position unless it duplicates the last prefix point:

```typescript
export function truncateTrajectory(
  trajectory: Trajectory,
  t: number,
): { points: Vec3[]; position: Vec3 | null } {
  const position = trackPositionAt(trajectory, t);
  if (position == null) return { points: [], position: null };
  const prefix = trajectory.points.filter((_, i) => trajectory.times[i] <= t);
  const last = prefix[prefix.length - 1];
  if (last == null || last.x !== position.x || last.y !== position.y || last.z !== position.z) {
    prefix.push(position);
  }
  return { points: prefix, position };
}
```

6. Refactor `sceneOfTracks` to compute axes via an internal `axesOf(rawPoints, kind, trajectories)` helper, then add:

```typescript
export function sceneOfTracksUpTo(
  tracks: { name: string; trajectory: Trajectory }[],
  t: number,
): TrajectoryScene | null {
  const full = sceneOfTracks(tracks);
  if (full == null) return null;
  const colorOf = new Map(full.series.map((s) => [s.name, s.color]));
  // rebuild each series from truncateTrajectory, normalized with full.axes via tickFraction
  // markers = one start marker per non-empty prefix + one position marker per vehicle
  // legend = full.legend (unchanged)
}
```

Normalize each truncated raw point with `tickFraction(value, full.axes[i].min, full.axes[i].max)`. Map truncated series back to names in `full.series` order; skip vehicles whose prefix is empty (no dot either — `trackPositionAt` is null only when the trajectory itself is empty, which `sceneOfTracks` already filtered). `shouldResetOrbit` is `(prevKey, nextKey) => prevKey !== nextKey`.

7. Update the four existing `trajectoryOf` `toEqual` literals to include `times` (`[0]` for single-row, `[0, 1]`... — the dropped-row test keeps row index 1, so `times: [1]`), and add `times: [0, 1]` (matching each literal's point count) to every `sceneOf`/`sceneOfTracks` input literal in the file (`GEO_SCENE`, single-point, two-point, multi-vehicle, custom-name, thin-scene, padded tests).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd workbench/web && npm test -- plot3d.test.ts`
Expected: PASS, all tests including the updated literals.

- [ ] **Step 5: Commit**

```bash
git add workbench/web/src/plot3d.ts workbench/web/src/plot3d.test.ts
git commit -m "feat: timed trajectories and scrubbed 3D scenes"
```

---

### Task 2: Slider in ResultsPane

**Files:**
- Modify: `workbench/web/src/ResultsPane.tsx`
- Modify: `workbench/web/src/ResultsPane.test.ts`

**Interfaces:**
- Consumes from Task 1: `sliderTimes`, `sceneOfTracksUpTo`, `Trajectory` with `times` (via existing `trajectoriesFromPlot`, which now carries times).
- Produces: slider state and scrubbed scene consumed by Task 3's `TrajectoryView` props (`scene`, `label`, `resetKey`).
  - `resetKey: string` = `tracks.map((t) => `${t.name}:${t.trajectory.points.length}`).join("|")` — stable across scrub ticks, changes on a new run.

- [ ] **Step 1: Write the failing tests**

Append to `workbench/web/src/ResultsPane.test.ts`:

```tsx
it("scrubs the 3D path with a time slider that defaults to the full run", () => {
  store.setState({
    program: "hyper3",
    lastPlot: {
      columns: ["time", "latx", "lonx", "alt"],
      rows: [
        { time: 0, latx: 28.43, lonx: -80.55, alt: 3000 },
        { time: 1, latx: 28.44, lonx: -80.54, alt: 3200 },
        { time: 2, latx: 28.45, lonx: -80.53, alt: 3400 },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  const slider = host.querySelector('input[type="range"][aria-label="Trajectory time"]') as HTMLInputElement | null;
  expect(slider).not.toBeNull();
  expect(slider!.min).toBe("0");
  expect(slider!.max).toBe("2");
  expect(slider!.value).toBe("2");
  expect(host.textContent).toContain("2 s");
  act(() => {
    slider!.value = "0";
    slider!.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(host.textContent).toContain("0 s");
});

it("hides the slider when the run has a single moment", () => {
  store.setState({
    program: "hyper3",
    lastPlot: {
      columns: ["time", "latx", "lonx", "alt"],
      rows: [{ time: 0, latx: 28.43, lonx: -80.55, alt: 3000 }],
    },
    selectedColumns: [],
  });
  const host = mount();
  expect(host.querySelector('input[type="range"][aria-label="Trajectory time"]')).toBeNull();
});

it("returns the slider to the end when a new run arrives", () => {
  store.setState({
    program: "hyper3",
    lastPlot: {
      columns: ["time", "latx", "lonx", "alt"],
      rows: [
        { time: 0, latx: 1, lonx: 2, alt: 3 },
        { time: 5, latx: 1.1, lonx: 2.1, alt: 4 },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  const slider = host.querySelector('input[type="range"][aria-label="Trajectory time"]') as HTMLInputElement | null;
  act(() => {
    slider!.value = "0";
    slider!.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(host.textContent).toContain("0 s");
  act(() => {
    store.setState({
      lastPlot: {
        columns: ["time", "latx", "lonx", "alt"],
        rows: [
          { time: 0, latx: 1, lonx: 2, alt: 3 },
          { time: 9, latx: 1.2, lonx: 2.2, alt: 5 },
        ],
      },
    });
  });
  expect(host.textContent).toContain("9 s");
});
```

Note: React 18 `onChange` on range inputs in happy-dom responds to `input` events dispatched with the value set directly on the element, matching the existing `toggle.click()` patterns in this file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd workbench/web && npm test -- ResultsPane.test.ts`
Expected: FAIL — no `input[type="range"]` in the document.

- [ ] **Step 3: Write the minimal implementation**

In `workbench/web/src/ResultsPane.tsx`:

```tsx
import { sceneOfTracks, sceneOfTracksUpTo, sliderTimes, trajectoriesFromPlot } from "./plot3d";

// inside ResultsPane, after tracks/scene memos:
const times = useMemo(() => sliderTimes(tracks), [tracks]);
const [timeIndex, setTimeIndex] = useState(times.length - 1);
useEffect(() => {
  setTimeIndex(times.length - 1);
}, [lastPlot]);
const clamped = times.length === 0 ? 0 : Math.min(Math.max(timeIndex, 0), times.length - 1);
const moment = times[clamped];
const viewScene = useMemo(
  () => (scene == null || times.length < 2 ? scene : sceneOfTracksUpTo(tracks, moment)),
  [scene, tracks, times, moment],
);
const resetKey = useMemo(
  () => tracks.map((t) => `${t.name}:${t.trajectory.points.length}`).join("|"),
  [tracks],
);
```

Render below the trajectory view (replacing `<TrajectoryView scene={scene} .../>` with `viewScene`):

```tsx
{viewScene != null && trajectoryKind != null ? (
  <>
    <TrajectoryView scene={viewScene} label={trajectoryLabel} resetKey={resetKey} />
    {times.length >= 2 ? (
      <div className="mb-3">
        <label className={CAPTION_CLASS} htmlFor="trajectory-time">
          time {moment} s
        </label>
        <input
          id="trajectory-time"
          type="range"
          min={0}
          max={times.length - 1}
          value={clamped}
          step={1}
          aria-label="Trajectory time"
          className="w-full"
          onChange={(event) => setTimeIndex(Number(event.target.value))}
        />
      </div>
    ) : null}
  </>
) : null}
```

Keep passing the full `scene` when `times.length < 2` so single-moment runs render exactly as before. `TrajectoryView` gains the `resetKey` prop in Task 3 — add the prop to its signature in this task if implementing sequentially in one pass, but the Task 3 tests own its behavior.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd workbench/web && npm test -- ResultsPane.test.ts`
Expected: PASS. (TypeScript error on the new `resetKey` prop is resolved in Task 3; if running `npm run build` mid-task, expect that single error until Task 3 lands.)

- [ ] **Step 5: Commit**

```bash
git add workbench/web/src/ResultsPane.tsx workbench/web/src/ResultsPane.test.ts
git commit -m "feat: time slider scrubs the 3D trajectory"
```

---

### Task 3: Position dots and orbit preservation in TrajectoryView

**Files:**
- Modify: `workbench/web/src/TrajectoryView.tsx`
- Modify: `workbench/web/src/ResultsPane.test.ts` (append one test)
- Modify: `UPDATES.md` (new `0.189.0` entry on top)

**Interfaces:**
- Consumes from Task 1: `scene.markers` with `kind === "position"` carrying `color`; `shouldResetOrbit`.
- Consumes from Task 2: `resetKey` prop value.
- Produces: `{ scene: TrajectoryScene; label: string; resetKey: string }` component that keeps the user's orbit across scrub ticks.

- [ ] **Step 1: Write the failing tests**

`plot3d.test.ts` coverage for position markers already landed in Task 1. Append to `workbench/web/src/ResultsPane.test.ts`:

```tsx
it("keeps the orbit canvas mounted while scrubbing", () => {
  store.setState({
    program: "hyper3",
    lastPlot: {
      columns: ["time", "latx", "lonx", "alt"],
      rows: [
        { time: 0, latx: 1, lonx: 2, alt: 3 },
        { time: 1, latx: 1.1, lonx: 2.1, alt: 4 },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  const before = host.querySelector("canvas");
  expect(before).not.toBeNull();
  const slider = host.querySelector('input[type="range"][aria-label="Trajectory time"]') as HTMLInputElement | null;
  act(() => {
    slider!.value = "0";
    slider!.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(host.querySelector("canvas")).not.toBeNull();
  expect(host.textContent).toContain("Drag to orbit. Scroll to zoom.");
});
```

- [ ] **Step 2: Run the test to verify the prop contract fails**

Run: `cd workbench/web && npm run build`
Expected: FAIL — `TrajectoryView` does not accept `resetKey` (`Property 'resetKey' does not exist`).

- [ ] **Step 3: Write the minimal implementation**

In `workbench/web/src/TrajectoryView.tsx`:

```tsx
export function TrajectoryView({
  scene,
  label,
  resetKey,
}: {
  scene: TrajectoryScene;
  label: string;
  resetKey: string;
}) {
  // ...
  const keyRef = useRef(resetKey);
  useEffect(() => {
    // ... existing setup ...
    if (shouldResetOrbit(keyRef.current, resetKey)) {
      keyRef.current = resetKey;
      orbitRef.current = { azimuth: 45, elevation: 25, distance: Math.max(radius * 3.5, 1) };
    }
    // ... rest of setup, draw() ...
  }, [scene, theme, resetKey]);
```

i.e. replace the unconditional `orbitRef.current = {...}` line with the guarded version; everything else in the effect stays. Then render position markers as spheres next to the existing start/end sprite loop:

```tsx
for (const marker of scene.markers) {
  if (marker.kind === "position") {
    const geometry = new THREE.SphereGeometry(size * 0.9, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: marker.color ?? "#0f766e" });
    const dot = new THREE.Mesh(geometry, material);
    const at = offsetBy(marker.point, center);
    dot.position.set(at.x, at.y, at.z);
    world.add(dot);
    bin.push(geometry, material);
    continue;
  }
  // ... existing sprite code for start/end ...
}
```

Add `shouldResetOrbit` to the `./plot3d` import. Effect dependency array becomes `[scene, theme, resetKey]`.

- [ ] **Step 4: Run all checks to verify they pass**

Run: `cd workbench/web && npm run build`
Expected: PASS (tsc clean).

Run: `cd workbench/web && npm test`
Expected: PASS (full web suite: `plot3d.test.ts`, `ResultsPane.test.ts`, and the rest).

- [ ] **Step 5: Add the UPDATES.md entry and commit**

Prepend to `UPDATES.md`:

```markdown
# Updates

## 0.189.0 - 3D time slider
- A time slider below the orbit view scrubs the run: partial path up to the selected moment plus one position dot per vehicle in its track color. Axes stay on the full-run scale; orbit/zoom survive scrubbing and reset on a new run.
- Tests: `plot3d.test.ts`, `ResultsPane.test.ts`.
```

```bash
git add workbench/web/src/TrajectoryView.tsx workbench/web/src/ResultsPane.test.ts UPDATES.md
git commit -m "feat: position dots and orbit preservation for the 3D time slider"
```

---

## Self-Review

1. **Spec coverage:** 3D-only scope — Tasks 1–3 touch only `plot3d`, `ResultsPane`, `TrajectoryView`; charts/ground track untouched. Shared time domain — `sliderTimes` unions all tracks. Partial path + dots — `truncateTrajectory` + `position` markers. Fixed axes — `sceneOfTracksUpTo` reuses full-run axes. Orbit survives scrub, resets on new run — `resetKey` + `shouldResetOrbit`. Slider hidden for <2 moments — Task 2 conditional. Default full run + reset on new run — Task 2 state/effect/tests.
2. **Placeholder scan:** every step has exact code, exact commands, exact expected output. No TBD/TODO/edge-case hand-waving; clamping, empty-trajectory nulls, and the single-moment fallback are specified inline.
3. **Type consistency:** `Trajectory.times` introduced once in Task 1 and threaded through Tasks 2–3; marker `color` only on `position`; `resetKey` produced in Task 2, consumed in Task 3 with the same `string` type; existing-test literal updates enumerated in Task 1 Step 3.
