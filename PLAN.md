# CV Workstream Plan — Group 4 Fitness App

**Owner:** You (CV Lead)
**Tech:** Python (OpenCV, MediaPipe)

---

## Feasibility Assessment

### Feature 1: Body Progress Tracking — FEASIBLE WITH CAVEATS

MediaPipe Pose detects 33 body landmarks in real time and is free/open-source. Multiple open-source projects already use it for body measurement (see references below). However, there's a hard limitation: **2D photos can't give you true circumference measurements** (chest, waist). What you *can* reliably extract are ratio-based measurements — shoulder width relative to hip width, torso-to-leg proportions, etc. These ratios are stable across photos taken at different distances and are enough to show directional progress ("your shoulder-to-waist ratio improved by 8%").

**What works well:**
- Landmark-to-landmark distances (shoulder width, hip width, arm length)
- Ratio-based progress tracking (normalizes away camera distance)
- Side-by-side visual comparison with skeleton overlay

**What's unreliable:**
- Absolute measurements in cm/inches from a single uncalibrated photo
- Circumference estimates (chest, waist) — these require either a reference object for scale or 3D reconstruction
- Accuracy drops with baggy clothing, poor lighting, or partially visible bodies

**Recommendation:** Scope it as **relative progress tracking** (ratios + visual comparison), not absolute body measurement. This is honest, achievable, and still useful.

### Feature 2: Pushup Form & Rep Tracking — HIGHLY FEASIBLE

This is a well-trodden use case for MediaPipe. Google themselves demo pushup/squat counters in their BlazePose blog post. The approach (track elbow angle across frames, count up/down transitions) is simple math on top of reliable landmark data. Multiple tutorials and open-source implementations exist.

**What works well:**
- Rep counting via angle thresholds (elbow angle for pushups)
- Basic form checks (hip sag detection, elbow flare)
- Real-time performance on laptop webcams, even without GPU

**What's tricky:**
- Camera angle matters — side view works best for pushups
- Thresholds need tuning per person (long arms vs short arms)
- MediaPipe is single-person only (fine for this use case)

**Recommendation:** Start here as your warm-up — you'll have a working demo fastest with this feature.

---

## Free APIs & Tools

### CV / Pose Estimation (your domain)

| Tool | Cost | What it does | Link |
|---|---|---|---|
| **MediaPipe Pose** | Free, open-source (Apache 2.0) | 33 body landmarks, real-time, runs on CPU | [google.dev/mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |
| **OpenCV** | Free, open-source (Apache 2.0) | Image/video processing, drawing overlays | [opencv.org](https://opencv.org/) |

### Open-source body measurement references

| Repo | Notes |
|---|---|
| `JavTahir/Live-Measurements-Api` | Flask API: front + side photo → measurements via MediaPipe |
| `farazBhatti/Human-Body-Measurements-using-Computer-Vision` | Single-image anthropometric extraction |
| `ankesh007/Body-Measurement-using-Computer-Vision` | 2D image → real-world body measurements |

### Nutrition / Calories (for the other subteam, but good to know)

| API | Cost | Notes |
|---|---|---|
| **USDA FoodData Central** | 100% free, 1,000 req/hr | 300K+ foods, government-backed, public domain data. Sign up at [fdc.nal.usda.gov](https://fdc.nal.usda.gov/api-key-signup/) |
| **Open Food Facts** | 100% free, no auth needed | 4M+ products, open-source, community-maintained. [world.openfoodfacts.org/data](https://world.openfoodfacts.org/data) |
| **Edamam** | Free tier available | Natural language food parsing ("1 cup of rice" → nutrition). [edamam.com](https://www.edamam.com/) |

### UIUC Dining Hall Data (for the other subteam)

| Source | Notes |
|---|---|
| **xasos/UIUC-API** | Open-source wrapper around UIUC dining data — returns menu items, dining hall info, meals by date |
| **web.housing.illinois.edu/diningmenus** | Official menu page; can be scraped. The Illinois App also has this data |

**Strategy for calories:** Scrape the dining hall menu item names → fuzzy-match against the USDA FoodData Central API to get calorie/macro data. This avoids needing any paid API.

---

## Phase 1: Foundation

**Goal:** Environment ready, approach validated, contracts defined.

- [ ] Set up Python dev environment (`pip install opencv-python mediapipe flask numpy`)
- [ ] Run the MediaPipe Pose "hello world" — draw 33 landmarks on a test image
- [ ] Clone and run `JavTahir/Live-Measurements-Api` to see what's already been built
- [ ] Decide your measurement set: shoulder width ratio, hip width ratio, torso length ratio, arm span ratio
- [ ] Test landmark stability: take 5 photos of yourself in same pose, compare extracted ratios — are they consistent?
- [ ] Define your API contract (JSON schema) and share with the database and frontend subteams
- [ ] Coordinate with frontend on image capture method (webcam via browser → POST to your Flask endpoint)

**Deliverable:** A README defining inputs/outputs:
> *Input:* front-facing photo (JPEG) → *Output:* `{ shoulder_ratio, hip_ratio, torso_ratio, ..., timestamp }`

---

## Phase 2: Prototype

**Goal:** Both features work in isolation, even if rough.

### Pushup Tracker (start here — faster win)
- [ ] Get MediaPipe Pose running on live webcam feed
- [ ] Calculate elbow angle per frame using landmarks 11/13/15 (left) and 12/14/16 (right)
- [ ] Implement rep counter: "down" (angle < ~90°) → "up" (angle > ~160°) transitions
- [ ] Add basic form flag: detect hip sag by checking hip-shoulder-ankle alignment

### Body Measurement
- [ ] Extract landmark distances from a static image (shoulder-to-shoulder, hip-to-hip, shoulder-to-hip)
- [ ] Normalize all measurements as ratios (e.g., shoulder_width / hip_width)
- [ ] Test consistency: same person, different photos → ratios should be within ~5%
- [ ] Handle failure case: return error JSON if pose not detected or confidence is low

**Deliverable:** Two Python scripts that work standalone. Demo to the team.

---

## Phase 3: Core Development

**Goal:** Robust modules with clean interfaces.

### Body Measurement
- [ ] Add input validation: reject photos where MediaPipe confidence < threshold
- [ ] Support comparison mode: two photos in → delta report out
- [ ] Return results as structured JSON matching the database team's schema
- [ ] Add a visual output option: image with skeleton overlay + measurement labels

### Pushup Tracker
- [ ] Tune thresholds using at least 3 different people
- [ ] Add form quality score (0–100) based on: hip alignment, depth, elbow flare
- [ ] Add real-time overlay: skeleton + rep count + "fix your hips" feedback text
- [ ] Package as a class with a clean API

**Deliverable:** Two importable Python modules:
```python
# body_measure.py
def analyze_body(image_path: str) -> dict

# pushup_tracker.py
class PushupTracker:
    def process_frame(self, frame) -> dict
```

---

## Phase 4: Integration

**Goal:** CV code works end-to-end inside the app.

- [ ] Wrap modules in Flask/FastAPI endpoints:
  - `POST /api/body-analyze` — accepts image, returns measurements
  - `WS /api/pushup-stream` — WebSocket for real-time pushup tracking
- [ ] Connect with frontend: webcam/upload UI → your API → response rendered
- [ ] Connect with database: measurement results and rep logs stored with user + timestamp
- [ ] Test full loops:
  - Upload photo → CV processes → result stored → progress chart updated
  - Start pushup session → frames streamed → reps counted → session saved
- [ ] Test across different webcams, lighting, and body types

**Deliverable:** Working API endpoints integrated into the app.

---

## Phase 5: Polish

**Goal:** Smooth UX, documented code, demo-ready.

- [ ] Optimize speed (resize input frames, skip frames in video mode)
- [ ] Clean up the skeleton overlay visuals
- [ ] Add user guidance: "Stand 6 ft from camera, wear fitted clothes, good lighting"
- [ ] Record your section of the demo video
- [ ] Document code and API endpoints for final submission

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Body landmarks not stable enough for ratio tracking | Test early (Phase 1). If ratios vary >10% across same-person photos, switch to silhouette contour approach |
| Pushup detection doesn't generalize across body types | Use relative angles, not absolute positions. Tune with 3+ testers |
| Webcam latency makes real-time tracking choppy | Process every 2nd–3rd frame; downscale to 480p for inference |
| Frontend integration is painful | Define API contract in Phase 1. Both sides build to the same spec |
| Absolute body measurements requested by team | Push back early — 2D photos can't do this reliably without calibration. Ratios are the honest scope |

---

## Dependencies on Other Subteams

| Who | What they need from you | What you need from them |
|---|---|---|
| **Frontend** | API endpoint spec + sample JSON responses | Webcam capture UI that sends frames/images to your API |
| **Database** | JSON schema for measurement records and pushup session logs | Storage endpoints that accept your JSON |
| **You → Both** | Deliver API spec by end of Phase 1 | — |

---

## Reference Links

- [MediaPipe Pose Landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [BlazePose blog post (Google Research)](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/)
- [Mr. Pose — pushup counter tutorial](https://logessiva.medium.com/an-easy-guide-for-pose-estimation-with-googles-mediapipe-a7962de0e944)
- [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide/)
- [Open Food Facts API](https://world.openfoodfacts.org/data)
- [xasos/UIUC-API (dining hall data)](https://github.com/xasos/UIUC-API)
- [JavTahir/Live-Measurements-Api](https://github.com/JavTahir/Live-Measurements-Api)
- [Roboflow body measurement guide](https://blog.roboflow.com/body-measurement/)

### Key MediaPipe Landmarks for Pushups
```
11/12 — Shoulders
13/14 — Elbows
15/16 — Wrists
23/24 — Hips
25/26 — Knees
27/28 — Ankles
```
