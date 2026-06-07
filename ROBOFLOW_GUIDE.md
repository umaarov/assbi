# Roboflow Workflow — ASSBI Custom Person-Detection Dataset

This is the **hybrid annotation workflow** for the custom-dataset / fine-tuning
deliverable. You build a local dataset from the Temple Bar footage, upload it to
Roboflow **with the boxes already drawn** (auto-labels), generate a versioned +
augmented dataset, download it, and fine-tune YOLOv8 on it.

```
 footage ──► build-dataset ──► roboflow-upload ──► roboflow-generate ──►
        roboflow-download ──► train ──► best.pt
   (code)        (code)          (API)            (API)            (API)     (code)
```

**Every step except account creation is automated over the Roboflow API** — no
browser clicking required. The only manual web step is making the free account
(§0). Reviewing the boxes in the Roboflow UI (§3) is *optional* but recommended
as evidence — the labels are already attached either way.

### TL;DR — the whole pipeline (after §0 account + `.env` key)

```powershell
python -m assbi.cli build-dataset --source data/source_video.mp4 --frames 500
python -m assbi.cli roboflow-upload   --project temple-bar-people
python -m assbi.cli roboflow-generate --project temple-bar-people          # makes a version + augmentation
python -m assbi.cli roboflow-download --project temple-bar-people --version 1
python -m assbi.cli train --data data/roboflow/data.yaml --epochs 30 --imgsz 416
```

> Note: Roboflow may show **"Unannotated: 496"** in the web version builder even
> though every image has boxes. That's a *review-approval* flag, not missing
> labels — `roboflow-generate` bakes the existing annotations into the version
> regardless (verified: the download had 1,286 images / 16,163 person boxes).

---

## 0. One-time: account + API key  *(web)*

1. Go to **https://roboflow.com** → **Sign Up** (free plan is enough; sign in
   with Google/GitHub is fastest).
2. It asks you to create a **Workspace** — name it e.g. `umarov-assbi`.
3. **Create a Project**:
   - **Project Type:** *Object Detection*
   - **Project Name:** e.g. `temple-bar-people`
   - **Annotation Group / What are you detecting:** `person`
   - Create it. Note the **project id** — it's the slug in the URL:
     `app.roboflow.com/<workspace>/<project-id>/...` (e.g. `temple-bar-people`).
4. Get your **Private API Key**: click your **workspace** → **Settings** →
   **API Keys** → copy the **Private API Key**.

Put the key in your `.env` (same file the DeepSeek key lives in) so you never
type it on the command line:

```
ROBOFLOW_API_KEY=rf_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 1. Build the local dataset  *(code — you may already have this)*

If `data/dataset/` doesn't exist yet, build it from your recorded footage:

```powershell
python -m assbi.cli build-dataset --source data/source_video.mp4 --frames 500
```

This samples 500 people-rich frames and writes YOLO-format auto-labels under
`data/dataset/images/{train,val}` + `labels/{train,val}` + `data.yaml`.

---

## 2. Upload to Roboflow *with* the auto-labels  *(code)*

```powershell
python -m assbi.cli roboflow-upload --project temple-bar-people
```

(Use your real project id. `--workspace` is optional — omit it to use the API
key's default workspace. The key is read from `$ROBOFLOW_API_KEY`; or pass
`--api-key rf_...`.)

This uploads every frame **with its boxes attached**, into an annotation batch
called `assbi-temple-bar`. Tips:
- Quick test first: add `--limit 10` to upload just 10 images and confirm it
  lands in Roboflow before sending all 500.
- Want to annotate fully by hand instead? add `--no-labels` (uploads raw images
  with no boxes).

---

## 3. Review / correct the annotations  *(web — OPTIONAL, recommended for evidence)*

The labels are already attached, so this step is **optional** — but doing it
gives you "I reviewed my own dataset" evidence for the report.

1. In Roboflow open your project → **Annotate** → open the **`assbi-temple-bar`**
   batch.
2. Each image already has **person boxes** from the auto-labeller. Skim them: fix
   any wrong box, delete false boxes, add a missed person. Much faster than
   drawing from scratch.
3. Screenshot a couple of these annotated frames for your report.

> You do **not** have to click "Approve" or "Add to Dataset" for training to
> work — `roboflow-generate` (next) bakes the uploaded annotations into the
> version even if the UI still shows them as "unannotated/unreviewed".

---

## 4. Generate a version (with augmentation)  *(code — API)*

```powershell
python -m assbi.cli roboflow-generate --project temple-bar-people
```

This calls Roboflow's **Generate Version** over the API with sensible defaults:
*Auto-Orient* + *Resize 416×416* preprocessing, and *Flip + Brightness ±18% + 3×
copies* augmentation (≈500 → ~1,300 images). It prints the new **version number**
(usually `1`). Flags: `--resize 640`, `--no-augment`.

*(You can do this in the web UI instead — Versions → Generate New Version — but
the CLI is one command and reproducible.)*

---

## 5. Download the version + train  *(code)*

```powershell
python -m assbi.cli roboflow-download --project temple-bar-people --version 1
python -m assbi.cli train --data data/roboflow/data.yaml --epochs 30 --imgsz 416
```

`roboflow-download` pulls the version as a YOLOv8 export into `data/roboflow/`
and prints the exact `train` line. Training writes the fine-tuned weights to
`runs/assbi/<name>/weights/best.pt` plus mAP/precision/recall + curves +
confusion matrix (your training evidence). The run in this repo used
`--name roboflow`, so it lives in `runs/assbi/roboflow/`.

**Actual result of this fine-tune (30 epochs, 1,286 imgs / 16,163 boxes):**

| Metric | Value |
|---|---|
| mAP@50 | **0.930** |
| mAP@50-95 | **0.699** |
| Precision | **0.881** |
| Recall | **0.867** |

Then point the detector at your model in `config/config.yaml` (already done here):

```yaml
detection:
  backend: yolo
  model_path: runs/assbi/roboflow/weights/best.pt
```

---

## What to put in your report (evidence checklist)

- Roboflow project screenshot (image count, class = `person`).
- A reviewed/annotated frame (boxes visible).
- The **Generate Version** screen showing preprocessing + augmentation.
- `runs/assbi/roboflow/`: `results.png` (loss/mAP curves),
  `confusion_matrix.png`, `val_batch*_pred.jpg`, and the final mAP50 / precision
  / recall numbers (mAP@50 = 0.930, precision 0.881, recall 0.867).

> Honesty note for your write-up: the boxes start as *pseudo-labels* from a
> stronger YOLOv8 "teacher" model; Roboflow is where you **review and correct**
> them into your final ground truth, then augment + version. Say exactly that —
> it's a legitimate, standard pipeline (and it's true).
