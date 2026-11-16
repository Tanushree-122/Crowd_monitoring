# VISORA — Crowd Intelligence Dashboard

VISORA is a local web application for estimating crowd counts in video. A
browser-based dashboard sends a selected video to a Flask API. The backend
samples frames, runs CSRNet crowd-density inference, and returns estimates,
summary statistics, a density map, and threshold-based alerts.

> **Important:** Crowd counts are model estimates, not ground-truth counts.
> Their accuracy depends on the checkpoint and how closely its training data
> matches the camera footage. Do not use this academic/demo project as a
> safety-critical system.

## Features

- Upload and preview MP4, AVI, MOV, MKV, and WEBM videos
- CSRNet-based density estimation and heatmap generation
- Current, peak, average, and minimum count summaries
- Configurable density levels and alerts
- Automatic CPU/CUDA device selection by PyTorch
- REST API and a responsive, static HTML/CSS/JavaScript dashboard

## Requirements

- Windows, macOS, or Linux
- Python **3.11** recommended; Python **3.12.10** was also verified with the pinned dependencies
- pip and a modern browser
- Approximately 2 GB of free space for the Python environment and model
- A compatible CSRNet checkpoint for video analysis (see [Model weights](#model-weights-required-for-analysis))

CPU inference works, though it may be slow. An NVIDIA GPU with a compatible
CUDA-enabled PyTorch installation can speed up inference. No Node.js or
frontend package installation is required.

## Get the project

After pushing this project to GitHub, replace the example account and
repository below with your GitHub details:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY/Crowd_monitoring
```

If `Crowd_monitoring` is the repository root, omit `/Crowd_monitoring` from
the `cd` command. Alternatively, download the repository ZIP from GitHub and
extract it.

## Install the backend

Create a virtual environment from the `Crowd_monitoring` directory:

### Windows PowerShell

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pinned dependencies include Flask, OpenCV, NumPy, PyTorch, and
TorchVision. Installing PyTorch can take a few minutes and requires a
reliable internet connection.

## Model weights (required for analysis)

The pretrained model is a large binary file and is excluded from Git by
`.gitignore`. A fresh clone will **not** contain the checkpoint, even if it
exists in the project owner's local folder. The dashboard can open without
the weights, but analyzing a video requires a compatible checkpoint.

1. Obtain a pretrained **PyTorch CSRNet** checkpoint from a trusted source.
   It must match the CSRNet architecture in `backend/csrnet.py` (VGG16
   frontend, dilated-convolution backend). A checkpoint for another model or
   a structurally different CSRNet will not work.
2. Save or rename the checkpoint to:

   ```text
   backend/models/csrnet.pth
   ```

3. Keep the checkpoint out of Git unless you have permission to redistribute
   it and have checked GitHub's file-size limits. The repository provides
   `backend/models/PLACE_WEIGHTS_HERE.txt` as a reminder.

Check the checkpoint's license and source before using or sharing it. Model
results are only meaningful when the weights are compatible and suitable for
the footage. The first health check loads the model and can take some time.

## Run VISORA

Keep the backend virtual environment activated. Open **two terminals** from
the `Crowd_monitoring` directory.

### Terminal 1 — Flask API

```bash
cd backend
```

Activate `.venv` in this terminal if it is not already active, then run:

```bash
python app.py
```

The API listens on `http://127.0.0.1:5000`. Check that it is responding:

```text
http://127.0.0.1:5000/api/health
```

The health response reports whether the server is running and whether the
CSRNet model loaded. A missing checkpoint does not prevent the server from
starting, but video analysis will fail until compatible weights are added.

### Terminal 2 — Frontend

From the project directory, serve the static frontend:

```bash
cd frontend
python -m http.server 8000
```

Open **http://127.0.0.1:8000** in your browser. Leave both terminals running
while using the app. To stop either server, press `Ctrl+C` in its terminal.

The frontend calls the backend at `http://127.0.0.1:5000`, configured by
`API_BASE` near the top of `frontend/script.js`. If you change the API host or
port, update that value too. Flask CORS is enabled for local frontend access.

## Analyze a video

1. Confirm the API is running and `/api/health` reports `model_status:
   "loaded"`.
2. Open `http://127.0.0.1:8000`.
3. Choose a supported video no larger than **200 MB**.
4. Select **Start Analysis** and wait for the results.
5. Use **Reset** before starting another analysis.

The backend analyzes a sampled subset of frames. By default it processes
every 60th frame and stops after 150 sampled frames. These settings are in
`backend/config.py`; changing them affects processing time and temporal
coverage. It creates upload and output directories automatically.

## Configuration

Edit `backend/config.py` and restart Flask after changing settings:

| Setting | Default | Description |
|---|---:|---|
| `FRAME_SAMPLE_INTERVAL` | `60` | Analyze every Nth video frame. |
| `MAX_FRAMES_TO_PROCESS` | `150` | Maximum number of sampled frames per video. |
| `FRAME_RESIZE_MAX_DIM` | `640` | Maximum image dimension passed to CSRNet. |
| `MAX_CONTENT_LENGTH_MB` | `200` | Maximum upload size. |
| `DENSITY_THRESHOLDS` | `20`, `50`, `100` | Count boundaries for LOW, MEDIUM, HIGH, and CRITICAL levels. |

`FRAME_SAMPLE_INTERVAL` and `MAX_FRAMES_TO_PROCESS` can also be set with
environment variables before starting Flask:

```bash
FRAME_SAMPLE_INTERVAL=30 MAX_FRAMES_TO_PROCESS=200 python app.py
```

On Windows PowerShell:

```powershell
$env:FRAME_SAMPLE_INTERVAL = "30"
$env:MAX_FRAMES_TO_PROCESS = "200"
python app.py
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Server and model status |
| `POST` | `/api/upload` | Upload a video |
| `POST` | `/api/analyze` | Analyze an uploaded video |
| `GET` | `/api/status` | Read current processing status |
| `POST` | `/api/reset` | Reset in-memory analysis state |
| `GET` | `/outputs/density_maps/<filename>` | Retrieve a generated density map |

The dashboard sends the video directly to `/api/analyze` as multipart form
data. The API also supports uploading first and then analyzing a stored
filename. See `backend/app.py` for request and response details.

## Troubleshooting

| Problem | What to check |
|---|---|
| `python` or `py` is not recognized | Install Python 3.11 (recommended) or 3.12 and enable the option to add it to PATH. On Windows, `py -3.11` selects Python 3.11. |
| `Activate.ps1` is blocked | Use the Command Prompt activation instructions above, or follow your organization's PowerShell policy. |
| `No module named ...` | Activate the backend `.venv`, then run `pip install -r requirements.txt` from `backend`. |
| CSRNet checkpoint not found | Place the compatible file at `backend/models/csrnet.pth`. |
| Checkpoint incompatible | Confirm it is for the architecture in `backend/csrnet.py`; changing the filename alone does not make a checkpoint compatible. |
| Backend not reachable / CORS error | Keep Flask running on port 5000 and open the frontend through `http://127.0.0.1:8000`, not by double-clicking `index.html`. |
| Port already in use | Stop the other process using port 5000 or 8000, or configure the server and frontend API URL to use another port. |
| Video cannot be opened | Try another supported video/container or re-encode it with a broadly supported codec such as H.264. |
| No frames processed | Check that the video is valid and that its duration/frame count is sufficient for the configured sampling interval. |
| Upload exceeds the limit | Use a smaller video or change `MAX_CONTENT_LENGTH_MB` in `backend/config.py`. |
| Analysis is slow | CPU inference can be slow. Increase the frame sampling interval, lower the resize limit, or use a correctly installed CUDA-enabled PyTorch build. |

## Limitations and evaluation

- Counts are density estimates for individual sampled frames; this project
  does not track people or count unique visitors.
- Sparse sampling can miss short-lived crowd changes.
- Camera angle, image quality, crowd density, checkpoint, and training-domain
  differences all affect estimates.
- Density and safety labels are based on configurable count thresholds; they
  are not certified safety guidance.
- The Flask app keeps analysis state in memory and is intended for local
  demonstrations, not production or multi-user deployment.
- MAE/MSE evaluation is meaningful only on annotated data with matching
  ground-truth counts. Do not report estimates from unannotated videos as
  accuracy measurements. `evaluate_against_ground_truth()` in
  `backend/crowd_analyzer.py` computes MAE and MSE for aligned count lists.

## Project structure

```text
Crowd_monitoring/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── crowd_analyzer.py
│   ├── csrnet.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   ├── models/                 # Add csrnet.pth here (not included in Git)
│   └── utils/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── outputs/                    # Generated density maps and videos
└── README.md
```
