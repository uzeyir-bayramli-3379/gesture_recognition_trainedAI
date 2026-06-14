# Gesture Recognition (trainedAI)

A temporal hand-gesture recognizer built from scratch — the sequel to [hand_pose_detector_trainedAI](https://github.com/uzeyir-bayramli-3379).

Where the pose detector classified a **single frame** of hand landmarks (a static pose), this project classifies **motion over time** — a short sequence of frames — using a 1D convolutional network over the time axis. The distinction matters: a wave and a raised, still hand look identical in any single frame; only the movement across frames tells them apart.

No high-level gesture libraries. The pipeline is hand-built end to end: landmark extraction → sequence collection → windowing → a from-scratch 1D-CNN → live inference.

## How it works

The system is split into four stages, one file each.

**1. Collection (`data_collection.py`)**
MediaPipe Hands extracts 21 landmarks per frame; each is normalized relative to the wrist and scaled by the wrist-to-index-MCP distance, giving a 63-value feature vector per frame (same normalization as the pose detector). Frames are recorded as labeled **sequences** — each press of `r` toggles recording a discrete performance, which is tagged with its own `seq_id` so individual takes stay grouped. `n` advances the class label. Output is appended to `gesture_data.csv`.

**2. Windowing (`windowing.py`)**
Variable-length sequences are turned into fixed-shape training examples. Rows are grouped by `seq_id`, then a fixed-length window (T = 30 frames) slides across each sequence with overlap, producing `(N, 30, 63)` features and `(N,)` labels. Windows never cross a sequence boundary. Each window carries the label of its source sequence.

**3. Training (`training.py`)**
The split is done at the **sequence level, not the window level** — whole sequences are held out for validation before windowing, so near-duplicate overlapping windows from the same performance can't leak across train and val. Features are transposed to `(N, 63, 30)` for `Conv1d` (channels = the 63 landmark coordinates, length = time). Trains with cross-entropy + Adam, saves weights to `scuba_model.pth`.

**4. Live inference (`testing_camera.py`)**
A rolling buffer holds the last 30 frames of landmarks. Once full, each new frame slides the buffer and runs the model. A softmax-confidence threshold gates predictions so low-confidence frames don't fire spurious gestures.

## Model

```
Conv1d(63 → 32, kernel=5, stride=2) → ReLU
Conv1d(32 → 32, kernel=3)           → ReLU
AdaptiveAvgPool1d(1)                 (collapses the time axis)
Linear(32 → 3)
```

Three output classes: two gestures and a no-gesture class. The model is deliberately small to suit the dataset size.

## Results

On a held-out validation set (sequences not seen in training):

```
Val accuracy: 0.97

Confusion matrix (rows = true, cols = predicted):
[[ 4  0  0]
 [ 0  4  0]
 [ 0  1 30]]
```

The two gesture classes separate cleanly; the only remaining confusion is between the no-gesture class and a gesture. The validation set is small, so these numbers are best read as "the pipeline works and the gestures are learnable," not as a robustness benchmark.

## Project structure

```
data_collection.py   # webcam → labeled landmark sequences → gesture_data.csv
windowing.py         # groups sequences, slides windows → (N, 30, 63) arrays
training.py          # 1D-CNN, sequence-level split, saves scuba_model.pth
testing_camera.py    # live inference with a rolling frame buffer
gesture_data.csv     # collected dataset
scuba_model.pth      # trained weights
```

## Setup

```bash
pip install torch mediapipe==0.10.9 opencv-contrib-python numpy pandas scikit-learn pillow
```

`mediapipe` is pinned to `0.10.9` (it exposes the `mp.solutions.hands` API this code uses and constrains `protobuf<4`). Do **not** install `tensorflow` in this environment — newer protobuf requirements from it conflict with mediapipe and break the import chain.

## Usage

```bash
# 1. Collect data: 'r' toggles recording a take, 'n' advances the class, 'q' quits
python data_collection.py

# 2. Train (windowing.py is imported by training.py)
python training.py

# 3. Run live recognition
python testing_camera.py
```

An optional GIF-overlay demo (play a looping GIF while a chosen gesture is detected) is included in `testing_camera.py` but commented out — point it at any GIF of your own and uncomment the block to enable it.

## Notes & limitations

- **The no-gesture class is the weakest link.** It needs broad, varied coverage (idle, drift, transitions, near-misses to the real gestures) to avoid false fires in live use; expanding it is the highest-leverage improvement.
- **Window length is tied to frame rate.** T = 30 frames assumes a roughly stable webcam fps; large fps drift changes the real-time duration a window covers.
- **Landmark detection is upstream of everything.** Poor lighting or low contrast means MediaPipe never produces landmarks for the model to see — even, diffuse front lighting helps.
