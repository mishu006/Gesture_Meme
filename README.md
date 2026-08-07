# Gesture Meme

A computer vision project that detects facial and hand gestures in real time using your webcam and displays a corresponding meme.

## Technologies

- **Python 3.10.0**
- **OpenCV**
- **MediaPipe**
- **NumPy**

## Requirements

- **Python 3.8 - 3.11** (MediaPipe does not support newer versions)

### 1. Create and activate a virtual environment

It's recommended to install dependencies inside a virtual environment to keep your project isolated from your global Python installation.

**Create the virtual environment** (run once, from the project folder):

    python3 -m venv venv

**Activate it:**

- **macOS / Linux:**

      source venv/bin/activate

- **Windows (Command Prompt):**

      venv\Scripts\activate.bat

- **Windows (PowerShell):**

      venv\Scripts\Activate.ps1

Once activated, your terminal prompt should show `(venv)` at the beginning of the line. To deactivate it later, simply run `deactivate`.

### 2. Install dependencies

With the virtual environment active, install the dependencies:

    pip install opencv-python mediapipe numpy

Or, if you have the requirements file:

    pip install -r requirements.txt

## Detected gestures

| Gesture | Meme |
|-------|------|
| **Raised or furrowed eyebrows** | `perro.jpeg` |
| **Tongue out** | `gato1.png` |
| **Finger touching the mouth** | `cristiano.png` |
| **Two hands on either side of the face** | `cara.jpeg` |
| **Two hands above the nose** | `Sonic.jpeg` |
| **Index and middle fingers extended** | `rata.jpeg` |

## Project structure

    gesture_meme/
    ├── main.py
    ├── requirements.txt
    ├── cara.jpeg
    ├── cristiano.png
    ├── gato1.png
    ├── perro.jpeg
    ├── rata.jpeg
    └── Sonic.jpeg

## Usage

1. Clone the repository
2. Create and activate a virtual environment (see above)
3. Install the dependencies
4. Place the images in the same folder as `main.py`
5. Run:

       python main.py

6. When it starts, look straight ahead with a neutral face during **calibration**
7. Once calibrated, try the gestures in front of the camera
8. Press **ESC** to exit

## Notes

- **Calibration** takes a few seconds at startup and is necessary for the gestures to work correctly
- The images must be in the **same folder** as `main.py`
- Works best with **good lighting**
- Cross-platform compatible: automatically selects the appropriate camera backend for **Windows**, **macOS**, and **Linux**