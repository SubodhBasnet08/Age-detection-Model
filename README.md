# Age Detection Model App

This runnable app follows the project proposal: people can usually describe a face as young, middle-aged, or old, and this app automates that process with a deep learning age detection model. It uses Python, OpenCV face detection, OpenCV DNN model loading, a pre-trained Caffe age model, and young/middle-aged/old output display.

## Run

```powershell
& 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Models

OpenCV DNN can import models from well-known deep learning frameworks. This project uses the available pre-trained Caffe age model files in `models/`:

- `age_deploy.prototxt`
- `age_net.caffemodel`

The Caffe model predicts one of eight age ranges, and the app maps those ranges into three project categories: young, middle-aged, or old. The folder includes the Caffe age model, so this implementation focuses on age detection. Face-based age detection predicts ranges and groups, not exact ages.
