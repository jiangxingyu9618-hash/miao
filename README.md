# Miao

A desktop cat that eats, sulks and sleeps. Your webcam's hand gestures drive it. Windows 10/11, x64.

## Run

Download `miao-portable-v1.0.0.zip` (208 MB) from [Releases](../../releases/tag/v1.0.0), unzip anywhere, double-click `启动.bat`. For gesture control, use `启动-手势.bat` instead.

No Python required. `runtime\` carries the interpreter and every dependency; deleting it breaks the app. Allow camera access when Windows asks, and close other apps that hold the webcam.

## Gestures

Rule mode is the default.

| Gesture | Action |
| --- | --- |
| Fist | Feed it, affection +5 |
| Open palm | Annoy it, affection -10 |
| Thumb-index pinch | Drag it along with your fingertips |

MLP mode, switched from the right-click menu.

| Gesture | Action |
| --- | --- |
| One-hand heart | Play together |
| Two-hand heart | Go to sleep |
| OK | Pet it, affection +5 |
| Thumbs down | Make it angry, affection -10 |

## From source

Python 3.11, 64-bit.

```
pip install -r requirements.txt
python pet.py
python pet.py --gesture
```

Recognition runs in a separate process (`gesture_service.py`); the main window polls its state file every 30 ms, so mediapipe and Qt never share an address space.

MIT
