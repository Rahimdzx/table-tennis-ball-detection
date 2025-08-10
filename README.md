# Table Tennis Ball Detection

![Demo GIF](media/demo.gif)

This project implements a computer vision system for detecting and analyzing table tennis ball trajectories and key events from video footage. It uses the Ultralytics YOLOv8 model for ball detection, along with custom logic to track ball bounces, net hits, and update player scores in real time.

---

## Features

- Detects table tennis ball position in video frames using a trained YOLOv8 model.
- Tracks the ball trajectory over time.
- Detects key game events such as bounces on either player’s side, net hits, and out-of-bound events.
- Displays scores and game events visually on the video frames.
- Supports interactive setup to define table corners and net position.
- Saves annotated output video with detected events.

---

## Repository Structure

├── configs/
│ └── data.yaml # Dataset configuration
├── Dataset/ # Images and annotations for training/validation
├── Models/ # Trained model weights
│ └── best.pt
├── Scripts/
│ ├── analyze_tennis_game.py # Main analysis script
│ ├── evaluate_model.py # Script to evaluate model performance
│ └── train_model.py # Training script for the YOLOv8 model
├── Videos/ # (Not included due to large file size)
├── media/ # GIF/demo images
├── requirements.txt # Python dependencies
└── README.md # This file

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rahimdzx/table-tennis-ball-detection.git
   cd table-tennis-ball-detection
Create and activate a Python virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate       # Linux/macOS
   .\venv\Scripts\activate        # Windows PowerShell
Install the required packages:
   pip install -r requirements.txt
   Contact : 
For questions or support, please contact: mouissatrabah@gmail.com
