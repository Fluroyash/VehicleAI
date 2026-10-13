# VehicleAI - Vehicle Detection & Traffic Analytics

VehicleAI is a computer vision project that detects and tracks vehicles in traffic videos and provides traffic analytics through an interactive Streamlit dashboard.

## Features

- Vehicle detection using YOLO26n
- Multi-object tracking using ByteTrack
- Vehicle classification: cars, motorcycles, buses, and trucks
- Vehicle counting across a configurable counting line
- Traffic density classification: Low, Medium, and High
- Average and peak visible vehicle statistics
- Traffic charts and time-interval analysis
- CSV export of crossing events and density statistics
- Annotated video preview and download

## Technology Stack

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- ByteTrack
- Pandas
- Streamlit

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
