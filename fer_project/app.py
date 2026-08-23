"""Root entry point for webcam + backend pipeline: python app.py"""
from src.inference.webcam_stream import run_webcam

if __name__ == "__main__":
    run_webcam(session_id="demo-session")
