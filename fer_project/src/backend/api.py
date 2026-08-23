"""Sends emotion probabilities + timestamps to backend (Phase 10)."""
import requests
from src.config_loader import load_config


def send_emotion_snapshot(session_id, timestamp, probabilities):
    cfg = load_config()
    url = cfg.backend.api_base_url + cfg.backend.emotion_endpoint.format(session_id=session_id)
    payload = {"timestamp": timestamp, "probabilities": probabilities}
    try:
        requests.post(url, json=payload, timeout=cfg.backend.timeout_seconds)
    except requests.RequestException as e:
        print(f"[backend] Failed to send emotion snapshot: {e}")
