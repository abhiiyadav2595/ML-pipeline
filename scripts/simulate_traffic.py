"""
Traffic & Drift Simulation Utility.
Generates live HTTP requests to the ML Model Serving endpoint.
Supports:
  - Normal traffic mode (baseline distribution).
  - Drift traffic mode (macroeconomic shock: high DTI, lower credit score, unemployment shift).
  - Spike traffic mode.
"""

import argparse
import os
import random
import sys
import time
from typing import Any, Dict

import requests

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.generate_data import generate_credit_dataset


def wait_for_api(base_url: str, max_retries: int = 30, delay: float = 2.0) -> bool:
    """Wait for ML API to become responsive."""
    health_url = f"{base_url.rstrip('/')}/health"
    print(f"Waiting for API service at {health_url}...")
    for i in range(max_retries):
        try:
            resp = requests.get(health_url, timeout=3)
            if resp.status_code == 200:
                print("API service is ready!")
                return True
        except Exception:
            pass
        time.sleep(delay)
    print("Warning: API service did not respond within timeout. Proceeding anyway...")
    return False


def send_inference(base_url: str, record: Dict[str, Any]) -> bool:
    """Send single prediction request."""
    record_clean = {k: v for k, v in record.items() if k != "default"}
    url = f"{base_url.rstrip('/')}/predict"
    try:
        resp = requests.post(url, json={"features": record_clean}, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        print(f"Error calling {url}: {e}")
        return False


def run_simulation(
    base_url: str = "http://localhost:8000",
    mode: str = "normal",
    num_requests: int = 200,
    delay_sec: float = 0.05,
):
    """
    Generate traffic based on selected mode.
    """
    print("\n=======================================================")
    print(" Starting Traffic Simulation")
    print(f" Target API: {base_url}")
    print(f" Mode: {mode.upper()}")
    print(f" Requests: {num_requests} | Delay: {delay_sec}s")
    print("=======================================================\n")

    is_drift = mode in ["drift", "shock"]
    df = generate_credit_dataset(
        n_samples=num_requests,
        random_state=random.randint(100, 9999),
        drift=is_drift,
    )

    records = df.to_dict(orient="records")
    successes = 0

    for i, rec in enumerate(records, 1):
        ok = send_inference(base_url, rec)
        if ok:
            successes += 1
        if i % 25 == 0 or i == num_requests:
            print(f"Progress: [{i}/{num_requests}] requests sent ({successes} successful)...")
        if delay_sec > 0:
            time.sleep(delay_sec)

    print(f"\nSimulation complete: {successes}/{num_requests} successful.")

    # Trigger explicit drift evaluation
    try:
        eval_url = f"{base_url.rstrip('/')}/drift/evaluate"
        print(f"Triggering on-demand drift evaluation at {eval_url}...")
        resp = requests.post(eval_url, timeout=10)
        if resp.status_code == 200:
            drift_data = resp.json()
            print("\n---------------- Drift Summary ----------------")
            print(f"Sample Size: {drift_data.get('sample_size')}")
            print(f"Dataset Drift Detected: {drift_data.get('dataset_drift_detected')}")
            print(f"Drift Ratio: {drift_data.get('drift_ratio') * 100:.1f}%")
            print(f"Drifted Features: {drift_data.get('drifted_features_count')}/{drift_data.get('total_features_count')}")
            print("-----------------------------------------------")
    except Exception as e:
        print(f"Could not trigger drift evaluation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Serving Traffic and Drift Simulator")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of ML API")
    parser.add_argument("--mode", type=str, choices=["normal", "drift"], default="normal", help="Simulation mode")
    parser.add_argument("--count", type=int, default=150, help="Number of requests")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between requests in seconds")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop alternating modes")

    args = parser.parse_args()

    # Ensure API is ready before starting loop or simulation
    wait_for_api(args.url)

    if args.loop:
        print("Running in continuous loop mode (alternates normal and drift traffic)...")
        while True:
            run_simulation(args.url, mode="normal", num_requests=args.count, delay_sec=args.delay)
            time.sleep(5)
            run_simulation(args.url, mode="drift", num_requests=args.count, delay_sec=args.delay)
            time.sleep(10)
    else:
        run_simulation(args.url, mode=args.mode, num_requests=args.count, delay_sec=args.delay)
