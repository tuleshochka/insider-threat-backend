import os
import sys
import time
import requests
import argparse
from pathlib import Path
import pandas as pd

API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1/events")

def get_latest_rows(file_path, nrows=1000):
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def send_event(anon_id, event_type, details):
    payload = {
        "anon_id": anon_id,
        "event_type": event_type,
        "details": details
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=2)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send event: {e}")
        return False

def stream_logon(file_path, delay=1.0):
    print(f"Streaming logon events from {file_path}")
    df = get_latest_rows(file_path)
    for _, row in df.iterrows():
        # Map dataset columns to our generic event details
        anon_id = str(row.get('user', 'unknown'))
        details = {
            "id": str(row.get('id', '')),
            "date": str(row.get('date', '')),
            "pc": str(row.get('pc', '')),
            "activity": str(row.get('activity', ''))
        }
        if send_event(anon_id, "logon", details):
            print(f"Sent logon event for user {anon_id}")
        time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="Stream logs to UEBA API")
    parser.add_argument("--dataset-dir", required=True, help="Path to CERT r4.2 dataset directory")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between events in seconds")
    parser.add_argument("--type", choices=["logon", "file", "http"], default="logon", help="Event type to stream")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Directory {dataset_dir} does not exist.")
        sys.exit(1)

    file_map = {
        "logon": "logon.csv",
        "file": "file.csv",
        "http": "http.csv"
    }

    target_file = dataset_dir / file_map[args.type]
    
    if args.type == "logon":
        stream_logon(target_file, delay=args.delay)
    else:
        print(f"Streaming for {args.type} not fully implemented yet.")

if __name__ == "__main__":
    main()
