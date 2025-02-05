#!/usr/bin/env python3

import argparse
import re
import webbrowser
import sys

def extract_meta_id(pod_name):
    """Extract metaId from pod name using the pattern: [pipeline]-[dataset]-[date]-[time]-[config][-][0-4digits]f."""
    # Pattern matches: pipeline-dataset-date-time-config[-][N]f
    # e.g., lsim-u100x-241225-1558-e6i005msf16g4f or lsim-u100x-241225-1558-e6i005msf16g4-433f
    pattern = r'([a-zA-Z]+)-([a-zA-Z0-9]+)-(\d{6})-(\d{4})-([a-zA-Z0-9]+)(?:-\d{0,4}|\d{0,4})f'
    match = re.search(pattern, pod_name)
    
    if not match:
        print(f"Error: Could not find required pattern in pod name: {pod_name}")
        print("Expected pattern: [pipeline]-[dataset]-[YYMMDD]-[HHMM]-[config][-][0-4digits]f")
        print("Examples: lsim-u100x-241225-1558-e6i005msf16g4f")
        print("          lsim-u100x-241225-1558-e6i005msf16g4-f")
        print("          lsim-u100x-241225-1558-e6i005msf16g4-5f")
        print("          lsim-u100x-241225-1558-e6i005msf16g4-433f")
        print("          lsim-u100x-241225-1558-e6i005msf16g4-4339f")
        sys.exit(1)
    
    return match.group(0)

def convert_to_grafana_format(meta_id):
    """Convert meta_id from hyphenated format to underscore format for Grafana."""
    return meta_id.replace('-', '_')

def main():
    parser = argparse.ArgumentParser(description='Open Grafana dashboard for a specific pod')
    parser.add_argument('pod_name', help='Pod name or prefix (e.g., lsim-u100x-241225-1558-e6i005msf16g4f or lsim-u100x-241225-1558-e6i005msf16g4-433f)')
    args = parser.parse_args()

    # Extract meta ID from pod name
    meta_id = extract_meta_id(args.pod_name)
    
    # Convert to Grafana format (replace hyphens with underscores)
    grafana_meta_id = convert_to_grafana_format(meta_id)
    
    # Construct Grafana URL
    base_url = "http://node11.bdcl:30833/d/de7qag4j0qsqoe/tagged-spark-jvm-profile"
    params = {
        "orgId": "1",
        "from": "now-24h",
        "to": "now",
        "timezone": "browser",
        "var-appId": "$__all",
        "var-executorId": "$__all",
        "refresh": "5s",
        "var-metaId": grafana_meta_id
    }
    
    url = f"{base_url}?" + "&".join(f"{k}={v}" for k, v in params.items())
    
    print(f"Extracted pattern: {meta_id}")
    print(f"Grafana meta ID: {grafana_meta_id}")
    print(f"Opening URL: {url}")
    
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Error opening browser: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
