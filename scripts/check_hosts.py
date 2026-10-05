"""
Localhost Endpoints Health Checker.
Run from terminal to verify all running project hosts and ports.
"""

import urllib.request
import sys

ENDPOINTS = [
    ("Grafana Dashboard", "http://localhost:3000", "Visual metrics & drift graphs"),
    ("FastAPI Swagger UI", "http://localhost:8000/docs", "Interactive API testing"),
    ("HTML Drift Report", "http://localhost:8000/drift/report", "Real-time statistical drift report"),
    ("API Health Endpoint", "http://localhost:8000/health", "Liveness & readiness probe"),
    ("Prometheus Metrics", "http://localhost:8000/metrics", "Scraped operational & drift gauges"),
    ("Prometheus Alerts", "http://localhost:9090/alerts", "Drift and performance alerting rules"),
]

def main():
    print("\n" + "=" * 75)
    print(" [*] Checking All ML Pipeline Localhost Services")
    print("=" * 75)
    
    all_ok = True
    for name, url, desc in ENDPOINTS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=3) as res:
                code = res.getcode()
                status = f"\033[92m[ONLINE - {code}]\033[0m"
        except Exception as e:
            status = f"\033[91m[OFFLINE]\033[0m"
            all_ok = False
            
        print(f" {status:<22} {name:<26} -> {url}")
        print(f"   |-> {desc}")

    print("=" * 75 + "\n")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
