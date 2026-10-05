@echo off
echo Opening all ML Pipeline localhost web services in your browser...
start http://localhost:3000
start http://localhost:8000/docs
start http://localhost:8000/drift/report
start http://localhost:9090/alerts
echo All hosts opened successfully!
