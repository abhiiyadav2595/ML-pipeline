# PowerShell script to open all project localhost web services in your browser
Write-Host "Opening all project localhost web services in your browser..." -ForegroundColor Cyan

Start-Process "http://localhost:3000"
Start-Process "http://localhost:8000/docs"
Start-Process "http://localhost:8000/drift/report"
Start-Process "http://localhost:9090/alerts"

Write-Host "All hosts opened successfully!" -ForegroundColor Green
