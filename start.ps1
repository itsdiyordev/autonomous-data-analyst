$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $Root "backend\.venv-windows"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Install uv first: winget install --id astral-sh.uv -e. Then reopen PowerShell."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Install Node.js 22.12+ or 24 LTS, then reopen PowerShell."
}
Push-Location "$Root\backend"
uv sync --python 3.12
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }
Pop-Location
Push-Location "$Root\frontend"
npm install
if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
Pop-Location
$BackendPath = "$Root\backend".Replace("'", "''")
$FrontendPath = "$Root\frontend".Replace("'", "''")
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$BackendPath'; uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$FrontendPath'; npm run dev"
Write-Host "Analytiq is starting. Open http://localhost:5173 and choose Try the live demo."
