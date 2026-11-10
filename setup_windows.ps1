$ErrorActionPreference = 'Stop'
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { throw 'Python Launcher not found. Install Python 3.11 from python.org and enable the launcher.' }
py -3.11 --version
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required. Other versions are not the tested target.' }
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
& .\.venv\Scripts\python.exe -m pytest -q
Write-Host 'Setup complete. Activate with: .\.venv\Scripts\Activate.ps1'
