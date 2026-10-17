$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path "$PSScriptRoot\.local" -Force | Out-Null
& "$PSScriptRoot\.venv\Scripts\python.exe" -m pytest tests --basetemp "$PSScriptRoot\.local\pytest" -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& "$PSScriptRoot\.venv\Scripts\python.exe" scripts/check_pages.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& "$PSScriptRoot\.venv\Scripts\python.exe" scripts/check_pages.py --public
exit $LASTEXITCODE
