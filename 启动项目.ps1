$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\app\streamlit_app.py" --server.address=127.0.0.1 --server.port=8503
