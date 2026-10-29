$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create the Python environment using the README setup steps first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'artifacts\pipeline.joblib'))) {
    throw 'Run python -m churn.train using the project environment first.'
}
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
Write-Host 'Customer Churn Workbench: http://127.0.0.1:8002'
& $taskPython -m uvicorn churn.api:app --host 127.0.0.1 --port 8002
