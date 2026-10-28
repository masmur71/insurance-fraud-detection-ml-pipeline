param([int]$Port = 5000)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force .mlflow | Out-Null
python -m mlflow server --host 127.0.0.1 --port $Port --workers 1 --backend-store-uri sqlite:///.mlflow/tracking.db --serve-artifacts --artifacts-destination .mlflow/artifacts
exit $LASTEXITCODE
