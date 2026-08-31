[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$IdfArguments
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $PSScriptRoot
$eimConfigPath = "C:\Espressif\tools\eim_idf.json"

if (-not (Test-Path -LiteralPath $eimConfigPath)) {
    throw "ESP-IDF Installation Manager configuration was not found: $eimConfigPath"
}

$eimConfig = Get-Content -LiteralPath $eimConfigPath -Raw | ConvertFrom-Json
$selectedId = $eimConfig.idfSelectedId
$installation = $eimConfig.idfInstalled |
    Where-Object { $_.id -eq $selectedId } |
    Select-Object -First 1

if ($null -eq $installation) {
    throw "The selected ESP-IDF installation '$selectedId' was not found."
}
if (-not (Test-Path -LiteralPath $installation.activationScript)) {
    throw "ESP-IDF activation script was not found: $($installation.activationScript)"
}

. $installation.activationScript
$env:IDF_TARGET = "esp32s3"
$projectPythonEnvironment = Join-Path $projectDirectory ".idf-python\idf6.0_py3.11_env"
$projectPython = Join-Path $projectPythonEnvironment "Scripts\python.exe"
$idfPython = $installation.python
if (Test-Path -LiteralPath $projectPython) {
    $env:IDF_PYTHON_ENV_PATH = $projectPythonEnvironment
    $env:Path = "$(Join-Path $projectPythonEnvironment 'Scripts');$env:Path"
    $idfPython = $projectPython
}
Push-Location -LiteralPath $projectDirectory
try {
    & $idfPython "$($installation.path)\tools\idf.py" @IdfArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
