[CmdletBinding()]
param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot $Python
$ProjectMetadata = Get-Content -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -Raw
$VersionMatch = [regex]::Match($ProjectMetadata, '(?m)^version\s*=\s*"([^"]+)"')

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}
if (-not $VersionMatch.Success) {
    throw "Project version was not found in pyproject.toml."
}
$Version = $VersionMatch.Groups[1].Value
$ExecutableName = "ISAAC-$Version-Windows-x64"
$ExecutablePath = Join-Path $ProjectRoot "dist\$ExecutableName.exe"

Push-Location $ProjectRoot
try {
    & $PythonPath -m pip install -e ".[desktop,packaging]"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

    & $PythonPath -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name $ExecutableName `
        --specpath "build" `
        --collect-all "isaac" `
        --collect-all "webview" `
        --hidden-import "webview.platforms.edgechromium" `
        "src\isaac\interfaces\desktop_entry.py"
    if ($LASTEXITCODE -ne 0) { throw "Windows application build failed." }

    if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
        throw "Standalone executable was not created: $ExecutablePath"
    }

    Write-Host "Built standalone native app: $ExecutablePath"
}
finally {
    Pop-Location
}
