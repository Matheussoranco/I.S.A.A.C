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

Push-Location $ProjectRoot
try {
    & $PythonPath -m pip install -e ".[desktop,packaging]"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

    & $PythonPath -m PyInstaller `
        --noconfirm `
        --clean `
        --windowed `
        --name "ISAAC" `
        --specpath "build" `
        --collect-all "isaac" `
        --collect-all "webview" `
        --hidden-import "webview.platforms.edgechromium" `
        "src\isaac\interfaces\desktop_entry.py"
    if ($LASTEXITCODE -ne 0) { throw "Windows application build failed." }

    $PackagePath = Join-Path $ProjectRoot "dist\ISAAC-$Version-Windows-x64.zip"
    if (Test-Path -LiteralPath $PackagePath -PathType Leaf) {
        Remove-Item -LiteralPath $PackagePath -Force
    }
    Compress-Archive `
        -LiteralPath (Join-Path $ProjectRoot "dist\ISAAC") `
        -DestinationPath $PackagePath `
        -CompressionLevel Optimal

    Write-Host "Built native app: $ProjectRoot\dist\ISAAC\ISAAC.exe"
    Write-Host "Built portable package: $PackagePath"
}
finally {
    Pop-Location
}
