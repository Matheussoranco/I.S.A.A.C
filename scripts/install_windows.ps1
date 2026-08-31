[CmdletBinding()]
param(
    [string]$Source = "",
    [switch]$NoDesktopShortcut,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectMetadata = Get-Content -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -Raw
$VersionMatch = [regex]::Match($ProjectMetadata, '(?m)^version\s*=\s*"([^"]+)"')
if (-not $VersionMatch.Success) {
    throw "Project version was not found in pyproject.toml."
}
if (-not $Source) {
    $Source = Join-Path $ProjectRoot "dist\ISAAC-$($VersionMatch.Groups[1].Value)-Windows-x64.exe"
}

$SourcePath = [IO.Path]::GetFullPath($Source)
$ExpectedRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "Programs"))
$InstallPath = [IO.Path]::GetFullPath((Join-Path $ExpectedRoot "ISAAC"))
$StagingPath = [IO.Path]::GetFullPath((Join-Path $ExpectedRoot "ISAAC.installing"))
$PreviousPath = [IO.Path]::GetFullPath((Join-Path $ExpectedRoot "ISAAC.previous"))

foreach ($Path in @($InstallPath, $StagingPath, $PreviousPath)) {
    if (-not $Path.StartsWith($ExpectedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing an install path outside $ExpectedRoot"
    }
}
$SourceIsExecutable = Test-Path -LiteralPath $SourcePath -PathType Leaf
$SourceIsFolder = Test-Path -LiteralPath $SourcePath -PathType Container
if ($SourceIsExecutable -and [IO.Path]::GetExtension($SourcePath) -ne ".exe") {
    throw "The standalone package must be an .exe file: $SourcePath"
}
if ($SourceIsFolder -and -not (Test-Path -LiteralPath (Join-Path $SourcePath "ISAAC.exe") -PathType Leaf)) {
    throw "ISAAC.exe was not found in the legacy package folder: $SourcePath"
}
if (-not $SourceIsExecutable -and -not $SourceIsFolder) {
    throw "I.S.A.A.C. package was not found: $SourcePath"
}

New-Item -ItemType Directory -Path $ExpectedRoot -Force | Out-Null
if (Test-Path -LiteralPath $StagingPath) {
    Remove-Item -LiteralPath $StagingPath -Recurse -Force
}
New-Item -ItemType Directory -Path $StagingPath | Out-Null
if ($SourceIsExecutable) {
    Copy-Item -LiteralPath $SourcePath -Destination (Join-Path $StagingPath "ISAAC.exe") -Force
} else {
    Copy-Item -Path (Join-Path $SourcePath "*") -Destination $StagingPath -Recurse -Force
}

if (Test-Path -LiteralPath $PreviousPath) {
    Remove-Item -LiteralPath $PreviousPath -Recurse -Force
}
if (Test-Path -LiteralPath $InstallPath) {
    Move-Item -LiteralPath $InstallPath -Destination $PreviousPath
}

try {
    Move-Item -LiteralPath $StagingPath -Destination $InstallPath
} catch {
    if (Test-Path -LiteralPath $PreviousPath) {
        Move-Item -LiteralPath $PreviousPath -Destination $InstallPath
    }
    throw
}

$Executable = Join-Path $InstallPath "ISAAC.exe"
$Shell = New-Object -ComObject WScript.Shell
$StartMenu = [Environment]::GetFolderPath("Programs")
$StartMenuShortcut = Join-Path $StartMenu "I.S.A.A.C.lnk"
$Shortcut = $Shell.CreateShortcut($StartMenuShortcut)
$Shortcut.TargetPath = $Executable
$Shortcut.WorkingDirectory = $InstallPath
$Shortcut.Description = "I.S.A.A.C. autonomous desktop agent"
$Shortcut.Save()

if (-not $NoDesktopShortcut) {
    $Desktop = [Environment]::GetFolderPath("Desktop")
    $DesktopShortcut = Join-Path $Desktop "I.S.A.A.C.lnk"
    $Shortcut = $Shell.CreateShortcut($DesktopShortcut)
    $Shortcut.TargetPath = $Executable
    $Shortcut.WorkingDirectory = $InstallPath
    $Shortcut.Description = "I.S.A.A.C. autonomous desktop agent"
    $Shortcut.Save()
}

Write-Host "Installed I.S.A.A.C. at $InstallPath"
Write-Host "Start Menu shortcut: $StartMenuShortcut"
if (Test-Path -LiteralPath $PreviousPath) {
    Write-Host "Previous version retained at $PreviousPath"
}
if (-not $NoLaunch) {
    Start-Process -FilePath $Executable -WorkingDirectory $InstallPath
}
