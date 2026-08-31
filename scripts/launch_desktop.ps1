$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$executable = Join-Path $projectRoot "desktop\publish\win-x64\UrbanFly.Desktop.exe"

if (-not (Test-Path -LiteralPath $executable)) {
    & (Join-Path $PSScriptRoot "build_desktop.ps1")
}

if (-not (Test-Path -LiteralPath $executable)) {
    throw "Desktop publish did not produce UrbanFly.Desktop.exe: $executable"
}

$env:URBANFLY_ROOT = $projectRoot
Start-Process -FilePath $executable -WorkingDirectory $projectRoot -WindowStyle Hidden
