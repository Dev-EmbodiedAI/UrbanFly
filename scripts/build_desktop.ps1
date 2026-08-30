$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $projectRoot "desktop\UrbanFly.Desktop\UrbanFly.Desktop.csproj"
$output = Join-Path $projectRoot "desktop\publish\win-x64"
$runningShell = Get-Process -Name UrbanFly -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq (Join-Path $output "UrbanFly.exe") }
if ($runningShell) { throw "Close UrbanFly Desktop before publishing; no live files will be replaced." }

Push-Location (Join-Path $projectRoot "frontend")
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
} finally {
    Pop-Location
}

dotnet restore $project
if ($LASTEXITCODE -ne 0) { throw "Desktop restore failed" }
dotnet publish $project `
    --configuration Release `
    --runtime win-x64 `
    --self-contained false `
    --output $output
if ($LASTEXITCODE -ne 0) { throw "Desktop publish failed" }

Write-Host "UrbanFly Desktop: $output\UrbanFly.exe"
