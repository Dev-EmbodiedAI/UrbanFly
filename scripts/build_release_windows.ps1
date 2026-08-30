param(
    [string]$Version = "1.0.0",
    [string]$OutputDirectory = "",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$releaseRoot = if ($OutputDirectory) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    Join-Path $projectRoot "dist\release"
}
$stage = Join-Path $projectRoot "dist\stage\windows\UrbanFly"
$pyinstallerDist = Join-Path $projectRoot "dist\pyinstaller\windows"
$pyinstallerWork = Join-Path $projectRoot "dist\pyinstaller\work-windows"
$desktopDist = Join-Path $projectRoot "dist\desktop\windows"
$releaseVenv = Join-Path $projectRoot "dist\venv-release-windows"

foreach ($candidate in @($stage, $pyinstallerDist, $pyinstallerWork, $desktopDist, $releaseVenv)) {
    $resolved = [System.IO.Path]::GetFullPath($candidate)
    if (-not $resolved.StartsWith(
        [System.IO.Path]::GetFullPath((Join-Path $projectRoot "dist")) + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe build target: $resolved"
    }
    if ([System.IO.Directory]::Exists($resolved)) {
        [System.IO.Directory]::Delete($resolved, $true)
    }
}
New-Item -ItemType Directory -Path $stage, $releaseRoot -Force | Out-Null

Push-Location (Join-Path $projectRoot "frontend")
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
} finally {
    Pop-Location
}

python -m venv $releaseVenv
$releasePython = Join-Path $releaseVenv "Scripts\python.exe"
& $releasePython -m pip install --disable-pip-version-check `
    -r (Join-Path $projectRoot "requirements-runtime.txt") `
    pyinstaller==6.16.0
if ($LASTEXITCODE -ne 0) { throw "release dependency install failed" }

& $releasePython -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name UrbanFly.Backend `
    --distpath $pyinstallerDist `
    --workpath $pyinstallerWork `
    --specpath (Join-Path $projectRoot "dist\pyinstaller") `
    --hidden-import scipy.spatial.transform._rotation_groups `
    --hidden-import scipy._external.array_api_compat.numpy.fft `
    (Join-Path $projectRoot "scripts\urbanfly_backend_entry.py")
if ($LASTEXITCODE -ne 0) { throw "backend freeze failed" }

# Conda Python keeps libffi outside the ordinary DLL directory. PyInstaller
# cannot always resolve it automatically, so include it when that layout is
# detected. Official python.org installations do not need this branch.
$pythonPrefix = (& python -c "import sys; print(sys.prefix)").Trim()
$ffiDll = Join-Path $pythonPrefix "Library\bin\ffi.dll"
if (Test-Path -LiteralPath $ffiDll) {
    Copy-Item -LiteralPath $ffiDll -Destination `
        (Join-Path $pyinstallerDist "UrbanFly.Backend\_internal\ffi.dll") -Force
}

$desktopProject = Join-Path $projectRoot "desktop\UrbanFly.Desktop\UrbanFly.Desktop.csproj"
dotnet publish $desktopProject `
    --configuration Release `
    --runtime win-x64 `
    --self-contained true `
    -p:DebugType=None `
    -p:DebugSymbols=false `
    --output $desktopDist
if ($LASTEXITCODE -ne 0) { throw "desktop publish failed" }

Copy-Item (Join-Path $desktopDist "*") $stage -Recurse -Force
New-Item -ItemType Directory -Path (Join-Path $stage "bin") | Out-Null
Copy-Item (Join-Path $pyinstallerDist "UrbanFly.Backend") `
    (Join-Path $stage "bin\UrbanFly.Backend") -Recurse -Force
New-Item -ItemType Directory -Path (Join-Path $stage "frontend") | Out-Null
Copy-Item (Join-Path $projectRoot "frontend\dist") `
    (Join-Path $stage "frontend\dist") -Recurse -Force
New-Item -ItemType Directory -Path (Join-Path $stage "data\helsinki_mesh") -Force | Out-Null
Copy-Item (Join-Path $projectRoot "data\helsinki_mesh\HelsinkiCentral1km") `
    (Join-Path $stage "data\helsinki_mesh\HelsinkiCentral1km") -Recurse -Force
New-Item -ItemType Directory -Path (Join-Path $stage "models") | Out-Null
Copy-Item (Join-Path $projectRoot "models\helsinki_observation_policy_v1.pt") `
    (Join-Path $stage "models") -Force
Copy-Item (Join-Path $projectRoot "models\helsinki_observation_policy_v1.metrics.json") `
    (Join-Path $stage "models") -Force
Copy-Item (Join-Path $projectRoot "models\helsinki_latent_world_model_v1.pt") `
    (Join-Path $stage "models") -Force
Copy-Item (Join-Path $projectRoot "models\helsinki_latent_world_model_v1.metrics.json") `
    (Join-Path $stage "models") -Force
Copy-Item (Join-Path $projectRoot "README.md"), `
    (Join-Path $projectRoot "LICENSE"), `
    (Join-Path $projectRoot "THIRD_PARTY_NOTICES.md") $stage -Force

$marker = [ordered]@{
    schema = "urbanfly-release-v1"
    version = $Version
    platform = "windows-x64"
    city = "HelsinkiCentral1km"
    city_asset_included = $true
    qwen_weights_included = $false
    qwen_integration = "OpenAI-compatible API; key supplied by environment only"
}
$marker | ConvertTo-Json | Set-Content -LiteralPath `
    (Join-Path $stage "urbanfly-release.json") -Encoding utf8

$env:URBANFLY_ROOT = $stage
$backendExe = Join-Path $stage "bin\UrbanFly.Backend\UrbanFly.Backend.exe"
$backend = Start-Process -FilePath $backendExe -WorkingDirectory $stage -PassThru -WindowStyle Hidden
try {
    $deadline = (Get-Date).AddSeconds(90)
    do {
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:8765/api/health" -TimeoutSec 3
            if ($health.status -eq "ok") { break }
        } catch { }
        if ($backend.HasExited) { throw "packaged backend exited with $($backend.ExitCode)" }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    if ($health.status -ne "ok") { throw "packaged backend health timeout" }
    $indexStatus = (& curl.exe --silent --show-error --output NUL `
        --write-out "%{http_code}" "http://127.0.0.1:8765/").Trim()
    if ($LASTEXITCODE -ne 0 -or $indexStatus -ne "200") {
        throw "packaged frontend readback failed: HTTP $indexStatus"
    }
} finally {
    if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
    Remove-Item Env:URBANFLY_ROOT -ErrorAction SilentlyContinue
}

$portable = Join-Path $releaseRoot "UrbanFly-Windows-x64-$Version-portable.zip"
if (Test-Path $portable) { Remove-Item -LiteralPath $portable -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $portable -CompressionLevel Optimal

if (-not $SkipInstaller) {
    $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
    if (-not $iscc) {
        $iscc = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"
    }
    if (-not (Test-Path -LiteralPath $iscc)) {
        $iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    }
    if (-not (Test-Path -LiteralPath $iscc)) {
        throw "Inno Setup 6 not found; install it or pass -SkipInstaller"
    }
    & $iscc `
        "/DVERSION=$Version" `
        "/DSOURCEDIR=$stage" `
        "/DOUTPUTDIR=$releaseRoot" `
        (Join-Path $projectRoot "packaging\windows\UrbanFly.iss")
    if ($LASTEXITCODE -ne 0) { throw "installer build failed" }
}

$artifacts = Get-ChildItem $releaseRoot -File | ForEach-Object {
    [ordered]@{
        name = $_.Name
        bytes = $_.Length
        sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$artifacts | ConvertTo-Json -Depth 3 | Set-Content `
    (Join-Path $releaseRoot "windows-release-manifest.json") -Encoding utf8
$artifacts | Format-Table
