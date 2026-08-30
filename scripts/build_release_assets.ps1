param(
    [string]$OutputDirectory = "",
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$releaseRoot = if ($OutputDirectory) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    Join-Path $projectRoot "dist\release"
}
$assetStage = Join-Path $projectRoot "dist\stage\assets"
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "dist"))
foreach ($target in @($assetStage, $releaseRoot)) {
    if (-not ([System.IO.Path]::GetFullPath($target)).StartsWith(
        $distRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe release target: $target"
    }
}
if ([System.IO.Directory]::Exists($assetStage)) {
    [System.IO.Directory]::Delete($assetStage, $true)
}
New-Item -ItemType Directory -Path `
    (Join-Path $assetStage "data\helsinki_mesh"), $releaseRoot -Force | Out-Null
Copy-Item (Join-Path $projectRoot "data\helsinki_mesh\HelsinkiCentral1km") `
    (Join-Path $assetStage "data\helsinki_mesh\HelsinkiCentral1km") -Recurse -Force
Copy-Item (Join-Path $projectRoot "THIRD_PARTY_NOTICES.md") $assetStage -Force

$cityZip = Join-Path $releaseRoot "UrbanFly-HelsinkiCentral1km-Assets-v$Version.zip"
if (Test-Path $cityZip) { Remove-Item -LiteralPath $cityZip -Force }
Compress-Archive -Path (Join-Path $assetStage "*") -DestinationPath $cityZip -CompressionLevel Optimal

$dataset = Join-Path $projectRoot "outputs\helsinki_dataset_v1\main_100_zero_stale_v1"
$datasetZip = Join-Path $releaseRoot "UrbanFly-Helsinki-Dataset-v$Version.zip"
if (Test-Path $datasetZip) { Remove-Item -LiteralPath $datasetZip -Force }
Compress-Archive -Path $dataset -DestinationPath $datasetZip -CompressionLevel Optimal

$demoSource = Join-Path $projectRoot `
    "outputs\world_model_long_range_v1\helsinki_1km_multipoint_world_model_3x.mp4"
$demoTarget = Join-Path $releaseRoot "UrbanFly-WorldModel-1km-Demo-3x.mp4"
Copy-Item -LiteralPath $demoSource -Destination $demoTarget -Force

$rows = Get-Item $cityZip, $datasetZip, $demoTarget | ForEach-Object {
    [ordered]@{
        name = $_.Name
        bytes = $_.Length
        sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$rows | ConvertTo-Json -Depth 3 | Set-Content `
    (Join-Path $releaseRoot "data-release-manifest.json") -Encoding utf8
$rows | Format-Table
