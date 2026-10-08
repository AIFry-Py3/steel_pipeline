$ErrorActionPreference = 'Stop'

# Run from the project root so relative paths inside the Python scripts resolve
Set-Location -Path $PSScriptRoot

# Ordered pipeline
$PipelineScripts = @(
    # '.\visual_extractor\llm_visual_extractor.py',
    # '.\test.py',
    # '.\visual_extractor\get_pdfs.py'
    '.\visual_extractor\llm_vis_locally.py'
    '.\data cleaning pipeline files\split_by_period_type.py',
    '.\data cleaning pipeline files\csv_sort_merge_dedup.py',
    '.\data cleaning pipeline files\consumption_from_cumulative.py',
    '.\data cleaning pipeline files\csv_time_series_data_engg_utils.py',
    '.\data cleaning pipeline files\ridge_imputation.py'
)

# Fail fast: python available?
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'python' not found on PATH." -ForegroundColor Red
    exit 1
}

# Fail fast: all scripts exist before anything runs
$Missing = $PipelineScripts | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
if ($Missing) {
    Write-Host "ERROR: Missing script(s):" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

$env:PYTHONUTF8 = '1'   # avoid console encoding crashes in Python

Write-Host "Starting data pipeline..." -ForegroundColor Cyan
$Total = $PipelineScripts.Count
$i = 0

foreach ($Script in $PipelineScripts) {
    $i++
    Write-Host "[$i/$Total] Running: $Script" -ForegroundColor Yellow
    $Timer = [System.Diagnostics.Stopwatch]::StartNew()

    $global:LASTEXITCODE = 0
    & python -u $Script
    $Code = $LASTEXITCODE
    $Timer.Stop()

    if ($Code -ne 0) {
        Write-Host "ERROR: exit code $Code -> $Script" -ForegroundColor Red
        Write-Host "Pipeline halted to prevent data corruption." -ForegroundColor Red
        exit $Code
    }

    Write-Host ("OK: finished in {0:N1}s" -f $Timer.Elapsed.TotalSeconds) -ForegroundColor Green
    Write-Host ('-' * 40) -ForegroundColor Gray
}

Write-Host "Pipeline completed successfully." -ForegroundColor Green