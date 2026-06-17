param(
    [string]$Dataset = "C:\Users\user\Desktop\AllFile\dataset",
    [string]$ModelOut = "",
    [string]$ReportsDir = "",
    [switch]$SkipTrain,
    [switch]$TrainBundle,
    [switch]$AllowGateFailure,
    [switch]$RunTransformerEval
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (!$ModelOut) {
    if ($TrainBundle) {
        $ModelOut = Join-Path $ProjectRoot "models\poker_policy_bundle.joblib"
    } else {
        $ModelOut = Join-Path $ProjectRoot "models\poker_policy.joblib"
    }
}
if (!$ReportsDir) {
    $ReportsDir = Join-Path $ProjectRoot "reports"
}

$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot "env\Scripts\python.exe")
)
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $PythonCandidates += $PythonCommand.Source
}

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if ($Candidate -and (Test-Path $Candidate)) {
        & $Candidate -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $Python = $Candidate
            break
        }
    }
}
if (!$Python) {
    Write-Error "Working Python was not found. Run .\install.ps1 first."
}
if (!(Test-Path $Dataset)) {
    Write-Error "Dataset folder not found: $Dataset"
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null


function Remove-DeliveryArtifacts {
    param([string]$Root)
    foreach ($Relative in @(".qodo", "__pycache__", "poker_agent\__pycache__", "scripts\__pycache__")) {
        $Target = Join-Path $Root $Relative
        if (Test-Path -LiteralPath $Target) {
            Remove-Item -LiteralPath $Target -Recurse -Force
        }
    }
}

function Remove-FailedHydraRuns {
    param([string]$Root)
    $HydraRoot = Join-Path $Root "reports\hydra"
    if (!(Test-Path -LiteralPath $HydraRoot)) {
        return
    }
    $ResolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    foreach ($RunFile in Get-ChildItem -LiteralPath $HydraRoot -Recurse -File -Filter "run.json" -ErrorAction SilentlyContinue) {
        try {
            $Run = Get-Content -LiteralPath $RunFile.FullName -Raw | ConvertFrom-Json
        } catch {
            continue
        }
        if ($Run.status -eq "failed") {
            $RunDirectory = $RunFile.Directory.FullName
            if (!$RunDirectory.StartsWith($ResolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing outside-root removal: $RunDirectory"
            }
            Remove-Item -LiteralPath $RunDirectory -Recurse -Force
        }
    }
}

$AuditReport = Join-Path $ReportsDir "dataset_audit.json"
$RepositoryAuditReport = Join-Path $ReportsDir "repository_audit.json"
$GateReport = Join-Path $ReportsDir "production_gate.json"
$RepoHygieneReport = Join-Path $ReportsDir "repo_hygiene.json"
$EventBenchmarkReport = Join-Path $ReportsDir "llm_event_benchmark.json"
$EventMethodologyReport = Join-Path $ReportsDir "llm_event_methodology.md"
$EventSchemaReport = Join-Path $ReportsDir "event_schema_dataset_report.json"
$EventSchemaMarkdownReport = Join-Path $ReportsDir "event_schema_dataset_report.md"
$Phase2BenchmarkCsv = Join-Path $ReportsDir "phase2_event_benchmark_results.csv"
$Phase2BenchmarkJson = Join-Path $ReportsDir "phase2_event_benchmark_results.json"
$Phase2BenchmarkPredictions = Join-Path $ReportsDir "phase2_event_benchmark_predictions.jsonl"
$Phase2BenchmarkMarkdown = Join-Path $ReportsDir "phase2_event_benchmark_report.md"
$GoldEvalReport = Join-Path $ReportsDir "llm_event_gold_eval.json"
$GoldPredictionsReport = Join-Path $ReportsDir "llm_event_gold_predictions.jsonl"
$GoldMarkdownReport = Join-Path $ReportsDir "llm_event_gold_report.md"
$TransformerEvalReport = Join-Path $ReportsDir "llm_transformer_gold_eval.json"
$TransformerMarkdownReport = Join-Path $ReportsDir "llm_transformer_gold_report.md"
$DecisionEvalReport = Join-Path $ReportsDir "llm_decision_agent_eval.json"
$DecisionMarkdownReport = Join-Path $ReportsDir "llm_decision_agent_report.md"

Write-Host "1/8 Auditing dataset..." -ForegroundColor Green
& $Python scripts\audit_dataset.py `
    --dataset $Dataset `
    --out $AuditReport `
    --missing-hole-cards flag `
    --max-feature-examples 50000

if (!$SkipTrain) {
    if ($TrainBundle) {
        Write-Host "2/8 Training routed policy bundle..." -ForegroundColor Green
        & $Python scripts\train_policy_bundle.py `
            --dataset $Dataset `
            --model-out $ModelOut `
            --max-examples 50000 `
            --max-iter 60 `
            --learning-rate 0.05 `
            --max-leaf-nodes 31 `
            --l2-regularization 0.02 `
            --class-weighting sqrt_balanced `
            --max-class-weight 6
    } else {
        Write-Host "2/8 Training leakage-aware single policy..." -ForegroundColor Green
        & $Python scripts\train_policy.py `
            --dataset $Dataset `
            --model-out $ModelOut `
            --policy hist_gradient_boosting `
            --max-examples 0 `
            --max-iter 90 `
            --learning-rate 0.05 `
            --max-leaf-nodes 31 `
            --l2-regularization 0.02 `
            --class-weighting sqrt_balanced `
            --max-class-weight 6 `
            --missing-hole-cards drop `
            --split-strategy stratified_hand_group
    }
} else {
    Write-Host "2/8 Skipping training; using existing model." -ForegroundColor Yellow
}

Write-Host "3/8 Running production gate..." -ForegroundColor Green
& $Python scripts\production_gate.py `
    --model $ModelOut `
    --audit-report $AuditReport `
    --out $GateReport
$GateExit = $LASTEXITCODE
if ($GateExit -ne 0 -and !$AllowGateFailure) {
    Write-Error "Production gate failed. Use -AllowGateFailure only when preparing a research/prototype delivery."
}

Write-Host "4/8 Running event extraction benchmark..." -ForegroundColor Green
& $Python scripts\llm_event_benchmark.py `
    --input (Join-Path $ProjectRoot "dataset\logs") `
    --out $EventBenchmarkReport `
    --methodology-out $EventMethodologyReport `
    --prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_prompt.txt") `
    --provider local_rules `
    --max-files 2 `
    --max-records 1000 `
    --min-confidence 0.2

Write-Host "4b/8 Building Phase 1 event schema dataset..." -ForegroundColor Green
& $Python scripts\run_hydra_experiment.py `
    experiments=build_event_schema_dataset `
    "python_executable=$Python"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Phase 1 event schema dataset build failed."
}

Write-Host "4c/8 Running Phase 2 event-normalization benchmark..." -ForegroundColor Green
& $Python scripts\run_hydra_experiment.py `
    experiments=phase2_event_benchmark `
    "python_executable=$Python"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Phase 2 event-normalization benchmark failed."
}

Write-Host "5/8 Running gold event extraction evaluation..." -ForegroundColor Green
& $Python scripts\llm_event_gold_eval.py `
    --gold (Join-Path $ProjectRoot "evaluation\event_extraction_gold.jsonl") `
    --out $GoldEvalReport `
    --predictions-out $GoldPredictionsReport `
    --report-out $GoldMarkdownReport `
    --minimal-prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_minimal.txt") `
    --permissive-prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_permissive.txt") `
    --strict-prompt (Join-Path $ProjectRoot "configs\prompts\event_extraction_strict.txt")

if ($RunTransformerEval) {
    Write-Host "5b/8 Running local instruction-model evaluation..." -ForegroundColor Green
    & $Python scripts\run_hydra_experiment.py `
        experiments=llm_transformer_gold_eval `
        "python_executable=$Python"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Local instruction-model evaluation failed."
    }
} elseif (!(Test-Path $TransformerEvalReport) -or !(Test-Path $TransformerMarkdownReport)) {
    Write-Error "Instruction-model reports are missing. Re-run with -RunTransformerEval."
}

Write-Host "5c/8 Running text-model decision baseline..." -ForegroundColor Green
& $Python scripts\run_hydra_experiment.py `
    experiments=llm_decision_baseline `
    "python_executable=$Python"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Text-model decision baseline failed."
}

Write-Host "6/8 Auditing repository..." -ForegroundColor Green
& $Python scripts\audit_repository.py `
    --root $ProjectRoot `
    --out $RepositoryAuditReport

Remove-DeliveryArtifacts -Root $ProjectRoot
Write-Host "7/8 Checking repository hygiene..." -ForegroundColor Green
& $Python scripts\check_repo_hygiene.py `
    --root $ProjectRoot `
    --json-out $RepoHygieneReport
if ($LASTEXITCODE -ne 0) {
    Write-Error "Repository hygiene check failed. Remove local tool metadata or delivery-only comments before rebuilding the ZIP."
}

Remove-DeliveryArtifacts -Root $ProjectRoot
Remove-FailedHydraRuns -Root $ProjectRoot
Write-Host "8/8 Rebuilding delivery ZIP..." -ForegroundColor Green

$GeneratedDirs = Get-ChildItem -Force -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Where-Object {
    $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\env\*"
}
foreach ($Dir in $GeneratedDirs) {
    if ($Dir.FullName.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $Dir.FullName -Recurse -Force
    }
}
$GeneratedFiles = Get-ChildItem -Force -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    ($_.Extension -in @(".pyc", ".pyo", ".pyd") -or $_.Name -eq "requirements-research.txt") -and
    $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\env\*"
}
foreach ($File in $GeneratedFiles) {
    if ($File.FullName.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $File.FullName -Force
    }
}
$ZipPath = Join-Path $ProjectRoot "release\poker-decision-agent.zip"
$Items = Get-ChildItem -Force | Where-Object {
    $_.Name -notin @(".git", ".qodo", ".venv", "env", "dataset", "sample_out", "smoke_dataset", "__pycache__", "release", "research_runs")
}
Compress-Archive -Path $Items.FullName -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "Delivery workflow complete." -ForegroundColor Green
Write-Host "Model: $ModelOut"
Write-Host "Audit: $AuditReport"
Write-Host "Repository audit: $RepositoryAuditReport"
Write-Host "Gate: $GateReport"
Write-Host "Repo hygiene: $RepoHygieneReport"
Write-Host "Event benchmark: $EventBenchmarkReport"
Write-Host "Event methodology: $EventMethodologyReport"
Write-Host "Event schema dataset: $EventSchemaReport"
Write-Host "Event schema report: $EventSchemaMarkdownReport"
Write-Host "Phase 2 benchmark CSV: $Phase2BenchmarkCsv"
Write-Host "Phase 2 benchmark JSON: $Phase2BenchmarkJson"
Write-Host "Phase 2 benchmark predictions: $Phase2BenchmarkPredictions"
Write-Host "Phase 2 benchmark report: $Phase2BenchmarkMarkdown"
Write-Host "Gold event eval: $GoldEvalReport"
Write-Host "Gold event report: $GoldMarkdownReport"
Write-Host "Instruction-model eval: $TransformerEvalReport"
Write-Host "Instruction-model report: $TransformerMarkdownReport"
Write-Host "Decision baseline eval: $DecisionEvalReport"
Write-Host "Decision baseline report: $DecisionMarkdownReport"
Write-Host "ZIP: $ZipPath"
