$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$LogsDir = Join-Path $RuntimeDir "logs"
$ApiPort = 18000
$StreamlitPort = 18501
$MlflowPort = 15000

$env:CHURNGUARD_API_BASE_URL = "http://127.0.0.1:$ApiPort"
$env:CHURNGUARD_MLFLOW_URL = "http://127.0.0.1:$MlflowPort"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Stop-RecordedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $PidPath = Join-Path $RuntimeDir "$Name.pid"
    if (-not (Test-Path $PidPath)) {
        return
    }

    $RecordedPid = (Get-Content $PidPath -Raw).Trim()
    if ($RecordedPid) {
        $Process = Get-Process -Id ([int]$RecordedPid) -ErrorAction SilentlyContinue
        if ($Process) {
            Stop-Process -Id $Process.Id -Force
        }
    }

    Remove-Item $PidPath -Force -ErrorAction SilentlyContinue
}

function Start-LoggedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    $StdOutPath = Join-Path $LogsDir "$Name.out.log"
    $StdErrPath = Join-Path $LogsDir "$Name.err.log"
    $Process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $StdOutPath `
        -RedirectStandardError $StdErrPath `
        -PassThru

    $Process.Id | Set-Content (Join-Path $RuntimeDir "$Name.pid")
    return $Process
}

function Wait-ForUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [int]$Attempts = 60,
        [int]$DelaySeconds = 2
    )

    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
        try {
            $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500) {
                return
            }
        }
        catch {
            Start-Sleep -Seconds $DelaySeconds
        }
    }

    throw "Timed out waiting for $Url"
}

function Invoke-JsonPost {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [object]$Body
    )

    $JsonBody = $Body | ConvertTo-Json -Depth 8
    return Invoke-RestMethod -Uri $Url -Method Post -ContentType "application/json" -Body $JsonBody
}

function Invoke-JsonArrayPost {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [object[]]$Body
    )

    $JsonBody = if ($Body.Count -eq 1) {
        "[" + ($Body[0] | ConvertTo-Json -Depth 8) + "]"
    }
    else {
        $Body | ConvertTo-Json -Depth 8
    }

    return Invoke-RestMethod -Uri $Url -Method Post -ContentType "application/json" -Body $JsonBody
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Force -Path "data" | Out-Null
}

if (-not (Test-Path "data\churnguard.db")) {
    New-Item -ItemType File -Force -Path "data\churnguard.db" | Out-Null
}

Stop-RecordedProcess -Name "api"
Stop-RecordedProcess -Name "streamlit"
Stop-RecordedProcess -Name "mlflow"

python data\generate_synthetic.py --rows 2000
python pipelines\training.py
python pipelines\evaluation.py
python -m pytest -q

$ApiProcess = Start-LoggedProcess -Name "api" -FilePath "python" -ArgumentList @(
    "-m",
    "uvicorn",
    "api.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "$ApiPort"
)

$MlflowAvailable = $false
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('mlflow') else 1)"
if ($LASTEXITCODE -eq 0) {
    $MlflowAvailable = $true
    Start-LoggedProcess -Name "mlflow" -FilePath "python" -ArgumentList @(
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        "sqlite:///./data/churnguard.db",
        "--default-artifact-root",
        "./mlruns",
        "--host",
        "127.0.0.1",
        "--port",
        "$MlflowPort"
    ) | Out-Null
}

$StreamlitProcess = Start-LoggedProcess -Name "streamlit" -FilePath "python" -ArgumentList @(
    "-m",
    "streamlit",
    "run",
    "ui/app.py",
    "--server.port",
    "$StreamlitPort",
    "--server.address",
    "127.0.0.1"
)

Wait-ForUrl -Url "http://127.0.0.1:$ApiPort/performance"
Wait-ForUrl -Url "http://127.0.0.1:$StreamlitPort"
if ($MlflowAvailable) {
    Wait-ForUrl -Url "http://127.0.0.1:$MlflowPort"
}

$PredictPayload = @(
    @{
        tenure_months = 12
        monthly_charges = 89.5
        contract_type = "month-to-month"
        num_support_tickets = 3
        avg_satisfaction_score = 5.8
        customer_notes = "Customer is frustrated with billing issues."
    }
)

$PredictResponse = Invoke-JsonArrayPost -Url "http://127.0.0.1:$ApiPort/predict" -Body $PredictPayload
$ExplainResponse = Invoke-JsonPost -Url "http://127.0.0.1:$ApiPort/explain" -Body $PredictPayload[0]
$DriftResponse = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/drift-report" -Method Get
$PerformanceResponse = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/performance" -Method Get

if (-not $PredictResponse[0].customer_id) {
    throw "Predict endpoint returned an invalid response."
}

if (-not $ExplainResponse.risk_level) {
    throw "Explain endpoint returned an invalid response."
}

if ($null -eq $DriftResponse) {
    throw "Drift endpoint did not return a response."
}

if (-not $PerformanceResponse.timestamp) {
    throw "Performance endpoint returned an invalid response."
}

Write-Host ""
Write-Host "ChurnGuard is running."
Write-Host "API:       http://127.0.0.1:$ApiPort"
Write-Host "Streamlit: http://127.0.0.1:$StreamlitPort"
if ($MlflowAvailable) {
    Write-Host "MLflow:    http://127.0.0.1:$MlflowPort"
}
else {
    Write-Host "MLflow:    skipped (package not installed locally)"
}
Write-Host ""
Write-Host ("Predict risk level: {0} ({1:P2})" -f $PredictResponse[0].risk_level, [double]$PredictResponse[0].churn_probability)
Write-Host ("Explain risk level: {0}" -f $ExplainResponse.risk_level)
Write-Host ("Drift items returned: {0}" -f @($DriftResponse).Count)
Write-Host ("Performance history points: {0}" -f @($PerformanceResponse.history).Count)
Write-Host "Logs:      .runtime\\logs"
