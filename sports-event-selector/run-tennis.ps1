param([int]$Epochs = 20, [int]$Threads = 4, [string]$Output = 'runs/tennis_chatgpt_repeat')
$ErrorActionPreference = 'Stop'
$workspaceDeps = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../work/deps'))
$runtimePython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$oldPythonPath = $env:PYTHONPATH
Push-Location $PSScriptRoot
try {
    if (Test-Path -LiteralPath $venvPython) {
        $selectedPython = $venvPython
        $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    } elseif ((Test-Path -LiteralPath $runtimePython) -and (Test-Path -LiteralPath $workspaceDeps)) {
        $selectedPython = $runtimePython
        $env:PYTHONPATH = $workspaceDeps + ';' + (Join-Path $PSScriptRoot 'src')
    } else {
        $selectedPython = 'python'
        $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    }
    & $selectedPython -m sports_events.pilot --epochs $Epochs --threads $Threads --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Tennis pilot exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
