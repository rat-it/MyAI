$env:KMP_DUPLICATE_LIB_OK="TRUE"
$ErrorActionPreference = 'Continue'
$jupyter = "c:\AI-Projects\MyAI\DL\dl_env\Scripts\jupyter.exe"

$notebooks = Get-ChildItem -Path "c:\AI-Projects\MyAI\*.ipynb" -Recurse | Where-Object { $_.FullName -notmatch "\.ipynb_checkpoints" }

$failed = @()

foreach ($nb in $notebooks) {
    Write-Output "Running notebook: $($nb.Name)"
    try {
        & $jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=600 $nb.FullName
        if ($LASTEXITCODE -ne 0) {
            Write-Output "Warning: Failed to execute $($nb.Name)"
            $failed += $nb.FullName
        } else {
            Write-Output "Successfully executed $($nb.Name)"
        }
    } catch {
        Write-Output "Error running $($nb.Name): $_"
        $failed += $nb.FullName
    }
}

Write-Output "All notebooks executed."

if ($failed.Length -gt 0) {
    Write-Output "Failed Notebooks:"
    $failed | ForEach-Object { Write-Output " - $_" }
    $failed | Out-File "c:\AI-Projects\MyAI\failed_notebooks.txt"
} else {
    Write-Output "All notebooks ran successfully!"
    if (Test-Path "c:\AI-Projects\MyAI\failed_notebooks.txt") {
        Remove-Item "c:\AI-Projects\MyAI\failed_notebooks.txt"
    }
}
