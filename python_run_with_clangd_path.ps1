param(
    [Parameter(Mandatory = $true)]
    [Alias('clangd_path')]
    [string]$ClangdPath,

    [Parameter(Mandatory = $true, Position = 0)]
    [string]$PythonFile,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs,

    [Parameter(Mandatory = $false)]
    [string]$PythonExe = "python"
)

$ErrorActionPreference = 'Stop'

try {
    $resolvedClangPath = (Resolve-Path -LiteralPath $ClangdPath).Path

    if (Test-Path -LiteralPath $resolvedClangPath -PathType Container) {
        $libclangDll = Join-Path $resolvedClangPath 'libclang.dll'
        if (-not (Test-Path -LiteralPath $libclangDll -PathType Leaf)) {
            throw "libclang.dll not found in clangd path: $libclangDll"
        }
    }
    elseif (-not (Test-Path -LiteralPath $resolvedClangPath -PathType Leaf)) {
        throw "clang path does not exist: $ClangdPath"
    }

    $resolvedPythonFile = (Resolve-Path -LiteralPath $PythonFile).Path

    $pythonCmd = Get-Command $PythonExe -ErrorAction Stop
    $pythonExePath = $pythonCmd.Source

    $launcher = @'
import os
import runpy
import sys

clang_bin = sys.argv[1]
target_file = sys.argv[2]

os.environ['LIBCLANG_PATH'] = clang_bin

sys.argv = [target_file] + sys.argv[3:]
runpy.run_path(target_file, run_name="__main__")
'@

    $allArgs = @('-c', $launcher, $resolvedClangPath, $resolvedPythonFile)
    if ($PythonArgs) {
        $allArgs += $PythonArgs
    }

    & $pythonExePath @allArgs
    exit $LASTEXITCODE
}
catch {
    Write-Error $_
    exit 1
}
