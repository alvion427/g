$pythonPath = Join-Path -Path $PSScriptRoot -ChildPath "venv\Scripts\python.exe"
$gPath = Join-Path -Path $PSScriptRoot -ChildPath "g.py"
& $pythonPath $gPath $args
