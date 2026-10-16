param(
    [Parameter(Mandatory = $false)]
    [string]$Config = "configs/local.yaml"
)

$ErrorActionPreference = "Stop"

python -m wound3d --config $Config