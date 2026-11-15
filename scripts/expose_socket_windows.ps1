<#
.SYNOPSIS
  Make the DeceptionX model socket (running inside WSL 2) reachable from OTHER machines on the LAN.

.DESCRIPTION
  WSL 2 in the default NAT mode only forwards *Windows localhost* to Linux. A client on another
  machine (Node 1 MacBook) connects to the Windows host's LAN IP, which WSL never sees. This script:
    1. finds the current WSL IP (it changes on every WSL restart, so re-run after reboots),
    2. adds a netsh portproxy  0.0.0.0:<Port> (Windows)  ->  <WSL IP>:<Port>,
    3. adds an inbound Windows Firewall rule for that TCP port (Private profile by default),
    4. prints the address clients should use.
  Run in an *elevated* PowerShell (Run as Administrator). Use -Remove to undo everything.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\expose_socket_windows.ps1
  powershell -ExecutionPolicy Bypass -File scripts\expose_socket_windows.ps1 -Port 5002 -Profile Private
  powershell -ExecutionPolicy Bypass -File scripts\expose_socket_windows.ps1 -Remove
#>
param(
    [int]$Port = 5002,
    [string]$Distro = "Ubuntu",
    [ValidateSet("Private", "Domain", "Public", "Any")]
    [string]$Profile = "Private",
    # Who may connect. LocalSubnet = only machines on the same LAN segment (safe on a Public profile).
    [string]$RemoteAddress = "LocalSubnet",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$ruleName = "DeceptionX WSL socket TCP $Port"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "Run this from an elevated PowerShell (Run as Administrator)."
}

if ($Remove) {
    netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$Port | Out-Null
    Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    Write-Host "Removed portproxy and firewall rule for port $Port."
    exit 0
}

# 1. Current WSL IP (first address of eth0)
# Parse in PowerShell: wsl.exe re-evaluates its command line in a shell, which would eat awk's $4.
$ipLine = (wsl -d $Distro -- ip -4 -o addr show eth0) -join " "
$wslIp = if ($ipLine -match 'inet (\d+\.\d+\.\d+\.\d+)') { $Matches[1] } else { $ipLine }
if (-not ($wslIp -match '^\d+\.\d+\.\d+\.\d+$')) {
    Write-Error "Could not determine WSL IP (got '$wslIp'). Is the '$Distro' distro running?"
}

# Mirrored networking already shares the host's interfaces; a proxy would then conflict.
$wslconfig = Join-Path $env:USERPROFILE ".wslconfig"
if ((Test-Path $wslconfig) -and (Select-String -Path $wslconfig -Pattern 'networkingMode\s*=\s*mirrored' -Quiet)) {
    Write-Warning "WSL is in mirrored networking mode: only the firewall rule is needed, skipping portproxy."
} else {
    # 2. Port proxy (replace any stale mapping from a previous WSL IP)
    netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$Port 2>$null | Out-Null
    netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$Port connectaddress=$wslIp connectport=$Port | Out-Null
}

# 3. Firewall rule (inbound TCP on the chosen profile only)
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
    -Profile $Profile -RemoteAddress $RemoteAddress | Out-Null
$active = (Get-NetConnectionProfile | Select-Object -ExpandProperty NetworkCategory) -join ","
if ($Profile -ne "Any" -and $active -notmatch $Profile) {
    Write-Warning "Active network profile is '$active' but the rule is for '$Profile' - it will not apply. Re-run with -Profile $active (or Any)."
}

# 4. Report
$lanIps = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)' -and $_.PrefixOrigin -ne 'WellKnown' } |
    Select-Object -ExpandProperty IPAddress
Write-Host "WSL $Distro IP           : $wslIp"
Write-Host "Port proxy               : 0.0.0.0:$Port -> ${wslIp}:$Port"
Write-Host "Firewall rule            : '$ruleName' ($Profile profile, remote: $RemoteAddress)"
Write-Host "Clients on the LAN connect to:"
foreach ($ip in $lanIps) { Write-Host "    ${ip}:$Port" }
Write-Host ""
Write-Host "Current proxies:"
netsh interface portproxy show v4tov4
Write-Host "Note: the WSL IP changes when WSL restarts - re-run this script after a reboot."
