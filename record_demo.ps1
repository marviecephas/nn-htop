# Ensure Chrome, ttyd, ffmpeg, and vhs are all on PATH for VHS recording
$chromePath = "C:\Program Files\Google\Chrome\Application"
$ttydPath = "C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\tsl0922.ttyd_Microsoft.Winget.Source_8wekyb3d8bbwe"
$ffmpegPath = "C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
$vhsPath = "C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\charmbracelet.vhs_Microsoft.Winget.Source_8wekyb3d8bbwe\vhs_0.12.0_Windows_x86_64"

$env:PATH = "$chromePath;$ttydPath;$ffmpegPath;$vhsPath;" + $env:PATH

Write-Host "Checking dependencies..." -ForegroundColor Cyan
Write-Host " - Chrome: $(Get-Command chrome.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Write-Host " - ttyd:   $(Get-Command ttyd.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Write-Host " - ffmpeg: $(Get-Command ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Write-Host " - vhs:    $(Get-Command vhs.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"

Write-Host "`nRecording demo.tape with VHS..." -ForegroundColor Green
vhs demo.tape

if (Test-Path demo.gif) {
    $size = (Get-Item demo.gif).Length / 1KB
    Write-Host "`nSUCCESS! demo.gif created ($([math]::Round($size, 1)) KB)" -ForegroundColor Green
} else {
    Write-Host "`nWarning: demo.gif was not found." -ForegroundColor Yellow
}
