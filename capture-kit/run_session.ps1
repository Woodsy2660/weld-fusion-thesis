# RealSense Capture Session Script (PowerShell)
# Edit ORDER to a RANDOMISED coupon sequence before running.
# Generate one with:  python -c "import random;c=[f'C{i:02d}' for i in range(1,16)];random.shuffle(c);print(' '.join(c))"

$ErrorActionPreference = "Stop"

$SESSION = "S1"
$STANDOFF = "300.0"
$ORDER = @("C11", "C04", "C15", "C02", "C09", "C07", "C13", "C01", "C06", "C14", "C03", "C10", "C05", "C12", "C08")  # <-- REPLACE
$REPEAT_COUPONS = @("C11", "C07", "C05")  # 3 coupons get 2 extra re-placements each

$i = 0

Write-Host ">>> session reference target (start)" -ForegroundColor Cyan
python capture.py REF top $SESSION $i --standoff $STANDOFF --note "session start reference"
if ($LASTEXITCODE -ne 0) { throw "Capture failed" }
$i++

Read-Host "Reference captured. Load first coupon and press Enter"

foreach ($c in $ORDER) {
    foreach ($face in @("top", "root")) {
        Read-Host ">>> Load $c, $face face. Press Enter when seated"
        python capture.py $c $face $SESSION $i --standoff $STANDOFF
        if ($LASTEXITCODE -ne 0) { throw "Capture failed for $c $face" }
        $i++
    }
}

foreach ($c in $REPEAT_COUPONS) {
    foreach ($r in 1, 2) {
        Read-Host ">>> REPEATABILITY: re-place $c, top face (rep $r). Press Enter"
        python capture.py $c top $SESSION $i --rep $r --standoff $STANDOFF --note "repeat placement for u_align"
        if ($LASTEXITCODE -ne 0) { throw "Repeatability capture failed for $c rep $r" }
        $i++
    }
}

Write-Host "`n>>> session reference target (end)" -ForegroundColor Cyan
Read-Host "Load reference target and press Enter"
python capture.py REF top $SESSION $i --standoff $STANDOFF --note "session end reference"
if ($LASTEXITCODE -ne 0) { throw "Final reference capture failed" }

Write-Host "`n>>> running checks - DO NOT break down the rig yet" -ForegroundColor Yellow
$RepoRoot = Split-Path -Parent $PSScriptRoot
python check_captures.py "$RepoRoot/data/raw/$SESSION"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAll captures PASSED - safe to break down" -ForegroundColor Green
} else {
    Write-Host "`nSome captures FAILED - recapture before breaking down!" -ForegroundColor Red
}
