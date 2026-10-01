@echo off
REM ============================================================================
REM  Vision pipeline: raw datasets -> staged -> merged -> processed -> trained
REM  Run from the repository root:  scripts\prepare_and_train.bat
REM ============================================================================
setlocal
cd /d "%~dp0.."

REM src/ is imported as a package (from src.vision.model import ...), so the
REM repo root must be on PYTHONPATH when running the scripts by path.
set "PYTHONPATH=%CD%;%PYTHONPATH%"

echo ============================================================================
echo  [1/4] Convert and remap annotations -^> data\staged
echo ============================================================================
python src\vision\convert_and_remap.py
if errorlevel 1 goto :failed

echo.
echo ============================================================================
echo  [2/4] Merge, deduplicate, split -^> data\processed\vision
echo ============================================================================
python src\vision\merge_dedup_split.py
if errorlevel 1 goto :failed

echo.
echo ============================================================================
echo  [3/4] Train ResNet-50
echo ============================================================================
python src\vision\train.py --config configs\vision_train.yaml
if errorlevel 1 goto :failed

echo.
echo ============================================================================
echo  [4/4] Evaluate best checkpoint
echo ============================================================================
python src\vision\evaluate.py --model models\vision\best_resnet50.pth
if errorlevel 1 goto :failed

echo.
echo Pipeline complete.
echo   Staged      : data\staged\
echo   Merged      : data\merged\
echo   Processed   : data\processed\vision\
echo   Checkpoint  : models\vision\best_resnet50.pth
echo   Results     : results\vision\confusion_matrices\
endlocal
exit /b 0

:failed
echo.
echo *** Pipeline FAILED at the step above (exit code %errorlevel%). Stopping. ***
endlocal
exit /b 1
