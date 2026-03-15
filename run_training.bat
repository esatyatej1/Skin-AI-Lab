@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo       Skin-AI-Lab: AI Generation Tier Selector
echo ========================================================
echo Please choose your generation mode:
echo.
echo [1] Lite   - Super Fast (2-3 mins), Basic Quality (64x64)
echo [2] Normal - Balanced (5-8 mins), Good Quality (128x128)
echo [3] Heavy  - Maximum Clarity (15+ mins), Best Quality (128x128)
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" (
    set MODE=lite
) else if "%choice%"=="2" (
    set MODE=normal
) else if "%choice%"=="3" (
    set MODE=heavy
) else (
    echo Invalid choice. Defaulting to Normal mode...
    set MODE=normal
)

echo Starting %MODE% mode in Linux environment...
wsl -d Ubuntu-22.04 -u root bash /mnt/c/123/setup.sh %MODE%

echo.
echo Training process finished.
pause
