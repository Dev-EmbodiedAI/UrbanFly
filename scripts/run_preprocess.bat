@echo off
REM UrbanFly OBJ 预处理流水线
cd /d %~dp0..

echo ============================================
echo  UrbanFly - OBJ Preprocessing Pipeline
echo ============================================
echo.
echo This will process the Paris city OBJ file and generate:
echo   1. Simplified city mesh (glTF, ~100K faces)
echo   2. 3D occupancy grid (for path planning)
echo   3. 2.5D heightmap
echo   4. Building metadata (for communication model)
echo   5. Scene configuration
echo.
echo WARNING: This may take 10-30 minutes and use 4-8GB of RAM!
echo.

set /p CONFIRM="Continue? (y/n): "
if /i not "%CONFIRM%"=="y" exit /b

echo.
echo Installing dependencies...
pip install trimesh numpy scipy Pillow -q

echo.
echo Running preprocessing pipeline...
python preprocess/preprocess_pipeline.py ^
    --obj "C:\Users\caste\Desktop\paris\Paris_city_only.obj" ^
    --output "data\scene" ^
    --ratio 0.01 ^
    --resolution 5.0

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed. Trying extract-only mode...
    python preprocess/preprocess_pipeline.py ^
        --obj "C:\Users\caste\Desktop\paris\Paris_city_only.obj" ^
        --output "data\scene" ^
        --extract-only
)

echo.
echo Done! Check data/scene/ for output files.
pause
