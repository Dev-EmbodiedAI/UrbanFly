@echo off
REM UrbanFly 一键启动脚本
echo ============================================
echo  UrbanFly - Multi-UAV Urban Delivery System
echo ============================================

REM 启动后端 (新窗口)
start "UrbanFly Backend" cmd /c "%~dp0run_server.bat"

REM 等待后端启动
timeout /t 3 /nobreak >nul

REM 启动前端 (新窗口)
start "UrbanFly Frontend" cmd /c "%~dp0run_frontend.bat"

echo.
echo Backend:  http://localhost:8765
echo Frontend: http://localhost:5173
echo WebSocket: ws://localhost:8765/ws
echo.
echo 按任意键退出...
pause >nul
