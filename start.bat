@echo off
chcp 65001 >nul
title 智慧课堂考勤管理系统

cd /d "%~dp0"

if not exist ".venv" (
    echo [ERROR] 虚拟环境 .venv 不存在，请先运行 uv sync
    pause
    exit /b 1
)

echo ============================================
echo   智慧课堂考勤管理系统 - 正在启动...
echo   地址: http://127.0.0.1:8000
echo   按 Ctrl+C 停止服务
echo ============================================

.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload

pause
