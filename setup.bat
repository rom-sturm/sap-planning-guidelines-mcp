@echo off
setlocal EnableDelayedExpansion

:: Arbeitsverzeichnis auf den Ordner des Scripts setzen
cd /d "%~dp0"

echo === SAP Planning Guidelines MCP Server - Setup (Windows) ===
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python nicht gefunden. Bitte Python 3.10+ installieren: https://python.org
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/3] Erstelle virtuelles Environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] venv konnte nicht erstellt werden.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtuelles Environment bereits vorhanden.
)

echo [2/3] Aktualisiere pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip-Update fehlgeschlagen.
    pause
    exit /b 1
)

echo.
echo [2/3] Installiere Abhaengigkeiten (kann mehrere Minuten dauern)...
.venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Installation fehlgeschlagen.
    pause
    exit /b 1
)

set "PYTHON_PATH=%~dp0.venv\Scripts\python.exe"
set "SERVER_PATH=%~dp0server.py"

echo.
echo [3/3] Installation abgeschlossen!
echo.
echo ============================================================
echo  Claude Desktop Konfiguration:
echo  %APPDATA%\Claude\claude_desktop_config.json
echo ============================================================
echo {
echo   "mcpServers": {
echo     "sac-planning-guidelines": {
echo       "command": "%PYTHON_PATH:\=\\%",
echo       "args": ["%SERVER_PATH:\=\\%"]
echo     }
echo   }
echo }
echo ============================================================
echo.
pause
