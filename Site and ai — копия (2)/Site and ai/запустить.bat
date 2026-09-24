@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo  Запуск методички...
echo  Не закрывайте это окно, пока работаете с сайтом.
echo  Чтобы остановить — закройте окно или нажмите Ctrl+C.
echo.

where powershell >nul 2>&1
if %errorlevel%==0 (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1"
  goto :end
)

where pwsh >nul 2>&1
if %errorlevel%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1"
  goto :end
)

echo  Не найден PowerShell — он есть в любой Windows 10/11.
echo  Откройте "Пуск", введите PowerShell и проверьте, что он запускается.
echo.
pause

:end
echo.
pause