@echo off
setlocal
cd /d "%~dp0"

if not exist "Datos\Septiembre2026.csv" (
    echo No se encontro Datos\Septiembre2026.csv
    exit /b 1
)

py -3.13 -m venv .venv-build
if errorlevel 1 (
    echo Se requiere Python 3.13 en el equipo de compilacion.
    exit /b 1
)

call ".venv-build\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b 1

pyinstaller --noconfirm --clean pizarra.spec
if errorlevel 1 exit /b 1

if not exist "dist\Pizarra\Datos" mkdir "dist\Pizarra\Datos"
copy /y "Datos\Septiembre2026.csv" "dist\Pizarra\Datos\Septiembre2026.csv" >nul
if errorlevel 1 exit /b 1

echo.
echo Compilacion terminada.
echo Entrega la carpeta dist\Pizarra completa al cliente.
echo El cliente puede reemplazar dist\Pizarra\Datos\Septiembre2026.csv.
endlocal
