$env:IDF_PATH = 'D:\.espressif\v6.0.2\esp-idf'
$env:IDF_TOOLS_PATH = 'C:\Espressif\tools'
$env:IDF_PYTHON_ENV_PATH = 'D:\ESP-Project\20260729ESP-DET\LVGL\tmp\idf-python'
$env:PYTHONPATH = 'C:\Espressif\tools\python\v6.0.2\venv\Lib\site-packages'
$env:ESP_IDF_VERSION = '6.0.2'
$env:ESP_ROM_ELF_DIR = 'C:\Espressif\tools\esp-rom-elfs\20241011'
$taskPython = 'D:\ESP-Project\20260729ESP-DET\LVGL\tmp\idf-python\Scripts\python.exe'
$env:PATH = 'C:\Users\WLPC\AppData\Roaming\uv\python\cpython-3.14.6-windows-x86_64-none;C:\Espressif\tools\cmake\4.0.3\bin;C:\Espressif\tools\ninja\1.12.1;C:\Espressif\tools\xtensa-esp-elf\esp-15.2.0_20251204\xtensa-esp-elf\bin;' + $env:PATH
& $taskPython "$env:IDF_PATH\tools\idf.py" -C "$PSScriptRoot\.." -B "$PSScriptRoot\..\..\build-vendor" -D "PYTHON=$taskPython" @args
exit $LASTEXITCODE


