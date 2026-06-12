@echo off
cd /d "d:\project G\greenwave"
echo === GATE TEST: --virtual --wav siren_15s.wav --sumo benz.sumocfg --duration 20 ===
.venv\Scripts\python.exe integration\pipeline.py --virtual --wav data\virtual_demo\siren_15s.wav --sumo sim\nets\benz_circle\benz.sumocfg --duration 20
echo === GATE TEST COMPLETE ===
