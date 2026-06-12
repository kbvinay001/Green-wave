@echo off
cd /d "d:\project G\greenwave"
echo === DEMO CHAIN TEST: --demo --duration 10 ===
.venv\Scripts\python.exe integration\pipeline.py --demo --duration 10
echo === DEMO CHAIN TEST COMPLETE ===
