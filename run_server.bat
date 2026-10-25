@echo off
cd /d "%~dp0"
set "GEMINI_REALTIME_MODEL=gemini-3.8-live"
rem Set GEMINI_REALTIME_API_KEY or GEMINI_REALTIME_API_KEYS before launching.
python server/run.py
