@echo off
title AI-Vision Communication Assistance for Paralyzed Individuals
echo =======================================================================
echo  AI-Vision-Based Communication Assistance for Paralyzed Individuals
echo  Zero-Calibration Hands-Free Eye Gaze and Head Movement System
echo =======================================================================
echo.
echo Starting application and launching browser...
python main.py
if errorlevel 1 (
    echo.
    echo [Notice] Application exited with an error.
    pause
)
