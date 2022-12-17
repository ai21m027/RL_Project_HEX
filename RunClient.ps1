param(
    [int]$jobNr
)

../venv/Scripts/Activate.ps1

python.exe .\CoachAssistant.py $jobNr > ./jobs/job_$jobNr/log.txt