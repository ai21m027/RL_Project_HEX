param(
    [int]$workerNr
)

../venv/Scripts/Activate.ps1

python.exe .\DistributedMCTSClient.py $workerNr > ./worker_$workerNr/log.txt