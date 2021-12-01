import psutil

for proc in psutil.process_iter():
    if proc.name() == "RocketLeague.exe":
        print(proc)
        proc.kill()