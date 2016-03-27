import atexit
from time import time
from datetime import timedelta, datetime


def elapsed_time_to_str(seconds):
    return str(timedelta(seconds=seconds))


def now_to_str(s):
    print(datetime.now().strftime("%d.%m.%Y %H:%M:%S"), '-', s)


def log(s, elapsed=None):
    line = "="*40
    print(line)
    now_to_str(s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)


def end_log():
    end = time()
    elapsed = end-start
    log("End Program", elapsed_time_to_str(elapsed))


def now():
    end = time()
    elapsed = end-start
    return elapsed_time_to_str(elapsed)

start = time()
atexit.register(end_log)
log("Start Program")
