"""Capture bounded serial diagnostics, optionally reset the CH340-connected board."""
import argparse
from pathlib import Path
import time
import sys
import serial

sys.stdout.reconfigure(errors='replace')

parser = argparse.ArgumentParser()
parser.add_argument('--port', default='COM5')
parser.add_argument('--seconds', type=float, default=25)
parser.add_argument('--reset', action='store_true')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'tmp/boot-animation.log')
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
port = serial.Serial(port=None, baudrate=115200, timeout=0.2)
port.dtr = False
port.rts = False
port.port = args.port
port.open()
try:
    if args.reset:
        port.rts = True
        time.sleep(0.1)
        port.rts = False
    end = time.monotonic() + args.seconds
    with args.output.open('wb') as log:
        while time.monotonic() < end:
            data = port.read(port.in_waiting or 1)
            if data:
                log.write(data)
                print(data.decode('utf-8', errors='replace'), end='', flush=True)
finally:
    port.close()
