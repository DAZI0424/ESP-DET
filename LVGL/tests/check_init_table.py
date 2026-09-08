from pathlib import Path
import re
root = Path(__file__).resolve().parents[1]
expected = []
for line in next(root.glob('*SPI&MCU*.txt')).read_text(encoding='utf-8-sig').splitlines():
    line = line.split('//')[0].strip()
    if line.startswith('0x'):
        values = [int(v, 16) for v in re.findall(r'0x[0-9a-fA-F]+', line)]
        expected.append(('Comm', values[0]))
        expected.extend(('Data', v) for v in values[1:])
    elif line.startswith('Delay'):
        expected.append(('Delay', int(re.search(r'\d+', line)[0])))
code = (root/'main/lcd.c').read_text(encoding='utf-8-sig')
init = code.split('// BEGIN VENDOR GC9B72 INIT')[1].split('// END VENDOR GC9B72 INIT')[0]
actual = []
for line in init.splitlines():
    if m := re.search(r'SPI_Write(Comm|Data)\((0x[\da-fA-F]+)\)', line):
        actual.append((m[1], int(m[2], 16)))
    elif m := re.search(r'pdMS_TO_TICKS\((\d+)\)', line):
        actual.append(('Delay', int(m[1])))
assert actual == expected, 'Port differs from the supplied GC9B72 SPI/MCU table'
print(f'PASS: {sum(k == "Comm" for k, _ in actual)} commands and all data/delays match vendor source')
