const { spawnSync } = require('node:child_process');
const { join, resolve } = require('node:path');

const root = resolve(__dirname, '..');
const compiler = 'C:\\Espressif\\tools\\xtensa-esp-elf\\esp-15.2.0_20251204\\xtensa-esp-elf\\bin\\xtensa-esp32s3-elf-g++.exe';

const compile = spawnSync(compiler, [
  '-std=gnu++23', '-fsyntax-only', '-fconstexpr-ops-limit=1000000000',
  '-I', join(root, 'main'),
  join(root, 'tests', 'recognition_decision_constexpr_test.cpp'),
], { encoding: 'utf8' });

if (compile.status !== 0) {
  process.stderr.write(compile.stdout || '');
  process.stderr.write(compile.stderr || '');
  process.exit(compile.status || 1);
}
console.log('recognition decision compile-time tests passed: 25/25');
