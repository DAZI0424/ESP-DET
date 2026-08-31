const fs = require("fs");

const source = fs.readFileSync("main/camera_web.c", "utf8");
const start = source.indexOf("static const char INDEX_HTML");
const end = source.indexOf("static esp_err_t", start);

if (start < 0 || end < 0) {
  throw new Error("INDEX_HTML block not found");
}

const block = source.slice(start, end);
const html = [...block.matchAll(/"(?:\\.|[^"\\])*"/g)]
  .map((match) => JSON.parse(match[0]))
  .join("");
const script = html.match(/<script>([\s\S]*?)<\/script>/);

if (!script) {
  throw new Error("embedded GUI script not found");
}

new Function(script[1]);
console.log(
  `embedded GUI JavaScript syntax passed: HTML ${html.length} bytes, JS ${script[1].length} bytes`,
);
