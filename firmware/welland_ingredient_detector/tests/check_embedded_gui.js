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

for (const id of ["takePhoto", "photoCount", "resetPhotoCount"]) {
  if (!html.includes(`id="${id}"`)) {
    throw new Error(`embedded GUI element missing: ${id}`);
  }
}

if (!script[1].includes("event.code!=='Space'") || !script[1].includes("takePhoto.click()")) {
  throw new Error("spacebar photo shortcut missing");
}

new Function(script[1]);

// Exercise the actual embedded renderer for the new non-locking outcome.
const title = {}, detail = {};
const renderBody = script[1].slice(script[1].indexOf("function renderRecognition(s)"),
  script[1].indexOf("async function poll()"));
const render = new Function("recognition", "recognitionTitle", "recognitionDetail",
  "renderControls", "renderDecision", "sessionArmed", renderBody + ";return renderRecognition;")(
  {}, title, detail, () => {}, () => {}, true);
render({recognition_state: "uncertain"});
if (!title.textContent.includes("调整角度") || !detail.textContent.includes("尚未锁定")) {
  throw new Error("uncertain results must request a new view instead of claiming a lock");
}
render({recognition_state: "locked", confidence_level: "medium",
  lock_reason: "three_frame_evidence", effective_inferences: 3});
if (!detail.textContent.includes("三帧证据达标") || detail.textContent.includes("强制")) {
  throw new Error("three-frame locks must describe accepted evidence");
}

const bootSource = fs.readFileSync("main/main.c", "utf8");
if (!bootSource.includes(".fb_count = 1,")) {
  throw new Error("VGA RGB565 must use one frame buffer for stable non-continuous capture");
}
if (!bootSource.includes(".grab_mode = CAMERA_GRAB_WHEN_EMPTY,")) {
  throw new Error("VGA RGB565 must avoid continuous CAMERA_GRAB_LATEST capture");
}
const cameraConfigSource = bootSource.slice(
  bootSource.indexOf("static esp_err_t configure_gc2145"),
  bootSource.indexOf("static void log_rgb565_statistics"),
);
if (cameraConfigSource.includes("sensor->set_hmirror") || cameraConfigSource.includes("sensor->set_vflip")) {
  throw new Error("GC2145 orientation must not use separate unreliable read-modify-write calls");
}
for (const token of [
  "GC2145_ANALOG_MODE1_BASE",
  "GC2145_HMIRROR_BIT",
  "GC2145_VFLIP_BIT",
  "GC2145_ANALOG_MODE1,\n                                               0xff,\n                                               orientation",
]) {
  if (!cameraConfigSource.includes(token)) {
    throw new Error(`GC2145 atomic orientation write missing: ${token}`);
  }
}
const appMain = bootSource.slice(bootSource.indexOf("void app_main(void)"));
const modelStart = appMain.indexOf("ingredient_detection_start()");
const webStart = appMain.indexOf("camera_web_start()");
const cameraStart = appMain.indexOf("esp_camera_init(&CAMERA_CONFIG)");
if (!(modelStart >= 0 && modelStart < webStart && webStart < cameraStart)) {
  throw new Error("model must reserve internal SRAM before Wi-Fi and camera initialization");
}
if (appMain.includes("ESP_ERROR_CHECK(")) {
  throw new Error("app_main startup failures must not trigger a reboot loop");
}
if (!appMain.includes('camera_web_set_camera_status(false, 0, "model_init_failed")')) {
  throw new Error("model startup failure must remain visible through the diagnostic GUI");
}

const detectorSource = fs.readFileSync("main/ingredient_detection.cpp", "utf8");
const contractCheck = detectorSource.slice(
  detectorSource.indexOf("bool validate_tensor_contract"),
  detectorSource.indexOf("class IngredientDetector"),
);
if (contractCheck.includes("std::abort()")) {
  throw new Error("tensor allocation failure must not abort and reboot the board");
}

const sdkDefaults = fs.readFileSync("sdkconfig.defaults", "utf8");
if (!sdkDefaults.includes("CONFIG_WELLAND_CAM_XCLK_MHZ=20")) {
  throw new Error("GC2145 XCLK must be configured at 20 MHz");
}
if (!sdkDefaults.includes("CONFIG_WELLAND_CAM_HMIRROR=y")) {
  throw new Error("GC2145 hardware horizontal mirroring must be enabled");
}
if (!sdkDefaults.includes("CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=0")) {
  throw new Error("small generic allocations must prefer PSRAM before the 96 KB SIMD arena is reserved");
}
if (!sdkDefaults.includes("# CONFIG_CAMERA_PSRAM_DMA is not set")) {
  throw new Error("VGA RGB565 must use internal DMA staging instead of direct PSRAM DMA");
}

console.log(
  `embedded GUI and boot order passed: HTML ${html.length} bytes, JS ${script[1].length} bytes`,
);
