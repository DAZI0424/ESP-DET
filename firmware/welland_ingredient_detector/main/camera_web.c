#include "camera_web.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_check.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_netif_ip_addr.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "img_converters.h"
#include "ingredient_detection.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#define SENSOR_FRAME_WIDTH 640U
#define SENSOR_FRAME_HEIGHT 480U
#define OUTPUT_FRAME_SIZE INGREDIENT_MODEL_WIDTH
#define OUTPUT_FRAME_BYTES (OUTPUT_FRAME_SIZE * OUTPUT_FRAME_SIZE * sizeof(uint16_t))
#define AI_TASK_CORE 1
#define AI_TASK_STACK_SIZE 12288
#define ACTIVE_PREVIEW_INTERVAL_MS 125U

_Static_assert(OUTPUT_FRAME_SIZE == INGREDIENT_MODEL_HEIGHT,
               "GUI stream and model input dimensions must match");
_Static_assert(SENSOR_FRAME_WIDTH >= SENSOR_FRAME_HEIGHT &&
                   SENSOR_FRAME_HEIGHT >= OUTPUT_FRAME_SIZE,
               "GC2145 frame must contain a square crop at model resolution");

typedef struct {
    uint8_t *data;
    size_t length;
    size_t capacity;
    bool failed;
    bool fixed_capacity;
} jpeg_buffer_t;

typedef enum {
    AI_JOB_VALIDATION,
} ai_job_kind_t;

typedef struct {
    ai_job_kind_t kind;
    SemaphoreHandle_t done;
    const uint8_t *validation_input;
    ingredient_detection_result_t *result;
    ingredient_performance_t *performance;
    esp_err_t *status;
} ai_job_t;

static const char *TAG = "camera_web";
static const char STREAM_BOUNDARY[] = "\r\n--frame\r\n";
static const char STREAM_TYPE[] = "multipart/x-mixed-replace;boundary=frame";
static uint16_t s_sensor_pid;
static bool s_camera_ready;
static const char *s_camera_state = "booting";
static char s_station_ip[16] = "not-connected";
static uint8_t *s_output_frame;
static httpd_handle_t s_ui_server;
static httpd_handle_t s_stream_server;
static SemaphoreHandle_t s_stream_mutex;
static QueueHandle_t s_ai_queue;
static TaskHandle_t s_ai_task;
static portMUX_TYPE s_state_lock = portMUX_INITIALIZER_UNLOCKED;
static ingredient_detection_result_t s_detection;

static SemaphoreHandle_t s_candidate_mutex;
static SemaphoreHandle_t s_preview_mutex;
static ingredient_frame_t s_candidate;
static bool s_candidate_ready;
static bool s_pipeline_active;
static uint32_t s_preview_sequence;
static TaskHandle_t s_capture_task;

static const char INDEX_HTML[] =
    "<!doctype html><html lang=\"zh-CN\"><head>"
    "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>食材识别</title><style>"
    "*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#eef2ff;"
    "font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif}"
    "main{width:min(94vw,1040px);margin:auto;padding:24px 0}"
    ".card{background:#151c31;border:1px solid #2a3554;border-radius:18px;padding:20px;"
    "box-shadow:0 18px 50px #0007}h1{font-size:24px;margin:0 0 4px}"
    ".sub,.state{color:#9fb0d3}.camera{display:block;width:224px;height:224px;"
    "min-width:224px;max-width:224px;min-height:224px;max-height:224px;aspect-ratio:1/1;"
    "margin:18px auto;background:#050811;border-radius:12px;object-fit:contain}"
    ".capture-tools{display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;margin:-8px 0 16px}"
    ".capture-tools button{border:0;border-radius:10px;padding:9px 13px;background:#2f70e8;color:#fff;"
    "font:inherit;font-weight:700;cursor:pointer}.capture-tools button.secondary{background:#35415e}"
    ".capture-tools button:disabled{opacity:.55;cursor:wait}.capture-count{display:flex;align-items:center;gap:4px;"
    "padding:8px 11px;border:1px solid #35415e;border-radius:10px;color:#b8c5e2;background:#10182b}"
    ".capture-count strong{color:#84f1b7;font-size:18px}.capture-tools small{width:100%;text-align:center;color:#9fb0d3}"
    ".metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}"
    ".metric,.results{background:#0e1528;border:1px solid #2a3554;border-radius:12px;padding:12px}"
    ".metric span{display:block;color:#9fb0d3;font-size:13px}.metric strong{font-size:21px;color:#84f1b7}"
    ".metric small{display:block;color:#7889ad;font-size:11px;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
    ".results{margin-top:10px}.results h2{font-size:15px;margin:0 0 8px;color:#9fb0d3}"
    ".recognition{margin-top:10px;padding:12px;border:1px solid #2a3554;border-radius:12px;"
    "background:#0e1528}.recognition strong,.recognition span{display:block}"
    ".recognition span{margin-top:3px;color:#9fb0d3;font-size:13px}"
    ".recognition.confirming strong{color:#ffd479}.recognition.locked{border-color:#2f8f68;"
    "background:#10271f}.recognition.locked strong{color:#84f1b7}"
    ".recognition.weighing{border-color:#477ee7;background:#101f3b}.recognition.weighing strong{color:#8fc5ff}"
    ".evidence-steps{display:flex;align-items:center;gap:8px;margin:12px 0}"
    ".evidence-steps i{display:grid;place-items:center;width:30px;height:30px;border-radius:50%;font-style:normal;"
    "font-weight:800;background:#202b46;color:#7181a5;border:1px solid #34415f}"
    ".evidence-steps b{height:2px;flex:1;background:#34415f}.evidence-steps i.used{background:#2f70e8;"
    "border-color:#5a91f0;color:white}.evidence-steps i.locked{background:#21845e;border-color:#51c894}"
    ".decision-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}"
    ".actions{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:12px}.actions button{width:100%;border:0;border-radius:12px;padding:12px 16px;"
    "background:#2f70e8;color:white;font:inherit;font-weight:700;cursor:pointer}"
    ".actions button:hover{background:#3b7df2}.actions button:disabled{opacity:.55;cursor:wait}"
    ".actions button.secondary{background:#35415e}.actions button.success{background:#21845e}"
    ".actions button.danger{background:#8d3f4a}.hidden{display:none!important}"
    ".records{margin-top:16px;background:#0e1528;border:1px solid #2a3554;border-radius:12px;padding:12px}"
    ".records-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}"
    ".record-actions{display:flex;gap:8px}"
    ".records h2{font-size:16px;margin:0}.records button{border:0;border-radius:9px;padding:8px 12px;"
    "background:#2f70e8;color:#fff;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}"
    ".records button.secondary{background:#35415e}.records button.secondary:hover{background:#465473}"
    ".records button:disabled{opacity:.5;cursor:not-allowed}.table-wrap{overflow:auto;max-height:360px}"
    "table{width:100%;min-width:820px;border-collapse:collapse;font-size:14px}"
    "th,td{padding:9px 10px;border-bottom:1px solid #2a3554;text-align:left}"
    "th{position:sticky;top:0;background:#18223a;color:#b8c5e2;z-index:1}"
    "td input{width:100%;min-width:110px;border:1px solid #405078;border-radius:7px;padding:6px 8px;"
    "background:#10182b;color:#eef2ff;font:inherit}.empty{text-align:center;color:#9fb0d3}"
    "ul{margin:0;padding-left:20px}li{margin:5px 0}.error{color:#ff9a9a}"
    "code{color:#8fc5ff}@media(max-width:680px){.metrics{grid-template-columns:1fr 1fr}.decision-grid{grid-template-columns:1fr 1fr}"
    ".actions{grid-template-columns:1fr}}</style></head><body><main><section class=\"card\">"
    "<h1>ESP32-S3 食材识别</h1>"
    "<div class=\"sub\">GC2145 · ESPDet INT8/INT16 · 224×224</div>"
    "<img id=\"camera\" class=\"camera\" width=\"224\" height=\"224\" alt=\"摄像头实时画面\">"
    "<div class=\"capture-tools\"><button id=\"chooseCaptureFolder\" class=\"secondary\" type=\"button\">选择保存文件夹</button>"
    "<button id=\"toggleDetectionBoxes\" class=\"secondary\" type=\"button\">关闭识别框</button>"
    "<button id=\"takePhoto\" type=\"button\" title=\"快捷键：空格\">拍照保存（空格）</button>"
    "<span class=\"capture-count\">拍照次数：<strong id=\"photoCount\">0</strong></span>"
    "<button id=\"resetPhotoCount\" class=\"secondary\" type=\"button\">次数清零</button>"
    "<small id=\"captureFolder\">尚未选择文件夹</small></div>"
    "<div class=\"metrics\"><div class=\"metric\"><span>识别响应耗时</span>"
    "<strong id=\"latency\">-- ms</strong></div><div class=\"metric\"><span>完整操作耗时</span>"
    "<strong id=\"fullLatency\">-- ms</strong></div><div class=\"metric\"><span>目标数量</span>"
    "<strong id=\"count\">--</strong></div></div>"
    "<div id=\"recognition\" class=\"recognition searching\"><strong id=\"recognitionTitle\">识别中</strong>"
    "<span id=\"recognitionDetail\">请将食材保持在画面中</span></div>"
    "<div id=\"evidenceSteps\" class=\"evidence-steps\"><i>1</i><b></b><i>2</i><b></b><i>3</i></div>"
    "<div class=\"decision-grid\">"
    "<div class=\"metric\"><span>画面质量门控</span><strong id=\"frameQuality\">--</strong><small id=\"frameQualityDetail\">等待画面</small></div>"
    "<div class=\"metric\"><span>有效证据帧</span><strong id=\"frameProgress\">0 / 3</strong><small>不合格和重复帧不计数</small></div>"
    "<div class=\"metric\"><span>累计证据第一名</span><strong id=\"candidateEvidence\">--</strong><small id=\"candidateSecond\">第二名：--</small></div>"
    "<div class=\"metric\"><span>累计证据差值</span><strong id=\"evidenceMargin\">--</strong><small>第二帧提前锁定阈值 15%</small></div>"
    "<div class=\"metric\"><span>一帧 / 两帧 / 三帧比例</span><strong id=\"outputMix\">--</strong><small>高 / 中 / 低可信度</small></div>"
    "<div class=\"metric\"><span>平均有效推理帧数</span><strong id=\"averageFrames\">--</strong><small id=\"completedLocks\">已完成 0 次</small></div></div>"
    "<div class=\"results\"><h2>识别内容</h2><ul id=\"results\"><li>等待第一帧识别</li></ul></div>"
    "<div class=\"actions\"><button id=\"restart\" type=\"button\">重新识别</button>"
    "<button id=\"startWeighing\" class=\"success hidden\" type=\"button\">开始称重</button>"
    "<button id=\"completeWeighing\" class=\"success hidden\" type=\"button\">称重完成</button>"
    "<button id=\"cancelRecognition\" class=\"danger hidden\" type=\"button\">取消本次</button></div>"
    "<section class=\"records\"><div class=\"records-head\"><h2>识别记录</h2>"
    "<div class=\"record-actions\"><button id=\"clearRecords\" class=\"secondary\" type=\"button\">清空记录</button>"
    "<button id=\"downloadExcel\" type=\"button\">下载 Excel</button></div></div>"
    "<div class=\"table-wrap\"><table><thead><tr><th>识别序列号</th><th>识别种类</th>"
    "<th>可信度</th><th>有效帧数</th><th>识别响应耗时</th><th>完整操作耗时</th><th>真实食材</th></tr></thead><tbody id=\"recordsBody\"></tbody>"
    "</table></div><datalist id=\"ingredientOptions\"></datalist></section>"
    "<p id=\"state\" class=\"state\">正在连接摄像头…</p>"
    "<p class=\"sub\">主 Wi-Fi：<code>" CONFIG_WELLAND_CAM_STA_SSID
    "</code> · 备用热点：<code>" CONFIG_WELLAND_CAM_AP_SSID "</code></p>"
    "</section></main><script>"
    "const labels=['苹果','香蕉','草莓','生菜','鸡蛋','橙子','茄子','黄瓜','胡萝卜','玉米'];"
    "const camera=document.getElementById('camera'),state=document.getElementById('state');"
    "const latency=document.getElementById('latency'),fullLatency=document.getElementById('fullLatency'),count=document.getElementById('count');"
    "const results=document.getElementById('results'),recognition=document.getElementById('recognition');"
    "const recognitionTitle=document.getElementById('recognitionTitle');"
    "const recognitionDetail=document.getElementById('recognitionDetail'),restart=document.getElementById('restart');"
    "const startWeighing=document.getElementById('startWeighing'),completeWeighing=document.getElementById('completeWeighing');"
    "const cancelRecognition=document.getElementById('cancelRecognition'),evidenceSteps=document.getElementById('evidenceSteps');"
    "const frameQuality=document.getElementById('frameQuality'),frameQualityDetail=document.getElementById('frameQualityDetail');"
    "const frameProgress=document.getElementById('frameProgress'),candidateEvidence=document.getElementById('candidateEvidence');"
    "const candidateSecond=document.getElementById('candidateSecond'),evidenceMargin=document.getElementById('evidenceMargin');"
    "const outputMix=document.getElementById('outputMix'),averageFrames=document.getElementById('averageFrames');"
    "const completedLocks=document.getElementById('completedLocks');"
    "const recordsBody=document.getElementById('recordsBody'),downloadExcel=document.getElementById('downloadExcel');"
    "const clearRecords=document.getElementById('clearRecords');"
    "const chooseCaptureFolder=document.getElementById('chooseCaptureFolder'),takePhoto=document.getElementById('takePhoto');"
    "const photoCount=document.getElementById('photoCount'),resetPhotoCount=document.getElementById('resetPhotoCount');"
    "const toggleDetectionBoxes=document.getElementById('toggleDetectionBoxes'),captureFolder=document.getElementById('captureFolder');"
    "const ingredientOptions=document.getElementById('ingredientOptions'),recordsKey='welland-recognition-records-v1';"
    "const photoCountKey='welland-photo-count-v1';"
    "let restarting=false,actionBusy=false,sessionArmed=false,sessionTiming=false,sessionToken=0,experienceElapsedMs=0;"
    "let timerStartedAt=0,fullFlowStartedAt=0,fullFlowElapsedMs=0,timerHandle=0;"
    "let captureDirectory=null,captureBusy=false,latestStatus=null,detectionBoxesEnabled=true,overlayBusy=false;"
    "let photoCountValue=loadPhotoCount();"
    "let records=loadRecords();ingredientOptions.innerHTML=labels.map(x=>'<option value=\"'+x+'\">').join('');"
    "function loadPhotoCount(){try{const value=Number(localStorage.getItem(photoCountKey)||0);"
    "return Number.isSafeInteger(value)&&value>=0?value:0}catch(e){return 0}}"
    "function renderPhotoCount(){photoCount.textContent=String(photoCountValue);"
    "resetPhotoCount.disabled=captureBusy||photoCountValue===0}"
    "function savePhotoCount(){try{localStorage.setItem(photoCountKey,String(photoCountValue))}catch(e){"
    "state.className='state error';state.textContent='拍照次数无法保存：'+e.message}}"
    "function incrementPhotoCount(){photoCountValue++;savePhotoCount();renderPhotoCount()}renderPhotoCount();"
    "function loadRecords(){try{const value=JSON.parse(localStorage.getItem(recordsKey)||'[]');"
    "return Array.isArray(value)?value:[]}catch(e){return []}}"
    "function saveRecords(){try{localStorage.setItem(recordsKey,JSON.stringify(records))}catch(e){"
    "state.className='state error';state.textContent='识别记录无法保存：'+e.message}}"
    "function renderRecords(){recordsBody.innerHTML='';downloadExcel.disabled=!records.length;clearRecords.disabled=!records.length;"
    "if(!records.length){const tr=document.createElement('tr'),td=document.createElement('td');"
    "td.colSpan=7;td.className='empty';td.textContent='完成一次识别锁定后将自动生成记录';"
    "tr.appendChild(td);recordsBody.appendChild(tr);return}"
    "records.forEach((item,index)=>{const tr=document.createElement('tr');"
    "[item.serial,item.category,item.confidence||'--',item.frames||'--',item.elapsedMs+' ms',(item.flowElapsedMs||item.elapsedMs)+' ms'].forEach(value=>{const td=document.createElement('td');"
    "td.textContent=value;tr.appendChild(td)});const td=document.createElement('td'),input=document.createElement('input');"
    "input.setAttribute('list','ingredientOptions');input.placeholder='请输入或选择';input.value=item.actual||'';"
    "input.onchange=()=>{records[index].actual=input.value.trim();saveRecords()};td.appendChild(input);"
    "tr.appendChild(td);recordsBody.appendChild(tr)})}renderRecords();"
    "function showElapsed(value,suffix){latency.textContent=Math.max(0,Math.round(value))+' ms'+(suffix||'')}"
    "function armExperience(){sessionToken++;sessionArmed=true;sessionTiming=false;experienceElapsedMs=0;"
    "timerStartedAt=0;fullFlowStartedAt=performance.now();fullFlowElapsedMs=0;clearInterval(timerHandle);"
    "latency.textContent='等待有效食材帧';fullLatency.textContent='0 ms · 操作中';"
    "timerHandle=setInterval(()=>{if(!sessionArmed)return;fullFlowElapsedMs=performance.now()-fullFlowStartedAt;"
    "fullLatency.textContent=Math.round(fullFlowElapsedMs)+' ms · 操作中';"
    "if(sessionTiming){experienceElapsedMs=performance.now()-timerStartedAt;showElapsed(experienceElapsedMs,' · 识别中')}},50)}"
    "function startExperienceTiming(s){if(sessionTiming)return;"
    "experienceElapsedMs=s.recognition_elapsed_ms||0;sessionTiming=true;"
    "timerStartedAt=performance.now()-experienceElapsedMs;"
    "showElapsed(experienceElapsedMs,' · 识别中')}"
    "function updateExperience(s){if(!sessionArmed)return;"
    "if(!sessionTiming&&s.experience_started)startExperienceTiming(s);"
    "else if(sessionTiming&&s.experience_started&&!s.experience_locked){"
    "experienceElapsedMs=s.recognition_elapsed_ms||experienceElapsedMs;"
    "timerStartedAt=performance.now()-experienceElapsedMs;showElapsed(experienceElapsedMs,' · 识别中')}"
    "if(!sessionTiming)latency.textContent='等待有效食材帧';"
    "const stopped=s.recognition_state==='locked'||s.recognition_state==='weighing';"
    "if(stopped&&s.detections.length)stopExperience(s)}"
    "function stopExperience(s){if(!sessionArmed||!sessionTiming||!s.experience_started||!s.detections.length)return;"
    "experienceElapsedMs=s.recognition_elapsed_ms||performance.now()-timerStartedAt;"
    "fullFlowElapsedMs=performance.now()-fullFlowStartedAt;"
    "const elapsedMs=Math.round(experienceElapsedMs);"
    "sessionArmed=false;sessionTiming=false;clearInterval(timerHandle);"
    "showElapsed(elapsedMs,' · 已锁定');fullLatency.textContent=Math.round(fullFlowElapsedMs)+' ms · 已完成';const detection=s.detections[0];"
    "const trust={high:'高',medium:'中',low:'低'}[s.confidence_level]||'--';"
    "records.push({serial:records.length?Number(records[records.length-1].serial)+1:1,"
    "category:labels[detection.category]||('类别 '+detection.category),confidence:trust,"
    "frames:s.effective_inferences||0,elapsedMs:elapsedMs,flowElapsedMs:Math.round(fullFlowElapsedMs),actual:''});"
    "saveRecords();renderRecords()}"
    "function cancelExperience(){sessionArmed=false;sessionTiming=false;experienceElapsedMs=0;"
    "timerStartedAt=0;fullFlowStartedAt=0;fullFlowElapsedMs=0;clearInterval(timerHandle);"
    "latency.textContent='-- ms';fullLatency.textContent='-- ms'}"
    "function captureFileName(){const d=new Date(),pad=(v,n=2)=>String(v).padStart(n,'0');"
    "const category=latestStatus&&latestStatus.detections&&latestStatus.detections.length?"
    "'_c'+latestStatus.detections[0].category:'';return 'gc2145_'+d.getFullYear()+pad(d.getMonth()+1)+pad(d.getDate())"
    "+'_'+pad(d.getHours())+pad(d.getMinutes())+pad(d.getSeconds())+'_'+pad(d.getMilliseconds(),3)+category+'.jpg'}"
    "async function pickCaptureFolder(){if(!('showDirectoryPicker' in window)){"
    "captureFolder.textContent='当前页面无法直接选择固定文件夹，将使用浏览器下载目录';"
    "state.className='state error';state.textContent='浏览器未开放目录写入；可在浏览器设置中开启“下载前询问保存位置”';return false}"
    "try{captureDirectory=await window.showDirectoryPicker({mode:'readwrite'});"
    "captureFolder.textContent='保存到：'+captureDirectory.name;state.className='state';"
    "state.textContent='照片保存文件夹已选择';return true}catch(e){if(e.name!=='AbortError'){"
    "state.className='state error';state.textContent='选择文件夹失败：'+e.message}return false}}"
    "function displayedFrameBlob(){return new Promise((resolve,reject)=>{try{"
    "if(!camera.complete||!camera.naturalWidth)throw Error('实时画面尚未就绪');"
    "const canvas=document.createElement('canvas');canvas.width=camera.naturalWidth;canvas.height=camera.naturalHeight;"
    "const context=canvas.getContext('2d');context.drawImage(camera,0,0,canvas.width,canvas.height);"
    "canvas.toBlob(blob=>blob?resolve(blob):reject(Error('JPEG 编码失败')),'image/jpeg',0.94)}"
    "catch(e){reject(e)}})}"
    "async function savePhotoBlob(blob,name){if(captureDirectory){"
    "let permission=await captureDirectory.queryPermission({mode:'readwrite'});"
    "if(permission!=='granted')permission=await captureDirectory.requestPermission({mode:'readwrite'});"
    "if(permission!=='granted')throw Error('没有所选文件夹的写入权限');"
    "const file=await captureDirectory.getFileHandle(name,{create:true}),writer=await file.createWritable();"
    "await writer.write(blob);await writer.close();return '已保存到 '+captureDirectory.name+' / '+name}"
    "if('showSaveFilePicker' in window){const file=await window.showSaveFilePicker({suggestedName:name,"
    "types:[{description:'JPEG 图片',accept:{'image/jpeg':['.jpg','.jpeg']}}]});const writer=await file.createWritable();"
    "await writer.write(blob);await writer.close();return '照片已保存：'+name}"
    "const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;"
    "document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);"
    "return '照片已下载：'+name}"
    "function renderBoxToggle(enabled){detectionBoxesEnabled=!!enabled;"
    "toggleDetectionBoxes.textContent=detectionBoxesEnabled?'关闭识别框':'开启识别框';"
    "toggleDetectionBoxes.setAttribute('aria-pressed',detectionBoxesEnabled?'true':'false')}"
    "toggleDetectionBoxes.onclick=async()=>{if(overlayBusy||captureBusy)return;overlayBusy=true;"
    "toggleDetectionBoxes.disabled=true;takePhoto.disabled=true;const enabled=!detectionBoxesEnabled;"
    "toggleDetectionBoxes.textContent='正在切换…';try{const response=await fetch('/overlay/boxes?enabled='+(enabled?'1':'0'),{method:'POST'});"
    "if(!response.ok)throw Error((await response.text())||('HTTP '+response.status));const value=await response.json();"
    "renderBoxToggle(value.enabled);state.className='state';state.textContent=value.enabled?'识别框已开启':'识别框已关闭，照片将不包含识别框';"
    "await new Promise(resolve=>setTimeout(resolve,700))}catch(e){state.className='state error';"
    "state.textContent='识别框切换失败：'+e.message;renderBoxToggle(detectionBoxesEnabled)}finally{overlayBusy=false;"
    "toggleDetectionBoxes.disabled=false;takePhoto.disabled=captureBusy}};"
    "chooseCaptureFolder.onclick=()=>pickCaptureFolder();resetPhotoCount.onclick=()=>{if(captureBusy)return;"
    "photoCountValue=0;savePhotoCount();renderPhotoCount();state.className='state';state.textContent='拍照次数已清零'};"
    "takePhoto.onclick=async()=>{if(captureBusy)return;"
    "if('showDirectoryPicker' in window&&!captureDirectory&&!(await pickCaptureFolder()))return;"
    "captureBusy=true;takePhoto.disabled=true;chooseCaptureFolder.disabled=true;toggleDetectionBoxes.disabled=true;const original=takePhoto.textContent;"
    "renderPhotoCount();"
    "takePhoto.textContent='正在拍照…';try{const name=captureFileName(),blob=await displayedFrameBlob();"
    "const message=await savePhotoBlob(blob,name);incrementPhotoCount();state.className='state';state.textContent=message}"
    "catch(e){if(e.name!=='AbortError'){state.className='state error';state.textContent='拍照失败：'+e.message}}"
    "finally{captureBusy=false;takePhoto.disabled=overlayBusy;chooseCaptureFolder.disabled=false;"
    "toggleDetectionBoxes.disabled=overlayBusy;takePhoto.textContent=original;renderPhotoCount()}};"
    "document.addEventListener('keydown',event=>{if(event.code!=='Space'||event.repeat)return;"
    "const target=event.target,tag=target&&target.tagName;"
    "if(target&&target.isContentEditable||tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT'||tag==='BUTTON')return;"
    "event.preventDefault();if(!takePhoto.disabled)takePhoto.click()});"
    "if(!('showDirectoryPicker' in window))captureFolder.textContent='使用浏览器下载目录；可开启“下载前询问保存位置”';"
    "function connect(){camera.crossOrigin='anonymous';camera.src=location.protocol+'//'+location.hostname+':81/stream?t='+Date.now()}"
    "camera.onload=()=>{state.className='state';state.textContent='实时画面已连接'};"
    "camera.onerror=()=>{state.className='state error';state.textContent='画面断开，正在重连…';"
    "setTimeout(connect,1000)};connect();"
    "function pct(value){return Math.round((Number(value)||0)*100)+'%'}"
    "function categoryName(value){return value>=0&&value<labels.length?labels[value]:'--'}"
    "function renderControls(s){const mode=s.recognition_state,busy=actionBusy||restarting||!s.camera_ready;"
    "restart.disabled=busy;startWeighing.disabled=busy;completeWeighing.disabled=busy;cancelRecognition.disabled=busy;"
    "startWeighing.classList.toggle('hidden',mode!=='locked');"
    "completeWeighing.classList.toggle('hidden',mode!=='weighing');"
    "cancelRecognition.classList.toggle('hidden',mode==='searching')}"
    "function renderDecision(s){const q=s.quality||{},mode=s.recognition_state;"
    "const stopped=mode==='locked'||mode==='weighing';"
    "frameQuality.textContent=stopped?'已停止':s.frame_quality_ok?'合格':q.scene_change?'场景切换':!s.frame_stable?'稳定确认中':!s.frame_fresh?'重复帧':'未通过';"
    "frameQuality.style.color=stopped||s.frame_quality_ok?'#84f1b7':'#ffd479';"
    "frameQualityDetail.textContent='权重 '+(q.weight_permille||0)+'‰ · 运动 '+(q.motion_permille||0)"
    "+'‰ · 清晰 '+(q.sharpness||0)+' · 亮度 '+(q.mean_luma||0)+' · 全局变化 '+(q.changed_permille||0)+'‰';"
    "frameProgress.textContent=(s.collected_frames||0)+' / '+(s.window_frames||3);"
    "candidateEvidence.textContent=categoryName(s.candidate_category)+(s.candidate_category>=0?' '+pct(s.candidate_score):'');"
    "candidateSecond.textContent='第二名：'+categoryName(s.candidate_second_category);"
    "evidenceMargin.textContent=pct(s.evidence_margin);"
    "const mix=s.output_ratios||{};outputMix.textContent=pct(mix.one_frame)+' / '+pct(mix.two_frame)+' / '+pct(mix.three_frame);"
    "averageFrames.textContent=s.completed_locks?Number(s.average_lock_inferences).toFixed(2):'--';"
    "completedLocks.textContent='已完成 '+(s.completed_locks||0)+' 次锁定';"
    "const used=stopped?s.effective_inferences:s.collected_frames;"
    "evidenceSteps.querySelectorAll('i').forEach((step,index)=>{step.className=index<used?(stopped?'locked':'used'):''})}"
    "function renderRecognition(s){const mode=s.recognition_state;recognition.className='recognition '+mode;"
    "renderControls(s);renderDecision(s);"
    "if(mode==='weighing'){recognitionTitle.textContent='类别已锁定，正在称重';"
    "recognitionDetail.textContent='移动到秤面不会改变类别；称重完成后点击“称重完成”复位';return}"
    "if(mode==='locked'){recognitionTitle.textContent='识别已锁定';"
    "const trust={high:'高',medium:'中',low:'低'}[s.confidence_level]||'--';"
    "const reason={single_high:'单帧高置信',two_frame_evidence:'两帧累计证据',three_frame_evidence:'三帧证据达标'}[s.lock_reason]||'已锁定';"
    "recognitionDetail.textContent=trust+'可信度 · '+reason+' · '+s.effective_inferences+' 个有效推理帧；请移到秤面'}"
    "else if(mode==='uncertain'){recognitionTitle.textContent='暂不确定，请调整角度';"
    "recognitionDetail.textContent='三帧证据不足或存在多个目标，尚未锁定；请调整角度或只展示一种食材'}"
    "else if(mode==='confirming'){recognitionTitle.textContent='正在累计识别证据';"
    "recognitionDetail.textContent=s.collected_frames===1?"
    "'已累计第 1 个有效帧，等待内容不同且质量合格的新帧':"
    "'两帧证据尚未达标，继续检查同一食材；第三帧仍不确定时不会强制锁定'}"
    "else if(!s.frame_stable){recognitionTitle.textContent='等待展示区域稳定';"
    "recognitionDetail.textContent='背景运动不会启动识别；请把食材稳定展示在中央区域'}"
    "else if(!s.frame_fresh){recognitionTitle.textContent='等待新鲜画面';"
    "recognitionDetail.textContent='重复摄像头帧不会作为独立证据'}"
    "else if(!s.frame_quality_ok){recognitionTitle.textContent='画面质量未通过';"
    "recognitionDetail.textContent='请改善清晰度或曝光；该帧不会累计证据'}"
    "else{recognitionTitle.textContent=sessionArmed?'等待食材识别':'识别中';"
    "recognitionDetail.textContent='请在前置摄像头前稳定展示食材'}"
    "}"
    "async function poll(){const token=sessionToken;try{const r=await fetch('/status?t='+Date.now(),{cache:'no-store'});"
    "if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();if(token!==sessionToken||restarting)return;latestStatus=s;"
    "if(!overlayBusy)renderBoxToggle(s.detection_boxes_enabled!==false);"
    "const stopped=s.recognition_state==='locked'||s.recognition_state==='weighing';"
    "updateExperience(s);"
    "count.textContent=s.detections.length;"
    "results.innerHTML=stopped&&s.detections.length?s.detections.map(d=>"
    "'<li>'+labels[d.category]+' · '+Math.round(d.score*100)+'% · '+({high:'高',medium:'中',low:'低'}[s.confidence_level]||'--')+'可信度</li>').join(''):"
    "s.recognition_state==='confirming'?'<li>累计领先：'+categoryName(s.candidate_category)+' · '+pct(s.candidate_score)+'</li>':"
    "'<li>'+(s.frame_quality_ok?'等待识别到食材':'当前帧不累计证据')+'</li>';renderRecognition(s)}"
    "catch(e){state.className='state error';state.textContent='状态读取失败：'+e.message}"
    "finally{setTimeout(poll,sessionArmed?100:500)}}poll();"
    "async function postCommand(url,button,busyText){if(actionBusy)return false;actionBusy=true;"
    "const original=button.textContent;button.textContent=busyText;"
    "[restart,startWeighing,completeWeighing,cancelRecognition].forEach(item=>item.disabled=true);"
    "try{const r=await fetch(url,{method:'POST'});if(!r.ok)throw Error((await r.text())||('HTTP '+r.status));"
    "state.className='state';state.textContent='操作成功';return true}"
    "catch(e){state.className='state error';state.textContent='操作失败：'+e.message;return false}"
    "finally{actionBusy=false;button.textContent=original}}"
    "restart.onclick=async()=>{if(restarting||actionBusy)return;restarting=true;armExperience();count.textContent='0';"
    "const ok=await postCommand('/recognition/restart',restart,'正在重启…');"
    "if(ok){results.innerHTML='<li>等待重新识别</li>';recognition.className='recognition searching';"
    "recognitionTitle.textContent='等待食材识别';recognitionDetail.textContent='请在前置摄像头前稳定展示食材'}"
    "else cancelExperience();restarting=false};"
    "startWeighing.onclick=async()=>{if(await postCommand('/weighing/start',startWeighing,'正在进入称重…')){"
    "recognition.className='recognition weighing';recognitionTitle.textContent='类别已锁定，正在称重';"
    "recognitionDetail.textContent='请把已识别食材放到秤面'}};"
    "completeWeighing.onclick=async()=>{if(await postCommand('/weighing/complete',completeWeighing,'正在完成…')){"
    "armExperience();count.textContent='0';results.innerHTML='<li>称重完成，等待下一种食材</li>';"
    "recognition.className='recognition searching';recognitionTitle.textContent='等待食材识别';"
    "recognitionDetail.textContent='锁定结果已清除，请展示下一种食材'}};"
    "cancelRecognition.onclick=async()=>{if(await postCommand('/recognition/cancel',cancelRecognition,'正在取消…')){"
    "armExperience();count.textContent='0';results.innerHTML='<li>本次已取消</li>';"
    "recognition.className='recognition searching';recognitionTitle.textContent='等待食材识别';"
    "recognitionDetail.textContent='累计证据和锁定类别已清除'}};"
    "window.addEventListener('storage',event=>{if(event.key===recordsKey){records=loadRecords();renderRecords()}});"
    "clearRecords.onclick=()=>{if(!records.length||!confirm('确定清空所有识别记录吗？'))return;"
    "records=[];saveRecords();renderRecords()};"
    "function xmlEscape(value){return String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;')"
    ".replace(/>/g,'&gt;').replace(/\"/g,'&quot;').replace(/'/g,'&apos;')}"
    "function le16(n){return[n&255,n>>>8&255]}function le32(n){return[n&255,n>>>8&255,n>>>16&255,n>>>24&255]}"
    "const crcTable=(()=>{const table=[];for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)"
    "c=(c&1)?0xedb88320^(c>>>1):c>>>1;table[n]=c>>>0}return table})();"
    "function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=crcTable[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0}"
    "function concatBytes(parts){const length=parts.reduce((sum,p)=>sum+p.length,0),out=new Uint8Array(length);"
    "let offset=0;parts.forEach(p=>{out.set(p,offset);offset+=p.length});return out}"
    "function makeZip(files){const encoder=new TextEncoder(),locals=[],centrals=[];let offset=0;const now=new Date();"
    "const dosTime=(now.getHours()<<11)|(now.getMinutes()<<5)|(now.getSeconds()>>1);"
    "const dosDate=((Math.max(1980,now.getFullYear())-1980)<<9)|((now.getMonth()+1)<<5)|now.getDate();"
    "files.forEach(file=>{const name=encoder.encode(file.name),data=encoder.encode(file.data),crc=crc32(data);"
    "const local=new Uint8Array([...le32(0x04034b50),...le16(20),...le16(0x0800),...le16(0),"
    "...le16(dosTime),...le16(dosDate),...le32(crc),...le32(data.length),...le32(data.length),"
    "...le16(name.length),...le16(0),...name]);locals.push(local,data);"
    "centrals.push(new Uint8Array([...le32(0x02014b50),...le16(20),...le16(20),...le16(0x0800),"
    "...le16(0),...le16(dosTime),...le16(dosDate),...le32(crc),...le32(data.length),...le32(data.length),"
    "...le16(name.length),...le16(0),...le16(0),...le16(0),...le16(0),...le32(0),...le32(offset),...name]));"
    "offset+=local.length+data.length});const central=concatBytes(centrals);"
    "const end=new Uint8Array([...le32(0x06054b50),...le16(0),...le16(0),...le16(files.length),"
    "...le16(files.length),...le32(central.length),...le32(offset),...le16(0)]);return concatBytes([...locals,central,end])}"
    "function buildXlsx(){const rows=[[\"识别序列号\",\"识别种类\",\"可信度\",\"有效帧数\",\"识别响应耗时\",\"完整操作耗时\",\"真实食材\"]];"
    "records.forEach(item=>rows.push([item.serial,item.category,item.confidence||'',item.frames||'',item.elapsedMs,item.flowElapsedMs||item.elapsedMs,item.actual||'']));"
    "const letters=['A','B','C','D','E','F','G'];let sheetRows='';rows.forEach((row,rowIndex)=>{let cells='';"
    "row.forEach((value,columnIndex)=>{const ref=letters[columnIndex]+(rowIndex+1);"
    "cells+=rowIndex>0&&(columnIndex===4||columnIndex===5)?`<c r='${ref}' s='1'><v>${Number(value)||0}</v></c>`:"
    "`<c r='${ref}' t='inlineStr'><is><t>${xmlEscape(value)}</t></is></c>`});"
    "sheetRows+=`<row r='${rowIndex+1}'>${cells}</row>`});"
    "const files=[{name:'[Content_Types].xml',data:`<?xml version='1.0' encoding='UTF-8'?>"
    "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/><Default Extension='xml' ContentType='application/xml'/><Override PartName='/xl/workbook.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'/><Override PartName='/xl/worksheets/sheet1.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/><Override PartName='/xl/styles.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml'/></Types>`},"
    "{name:'_rels/.rels',data:`<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='xl/workbook.xml'/></Relationships>`},"
    "{name:'xl/workbook.xml',data:`<?xml version='1.0' encoding='UTF-8'?><workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><sheets><sheet name='识别记录' sheetId='1' r:id='rId1'/></sheets></workbook>`},"
    "{name:'xl/_rels/workbook.xml.rels',data:`<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet1.xml'/><Relationship Id='rId2' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles' Target='styles.xml'/></Relationships>`},"
    "{name:'xl/styles.xml',data:`<?xml version='1.0' encoding='UTF-8'?><styleSheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><numFmts count='1'><numFmt numFmtId='164' formatCode='0 &quot;ms&quot;'/></numFmts><fonts count='1'><font><sz val='11'/><name val='Calibri'/></font></fonts><fills count='2'><fill><patternFill patternType='none'/></fill><fill><patternFill patternType='gray125'/></fill></fills><borders count='1'><border/></borders><cellStyleXfs count='1'><xf numFmtId='0' fontId='0' fillId='0' borderId='0'/></cellStyleXfs><cellXfs count='2'><xf numFmtId='0' fontId='0' fillId='0' borderId='0' xfId='0'/><xf numFmtId='164' fontId='0' fillId='0' borderId='0' xfId='0' applyNumberFormat='1'/></cellXfs></styleSheet>`},"
    "{name:'xl/worksheets/sheet1.xml',data:`<?xml version='1.0' encoding='UTF-8'?><worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><cols><col min='1' max='1' width='14' customWidth='1'/><col min='2' max='2' width='18' customWidth='1'/><col min='3' max='4' width='12' customWidth='1'/><col min='5' max='6' width='20' customWidth='1'/><col min='7' max='7' width='18' customWidth='1'/></cols><sheetData>${sheetRows}</sheetData></worksheet>`}];return makeZip(files)}"
    "downloadExcel.onclick=()=>{if(!records.length)return;const blob=new Blob([buildXlsx()],"
    "{type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}),url=URL.createObjectURL(blob);"
    "const link=document.createElement('a'),now=new Date(),pad=n=>String(n).padStart(2,'0');"
    "link.href=url;link.download=`识别记录_${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_"
    "${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.xlsx`;link.click();"
    "setTimeout(()=>URL.revokeObjectURL(url),1000)};"
    "</script></body></html>";

static esp_err_t initialize_nvs(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    return err;
}

static void wifi_event_handler(void *arg,
                               esp_event_base_t event_base,
                               int32_t event_id,
                               void *event_data)
{
    (void)arg;
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        portENTER_CRITICAL(&s_state_lock);
        strcpy(s_station_ip, "not-connected");
        portEXIT_CRITICAL(&s_state_lock);
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *event = event_data;
        portENTER_CRITICAL(&s_state_lock);
        snprintf(s_station_ip, sizeof(s_station_ip), IPSTR, IP2STR(&event->ip_info.ip));
        portEXIT_CRITICAL(&s_state_lock);
        ESP_LOGI(TAG, "GUI: http://" IPSTR "/", IP2STR(&event->ip_info.ip));
    }
}

static esp_err_t start_wifi(void)
{
    ESP_RETURN_ON_ERROR(initialize_nvs(), TAG, "NVS");
    esp_err_t err = esp_netif_init();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        return err;
    }
    err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        return err;
    }

    esp_netif_t *station = esp_netif_create_default_wifi_sta();
    if (station == NULL || esp_netif_create_default_wifi_ap() == NULL) {
        return ESP_FAIL;
    }
    ESP_ERROR_CHECK(esp_netif_set_hostname(station, "ingredient-cam"));

    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_RETURN_ON_ERROR(esp_wifi_init(&init), TAG, "Wi-Fi init");
    ESP_RETURN_ON_ERROR(
        esp_event_handler_instance_register(
            WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL, NULL),
        TAG,
        "Wi-Fi event");
    ESP_RETURN_ON_ERROR(
        esp_event_handler_instance_register(
            IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL, NULL),
        TAG,
        "IP event");

    wifi_config_t ap = {
        .ap = {
            .ssid = CONFIG_WELLAND_CAM_AP_SSID,
            .ssid_len = sizeof(CONFIG_WELLAND_CAM_AP_SSID) - 1,
            .channel = 6,
            .password = CONFIG_WELLAND_CAM_AP_PASSWORD,
            .max_connection = 2,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {.required = false},
        },
    };
    wifi_config_t sta = {
        .sta = {
            .ssid = CONFIG_WELLAND_CAM_STA_SSID,
            .password = CONFIG_WELLAND_CAM_STA_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_APSTA), TAG, "Wi-Fi mode");
    ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_AP, &ap), TAG, "AP config");
    ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_STA, &sta), TAG, "STA config");
    ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "Wi-Fi start");
    return esp_wifi_set_ps(WIFI_PS_NONE);
}

static size_t append_jpeg_data(void *arg, size_t index, const void *data, size_t length)
{
    jpeg_buffer_t *buffer = arg;
    if (buffer->failed || index != buffer->length || length > SIZE_MAX - buffer->length) {
        buffer->failed = true;
        return 0;
    }
    const size_t required = buffer->length + length;
    if (required > buffer->capacity) {
        if (buffer->fixed_capacity) {
            buffer->failed = true;
            return 0;
        }
        size_t capacity = buffer->capacity == 0 ? 8192 : buffer->capacity;
        while (capacity < required) {
            capacity *= 2;
        }
        uint8_t *data_new = realloc(buffer->data, capacity);
        if (data_new == NULL) {
            buffer->failed = true;
            return 0;
        }
        buffer->data = data_new;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->length, data, length);
    buffer->length = required;
    return length;
}

static bool valid_camera_frame(const camera_fb_t *frame)
{
    const size_t expected = SENSOR_FRAME_WIDTH * SENSOR_FRAME_HEIGHT * 2U;
    if (frame->format != PIXFORMAT_RGB565 ||
        frame->width != SENSOR_FRAME_WIDTH ||
        frame->height != SENSOR_FRAME_HEIGHT ||
        frame->len != expected) {
        ESP_LOGW(TAG, "invalid frame: %ux%u format=%d bytes=%zu",
                 frame->width, frame->height, frame->format, frame->len);
        return false;
    }
    return true;
}

static camera_fb_t *capture_camera_frame(void)
{
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame != NULL && !valid_camera_frame(frame)) {
        esp_camera_fb_return(frame);
        frame = NULL;
    }
    return frame;
}

static void capture_task(void *argument)
{
    (void)argument;
    uint8_t *pixels = s_output_frame;
    while (true) {
        portENTER_CRITICAL(&s_state_lock);
        const bool active = s_pipeline_active && s_camera_ready;
        portEXIT_CRITICAL(&s_state_lock);
        if (!active) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
        camera_fb_t *camera = capture_camera_frame();
        if (camera == NULL) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
        ingredient_frame_t candidate;
        // Driver timestamp is the actual captured frame's timestamp.
        const uint32_t captured_ms = (uint32_t)(camera->timestamp.tv_sec*1000ULL +
                                                   camera->timestamp.tv_usec/1000);
        ingredient_detection_prepare(camera->buf, camera->width, camera->height,
                                     pixels, captured_ms, &candidate);
        esp_camera_fb_return(camera);
        xSemaphoreTake(s_candidate_mutex, portMAX_DELAY);
        if (!s_candidate_ready || candidate.epoch != s_candidate.epoch ||
            recognition_candidate_replace(candidate.captured_ms, s_candidate.captured_ms,
                candidate.quality.weight_permille, s_candidate.quality.weight_permille,
                candidate.quality.accepted, s_candidate.quality.accepted)) {
            memcpy(s_output_frame+OUTPUT_FRAME_BYTES, pixels, OUTPUT_FRAME_BYTES);
            s_candidate = candidate;
            s_candidate_ready = true;
        }
        xSemaphoreGive(s_candidate_mutex);
        vTaskDelay(1);
    }
}

static void ai_task(void *argument)
{
    (void)argument;
    ai_job_t job;
    uint8_t *pixels = s_output_frame+2*OUTPUT_FRAME_BYTES;
    while (true) {
        if (xQueueReceive(s_ai_queue, &job, 0) == pdTRUE) {
            *job.status = ingredient_detection_validate_rgb565_be(
                job.validation_input, INGREDIENT_MODEL_WIDTH, INGREDIENT_MODEL_HEIGHT,
                job.result, job.performance);
            xSemaphoreGive(job.done);
            continue;
        }
        ingredient_frame_t candidate;
        bool ready = false;
        xSemaphoreTake(s_candidate_mutex, portMAX_DELAY);
        if (s_candidate_ready) {
            candidate = s_candidate;
            memcpy(pixels, s_output_frame+OUTPUT_FRAME_BYTES, OUTPUT_FRAME_BYTES);
            s_candidate_ready = false;
            ready = true;
        }
        xSemaphoreGive(s_candidate_mutex);
        if (!ready) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }
        const ingredient_detection_result_t result = ingredient_detection_run(pixels, &candidate);
        portENTER_CRITICAL(&s_state_lock);
        s_detection = result;
        portEXIT_CRITICAL(&s_state_lock);
        // Copy only. JPEG and socket writes never hold a producer/AI lock.
        xSemaphoreTake(s_preview_mutex, portMAX_DELAY);
        memcpy(s_output_frame+3*OUTPUT_FRAME_BYTES, pixels, OUTPUT_FRAME_BYTES);
        ++s_preview_sequence;
        xSemaphoreGive(s_preview_mutex);
    }
}

static esp_err_t submit_ai_job(const ai_job_t *job)
{
    if (s_ai_queue == NULL || job == NULL || job->done == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    return xQueueSend(s_ai_queue, job, portMAX_DELAY) == pdTRUE ? ESP_OK : ESP_FAIL;
}

static bool encode_processed_frame(jpeg_buffer_t *jpeg, uint8_t *pixels)
{
    jpeg->length = 0;
    jpeg->failed = false;
    camera_fb_t output = {
        .buf = pixels,
        .len = OUTPUT_FRAME_BYTES,
        .width = OUTPUT_FRAME_SIZE,
        .height = OUTPUT_FRAME_SIZE,
        .format = PIXFORMAT_RGB565,
    };
    const bool converted =
        frame2jpg_cb(&output, CONFIG_WELLAND_CAM_JPEG_QUALITY, append_jpeg_data, jpeg);
    return converted && !jpeg->failed && jpeg->length > 0;
}

static bool capture_raw_jpeg(jpeg_buffer_t *jpeg)
{
    jpeg->length = 0;
    jpeg->failed = false;
    camera_fb_t *frame = capture_camera_frame();
    if (frame == NULL) {
        return false;
    }
    const bool converted =
        frame2jpg_cb(frame, CONFIG_WELLAND_CAM_JPEG_QUALITY, append_jpeg_data, jpeg);
    esp_camera_fb_return(frame);
    return converted && !jpeg->failed && jpeg->length > 0;
}

static esp_err_t index_handler(httpd_req_t *request)
{
    httpd_resp_set_type(request, "text/html; charset=utf-8");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_send(request, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t status_handler(httpd_req_t *request)
{
    ingredient_detection_result_t detection;
    ingredient_recognition_status_t recognition;
    const bool detection_boxes_enabled =
        ingredient_detection_box_overlay_enabled();
    bool camera_ready;
    uint16_t sensor_pid;
    const char *camera_state;
    char station_ip[sizeof(s_station_ip)];
    portENTER_CRITICAL(&s_state_lock);
    detection = s_detection;
    camera_ready = s_camera_ready;
    sensor_pid = s_sensor_pid;
    camera_state = s_camera_state;
    memcpy(station_ip, s_station_ip, sizeof(station_ip));
    portEXIT_CRITICAL(&s_state_lock);

    if (ingredient_detection_get_recognition_status(&recognition) != ESP_OK) {
        recognition = (ingredient_recognition_status_t){
            .state = INGREDIENT_RECOGNITION_SEARCHING,
            .inference_active = false,
            .candidate_category = UINT8_MAX,
            .candidate_second_category = UINT8_MAX,
            .collected_frames = 0,
            .candidate_votes = 0,
            .window_frames = CONFIG_WELLAND_RECOGNITION_MAX_VALID_INFERENCES,
            .mean_score = 0.0f,
            .elapsed_ms = 0,
        };
    }
    const char *recognition_state = "searching";
    if (recognition.state == INGREDIENT_RECOGNITION_CONFIRMING) {
        recognition_state = "confirming";
    } else if (recognition.state == INGREDIENT_RECOGNITION_LOCKED) {
        recognition_state = "locked";
    } else if (recognition.state == INGREDIENT_RECOGNITION_WEIGHING) {
        recognition_state = "weighing";
    } else if (recognition.state == INGREDIENT_RECOGNITION_UNCERTAIN) {
        recognition_state = "uncertain";
    }
    const char *confidence_level = "none";
    if (recognition.confidence_level == 1) {
        confidence_level = "high";
    } else if (recognition.confidence_level == 2) {
        confidence_level = "medium";
    } else if (recognition.confidence_level == 3) {
        confidence_level = "low";
    }
    const char *lock_reason = "none";
    if (recognition.lock_reason == 1) {
        lock_reason = "single_high";
    } else if (recognition.lock_reason == 2) {
        lock_reason = "two_frame_evidence";
    } else if (recognition.lock_reason == 3) {
        lock_reason = "three_frame_evidence";
    }

    char chunk[1792];
    int length = snprintf(
        chunk,
        sizeof(chunk),
        "{\"camera_ready\":%s,\"state\":\"%s\",\"sensor\":\"GC2145\","
        "\"pid\":\"0x%04x\",\"resolution\":\"224x224\",\"station_ip\":\"%s\","
        "\"inference_ms\":%" PRIu32 ",\"sequence\":%" PRIu32 ","
        "\"recognition_state\":\"%s\",\"inference_active\":%s,"
        "\"candidate_category\":%d,\"candidate_second_category\":%d,"
        "\"collected_frames\":%u,"
        "\"candidate_votes\":%u,\"window_frames\":%u,"
        "\"candidate_score\":%.3f,\"weighted_evidence\":%.3f,"
        "\"evidence_margin\":%.3f,\"confidence_level\":\"%s\","
        "\"lock_reason\":\"%s\","
        "\"effective_inferences\":%u,\"completed_locks\":%" PRIu32 ","
        "\"lock_counts\":{\"high\":%" PRIu32 ",\"medium\":%" PRIu32
        ",\"low\":%" PRIu32 "},"
        "\"output_ratios\":{\"one_frame\":%.3f,\"two_frame\":%.3f,"
        "\"three_frame\":%.3f},"
        "\"average_lock_inferences\":%.3f,\"frame_fresh\":%s,\"frame_quality_ok\":%s,"
        "\"frame_stable\":%s,\"frame_roi_changed\":%s,"
        "\"quality\":{\"weight_permille\":%u,\"motion_permille\":%u,"
        "\"sharpness\":%u,\"mean_luma\":%u,\"dark_permille\":%u,"
        "\"bright_permille\":%u,\"changed_permille\":%u,\"scene_change\":%s},"
        "\"timing_us\":{\"quality\":%" PRIu32 ",\"preprocess\":%" PRIu32
        ",\"inference\":%" PRIu32 ",\"postprocess\":%" PRIu32
        ",\"decision\":%" PRIu32 ",\"total\":%" PRIu32 "},"
        "\"recognition_elapsed_ms\":%" PRIu32 ","
        "\"motion_candidate_active\":%s,\"experience_started\":%s,"
        "\"experience_locked\":%s,\"experience_elapsed_ms\":%" PRIu32
        ",\"detections\":[",
        camera_ready ? "true" : "false",
        camera_state,
        sensor_pid,
        station_ip,
        detection.elapsed_ms,
        detection.sequence,
        recognition_state,
        recognition.inference_active ? "true" : "false",
        recognition.candidate_category < INGREDIENT_CLASS_COUNT ? recognition.candidate_category : -1,
        recognition.candidate_second_category < INGREDIENT_CLASS_COUNT ?
            recognition.candidate_second_category : -1,
        (unsigned)recognition.collected_frames,
        (unsigned)recognition.candidate_votes,
        (unsigned)recognition.window_frames,
        recognition.mean_score,
        recognition.weighted_evidence,
        recognition.evidence_margin,
        confidence_level,
        lock_reason,
        (unsigned)recognition.effective_inferences,
        recognition.completed_locks,
        recognition.high_confidence_locks,
        recognition.medium_confidence_locks,
        recognition.low_confidence_locks,
        recognition.completed_locks == 0 ? 0.0f :
            recognition.frame_lock_counts[0] / (float)recognition.completed_locks,
        recognition.completed_locks == 0 ? 0.0f :
            recognition.frame_lock_counts[1] / (float)recognition.completed_locks,
        recognition.completed_locks == 0 ? 0.0f :
            recognition.frame_lock_counts[2] / (float)recognition.completed_locks,
        recognition.average_lock_inferences,
        recognition.frame_fresh ? "true" : "false",
        recognition.frame_quality_ok ? "true" : "false",
        recognition.frame_stable ? "true" : "false",
        recognition.frame_roi_changed ? "true" : "false",
        recognition.quality_weight_permille,
        recognition.motion_permille,
        recognition.sharpness,
        recognition.mean_luma,
        recognition.dark_permille,
        recognition.bright_permille,
        recognition.changed_permille,
        recognition.frame_scene_change ? "true" : "false",
        recognition.quality_us,
        recognition.preprocess_us,
        recognition.inference_us,
        recognition.postprocess_us,
        recognition.decision_us,
        recognition.total_us,
        recognition.elapsed_ms,
        recognition.motion_candidate_active ? "true" : "false",
        recognition.experience_started ? "true" : "false",
        recognition.experience_locked ? "true" : "false",
        recognition.experience_elapsed_ms);

    httpd_resp_set_type(request, "application/json");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    ESP_RETURN_ON_ERROR(httpd_resp_send_chunk(request, chunk, length), TAG, "status head");
    for (size_t i = 0; i < detection.count; ++i) {
        const ingredient_detection_item_t *item = &detection.items[i];
        length = snprintf(chunk,
                          sizeof(chunk),
                          "%s{\"category\":%u,\"score\":%.3f}",
                          i == 0 ? "" : ",",
                          item->category,
                          item->score);
        ESP_RETURN_ON_ERROR(httpd_resp_send_chunk(request, chunk, length), TAG, "status item");
    }
    length = snprintf(chunk,
                      sizeof(chunk),
                      "],\"detection_boxes_enabled\":%s,"
                      "\"free_internal\":%zu,\"free_psram\":%zu}",
                      detection_boxes_enabled ? "true" : "false",
                      heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                      heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_RETURN_ON_ERROR(httpd_resp_send_chunk(request, chunk, length), TAG, "status tail");
    return httpd_resp_send_chunk(request, NULL, 0);
}

static esp_err_t recognition_restart_handler(httpd_req_t *request)
{
    ESP_RETURN_ON_ERROR(ingredient_detection_restart(), TAG, "restart recognition");
    portENTER_CRITICAL(&s_state_lock);
    s_detection = (ingredient_detection_result_t){0};
    portEXIT_CRITICAL(&s_state_lock);
    httpd_resp_set_type(request, "application/json");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_sendstr(request, "{\"ok\":true,\"state\":\"searching\"}");
}

static esp_err_t recognition_cancel_handler(httpd_req_t *request)
{
    ESP_RETURN_ON_ERROR(ingredient_detection_cancel(), TAG, "cancel recognition");
    portENTER_CRITICAL(&s_state_lock);
    s_detection = (ingredient_detection_result_t){0};
    portEXIT_CRITICAL(&s_state_lock);
    httpd_resp_set_type(request, "application/json");
    return httpd_resp_sendstr(request, "{\"ok\":true,\"state\":\"searching\"}");
}

static esp_err_t weighing_start_handler(httpd_req_t *request)
{
    const esp_err_t status = ingredient_detection_start_weighing();
    if (status != ESP_OK) {
        httpd_resp_set_status(request, "409 Conflict");
        return httpd_resp_sendstr(request, "category must be locked before weighing starts");
    }
    httpd_resp_set_type(request, "application/json");
    return httpd_resp_sendstr(request, "{\"ok\":true,\"state\":\"weighing\"}");
}

static esp_err_t weighing_complete_handler(httpd_req_t *request)
{
    const esp_err_t status = ingredient_detection_complete_weighing();
    if (status != ESP_OK) {
        httpd_resp_set_status(request, "409 Conflict");
        return httpd_resp_sendstr(request, "no locked weighing session");
    }
    portENTER_CRITICAL(&s_state_lock);
    s_detection = (ingredient_detection_result_t){0};
    portEXIT_CRITICAL(&s_state_lock);
    httpd_resp_set_type(request, "application/json");
    return httpd_resp_sendstr(request, "{\"ok\":true,\"state\":\"searching\"}");
}

static esp_err_t box_overlay_handler(httpd_req_t *request)
{
    char query[32] = {0};
    char enabled_value[8] = {0};
    if (httpd_req_get_url_query_str(request, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query,
                              "enabled",
                              enabled_value,
                              sizeof(enabled_value)) != ESP_OK ||
        (strcmp(enabled_value, "0") != 0 && strcmp(enabled_value, "1") != 0)) {
        httpd_resp_set_status(request, "400 Bad Request");
        return httpd_resp_sendstr(request, "enabled must be 0 or 1");
    }

    const bool enabled = strcmp(enabled_value, "1") == 0;
    ingredient_detection_set_box_overlay_enabled(enabled);
    httpd_resp_set_type(request, "application/json");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_sendstr(request,
                             enabled ? "{\"enabled\":true}" :
                                       "{\"enabled\":false}");
}

static esp_err_t capture_handler(httpd_req_t *request)
{
    portENTER_CRITICAL(&s_state_lock);
    const bool camera_ready = s_camera_ready;
    portEXIT_CRITICAL(&s_state_lock);
    if (!camera_ready) {
        httpd_resp_set_status(request, "503 Service Unavailable");
        return httpd_resp_sendstr(request, "camera not ready");
    }
    if (xSemaphoreTake(s_stream_mutex, 0) != pdTRUE) {
        httpd_resp_set_status(request, "409 Conflict");
        return httpd_resp_sendstr(request, "close the MJPEG stream before capture");
    }

    jpeg_buffer_t jpeg = {0};
    const bool captured = capture_raw_jpeg(&jpeg);
    esp_err_t err = ESP_FAIL;
    if (captured) {
        httpd_resp_set_type(request, "image/jpeg");
        httpd_resp_set_hdr(request, "Cache-Control", "no-store");
        err = httpd_resp_send(request, (const char *)jpeg.data, jpeg.length);
    } else {
        httpd_resp_set_status(request, "500 Internal Server Error");
        err = httpd_resp_sendstr(request, "capture failed");
    }
    free(jpeg.data);
    xSemaphoreGive(s_stream_mutex);
    return err;
}

static esp_err_t validation_handler(httpd_req_t *request)
{
    const size_t expected_size =
        INGREDIENT_MODEL_WIDTH * INGREDIENT_MODEL_HEIGHT * sizeof(uint16_t);
    if ((size_t)request->content_len != expected_size) {
        httpd_resp_set_status(request, "400 Bad Request");
        return httpd_resp_sendstr(request, "expected 224x224 RGB565 big-endian payload");
    }
    if (s_ai_queue == NULL) {
        httpd_resp_set_status(request, "503 Service Unavailable");
        return httpd_resp_sendstr(request, "AI pipeline not ready");
    }

    uint8_t *input = heap_caps_aligned_alloc(16,
                                             expected_size,
                                             MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (input == NULL) {
        httpd_resp_set_status(request, "500 Internal Server Error");
        return httpd_resp_sendstr(request, "unable to allocate aligned validation input");
    }

    size_t offset = 0;
    while (offset < expected_size) {
        const int received = httpd_req_recv(request,
                                            (char *)input + offset,
                                            expected_size - offset);
        if (received == HTTPD_SOCK_ERR_TIMEOUT) {
            continue;
        }
        if (received <= 0) {
            free(input);
            return ESP_FAIL;
        }
        offset += (size_t)received;
    }

    ingredient_detection_result_t detection;
    ingredient_performance_t performance;
    esp_err_t inference_status = ESP_FAIL;
    SemaphoreHandle_t done = xSemaphoreCreateBinary();
    if (done == NULL) {
        free(input);
        return ESP_ERR_NO_MEM;
    }
    const ai_job_t job = {
        .kind = AI_JOB_VALIDATION,
        .done = done,
        .validation_input = input,
        .result = &detection,
        .performance = &performance,
        .status = &inference_status,
    };
    const esp_err_t submit_status = submit_ai_job(&job);
    if (submit_status == ESP_OK) {
        xSemaphoreTake(done, portMAX_DELAY);
    } else {
        inference_status = submit_status;
    }
    vSemaphoreDelete(done);
    free(input);
    if (inference_status != ESP_OK) {
        httpd_resp_set_status(request, "500 Internal Server Error");
        return httpd_resp_sendstr(request, esp_err_to_name(inference_status));
    }

    char chunk[1536];
    int length = snprintf(
        chunk,
        sizeof(chunk),
        "{\"input\":{\"width\":%u,\"height\":%u,\"format\":\"rgb565be\"},"
        "\"timing_us\":{\"preprocess\":%" PRIu32 ",\"inference\":%" PRIu32
        ",\"postprocess\":%" PRIu32 ",\"total\":%" PRIu32 "},"
        "\"model_memory_bytes\":{\"internal\":%zu,\"psram\":%zu,\"flash\":%zu},"
        "\"heap_bytes\":{"
        "\"internal_free_before\":%zu,\"internal_free_after\":%zu,"
        "\"internal_min_free\":%zu,\"internal_largest_before\":%zu,"
        "\"internal_largest_after\":%zu,"
        "\"psram_free_before\":%zu,\"psram_free_after\":%zu,"
        "\"psram_min_free\":%zu,\"psram_largest_before\":%zu,"
        "\"psram_largest_after\":%zu},\"detections\":[",
        INGREDIENT_MODEL_WIDTH,
        INGREDIENT_MODEL_HEIGHT,
        performance.preprocess_us,
        performance.inference_us,
        performance.postprocess_us,
        performance.total_us,
        performance.model_internal_bytes,
        performance.model_psram_bytes,
        performance.model_flash_bytes,
        performance.internal_free_before,
        performance.internal_free_after,
        performance.internal_min_free,
        performance.internal_largest_before,
        performance.internal_largest_after,
        performance.psram_free_before,
        performance.psram_free_after,
        performance.psram_min_free,
        performance.psram_largest_before,
        performance.psram_largest_after);
    if (length < 0 || (size_t)length >= sizeof(chunk)) {
        return ESP_ERR_INVALID_SIZE;
    }

    httpd_resp_set_type(request, "application/json");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    ESP_RETURN_ON_ERROR(httpd_resp_send_chunk(request, chunk, length), TAG, "validation head");
    for (size_t i = 0; i < detection.count; ++i) {
        const ingredient_detection_item_t *item = &detection.items[i];
        length = snprintf(chunk,
                          sizeof(chunk),
                          "%s{\"category\":%u,\"score\":%.6f,"
                          "\"box\":[%d,%d,%d,%d]}",
                          i == 0 ? "" : ",",
                          item->category,
                          item->score,
                          item->box[0],
                          item->box[1],
                          item->box[2],
                          item->box[3]);
        ESP_RETURN_ON_ERROR(httpd_resp_send_chunk(request, chunk, length), TAG, "validation item");
    }
    ESP_RETURN_ON_ERROR(httpd_resp_send_chunk(request, "]}", 2), TAG, "validation tail");
    return httpd_resp_send_chunk(request, NULL, 0);
}

static esp_err_t stream_handler(httpd_req_t *request)
{
    portENTER_CRITICAL(&s_state_lock);
    const bool camera_ready = s_camera_ready;
    portEXIT_CRITICAL(&s_state_lock);
    if (!camera_ready) {
        httpd_resp_set_status(request, "503 Service Unavailable");
        return httpd_resp_sendstr(request, "camera not ready");
    }
    if (xSemaphoreTake(s_stream_mutex, 0) != pdTRUE) {
        httpd_resp_set_status(request, "503 Service Unavailable");
        return httpd_resp_sendstr(request, "stream already in use");
    }

    httpd_resp_set_type(request, STREAM_TYPE);
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
    jpeg_buffer_t jpeg = {0};
    jpeg.data = heap_caps_malloc(OUTPUT_FRAME_BYTES,
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (jpeg.data == NULL) {
        xSemaphoreGive(s_stream_mutex);
        return ESP_ERR_NO_MEM;
    }
    jpeg.capacity = OUTPUT_FRAME_BYTES;
    jpeg.fixed_capacity = true;
    uint8_t *preview = heap_caps_malloc(OUTPUT_FRAME_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (preview == NULL) {
        free(jpeg.data);
        xSemaphoreGive(s_stream_mutex);
        return ESP_ERR_NO_MEM;
    }
    xSemaphoreTake(s_candidate_mutex, portMAX_DELAY);
    s_candidate_ready = false;
    xSemaphoreGive(s_candidate_mutex);
    portENTER_CRITICAL(&s_state_lock);
    s_pipeline_active = true;
    portEXIT_CRITICAL(&s_state_lock);
    esp_err_t err = ESP_OK;
    uint32_t last_sequence = 0;
    while (err == ESP_OK) {
        bool ready;
        xSemaphoreTake(s_preview_mutex, portMAX_DELAY);
        ready = s_preview_sequence != last_sequence;
        if (ready) {
            memcpy(preview, s_output_frame+3*OUTPUT_FRAME_BYTES, OUTPUT_FRAME_BYTES);
            last_sequence = s_preview_sequence;
        }
        xSemaphoreGive(s_preview_mutex);
        if (!ready) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        if (!encode_processed_frame(&jpeg, preview)) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        char header[64];
        const int header_length = snprintf(header, sizeof(header),
            "Content-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", jpeg.length);
        if ((err = httpd_resp_send_chunk(request, STREAM_BOUNDARY, sizeof(STREAM_BOUNDARY)-1)) != ESP_OK ||
            (err = httpd_resp_send_chunk(request, header, header_length)) != ESP_OK ||
            (err = httpd_resp_send_chunk(request, (const char *)jpeg.data, jpeg.length)) != ESP_OK) break;
        vTaskDelay(pdMS_TO_TICKS(ACTIVE_PREVIEW_INTERVAL_MS));
    }
    portENTER_CRITICAL(&s_state_lock);
    s_pipeline_active = false;
    portEXIT_CRITICAL(&s_state_lock);
    free(preview);
    free(jpeg.data);
    xSemaphoreGive(s_stream_mutex);
    return err;
}

static esp_err_t start_http_servers(void)
{
    httpd_config_t ui = HTTPD_DEFAULT_CONFIG();
    ui.core_id = 0;
    ui.stack_size = 8192;
    ui.max_open_sockets = 4;
    ui.max_uri_handlers = 12;
    ui.lru_purge_enable = true;
    ESP_RETURN_ON_ERROR(httpd_start(&s_ui_server, &ui), TAG, "UI server");

    const httpd_uri_t index = {.uri = "/", .method = HTTP_GET, .handler = index_handler};
    const httpd_uri_t status = {.uri = "/status", .method = HTTP_GET, .handler = status_handler};
    const httpd_uri_t restart = {
        .uri = "/recognition/restart",
        .method = HTTP_POST,
        .handler = recognition_restart_handler,
    };
    const httpd_uri_t cancel = {
        .uri = "/recognition/cancel",
        .method = HTTP_POST,
        .handler = recognition_cancel_handler,
    };
    const httpd_uri_t weighing_start = {
        .uri = "/weighing/start",
        .method = HTTP_POST,
        .handler = weighing_start_handler,
    };
    const httpd_uri_t weighing_complete = {
        .uri = "/weighing/complete",
        .method = HTTP_POST,
        .handler = weighing_complete_handler,
    };
    const httpd_uri_t box_overlay = {
        .uri = "/overlay/boxes",
        .method = HTTP_POST,
        .handler = box_overlay_handler,
    };
    const httpd_uri_t capture = {
        .uri = "/capture.jpg",
        .method = HTTP_GET,
        .handler = capture_handler,
    };
    const httpd_uri_t validation = {
        .uri = "/validate",
        .method = HTTP_POST,
        .handler = validation_handler,
    };
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &index), TAG, "index route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &status), TAG, "status route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &restart), TAG, "restart route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &cancel), TAG, "cancel route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &weighing_start), TAG, "weigh start route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &weighing_complete), TAG, "weigh complete route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &box_overlay), TAG, "box overlay route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &capture), TAG, "capture route");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_ui_server, &validation), TAG, "validation route");

    httpd_config_t stream = HTTPD_DEFAULT_CONFIG();
    stream.core_id = 0;
    stream.server_port = 81;
    stream.ctrl_port += 1;
    stream.stack_size = 12288;
    stream.max_open_sockets = 2;
    stream.lru_purge_enable = true;
    ESP_RETURN_ON_ERROR(httpd_start(&s_stream_server, &stream), TAG, "stream server");
    const httpd_uri_t stream_uri = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = stream_handler,
    };
    return httpd_register_uri_handler(s_stream_server, &stream_uri);
}

esp_err_t camera_web_start(void)
{
    s_stream_mutex = xSemaphoreCreateMutex();
    if (s_stream_mutex == NULL) {
        return ESP_ERR_NO_MEM;
    }
    ESP_RETURN_ON_ERROR(start_wifi(), TAG, "Wi-Fi");
    ESP_RETURN_ON_ERROR(start_http_servers(), TAG, "HTTP");
    ESP_LOGI(TAG, "connecting to %s", CONFIG_WELLAND_CAM_STA_SSID);
    ESP_LOGI(TAG, "fallback GUI: join %s and open http://192.168.4.1/",
             CONFIG_WELLAND_CAM_AP_SSID);
    return ESP_OK;
}

esp_err_t camera_web_start_pipeline(void)
{
    if (s_ai_queue != NULL) {
        return ESP_OK;
    }
    s_output_frame = heap_caps_aligned_alloc(16,
                                             4 * OUTPUT_FRAME_BYTES,
                                             MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_output_frame == NULL) {
        return ESP_ERR_NO_MEM;
    }
    s_candidate_mutex = xSemaphoreCreateMutex();
    s_preview_mutex = xSemaphoreCreateMutex();
    if (s_candidate_mutex == NULL || s_preview_mutex == NULL) {
        if (s_candidate_mutex) vSemaphoreDelete(s_candidate_mutex);
        if (s_preview_mutex) vSemaphoreDelete(s_preview_mutex);
        free(s_output_frame); s_output_frame = NULL;
        return ESP_ERR_NO_MEM;
    }
    s_ai_queue = xQueueCreate(1, sizeof(ai_job_t));
    if (s_ai_queue == NULL) {
        vSemaphoreDelete(s_candidate_mutex);
        vSemaphoreDelete(s_preview_mutex);
        free(s_output_frame);
        s_output_frame = NULL;
        return ESP_ERR_NO_MEM;
    }
    if (xTaskCreatePinnedToCore(ai_task,
                                "ingredient_ai",
                                AI_TASK_STACK_SIZE,
                                NULL,
                                5,
                                &s_ai_task,
                                AI_TASK_CORE) != pdPASS) {
        vQueueDelete(s_ai_queue);
        s_ai_queue = NULL;
        vSemaphoreDelete(s_candidate_mutex);
        vSemaphoreDelete(s_preview_mutex);
        free(s_output_frame);
        s_output_frame = NULL;
        return ESP_ERR_NO_MEM;
    }
    if (xTaskCreatePinnedToCore(capture_task, "ingredient_capture", 8192,
                               NULL, 4, &s_capture_task, 0) != pdPASS) {
        vTaskDelete(s_ai_task);
        vQueueDelete(s_ai_queue); s_ai_queue = NULL;
        vSemaphoreDelete(s_candidate_mutex);
        vSemaphoreDelete(s_preview_mutex);
        free(s_output_frame); s_output_frame = NULL;
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "pipeline ready: independent capture/preview core=0, AI core=%d, latest candidate only",
             AI_TASK_CORE);
    return ESP_OK;
}

void camera_web_set_camera_status(bool camera_ready,
                                  uint16_t sensor_pid,
                                  const char *state)
{
    portENTER_CRITICAL(&s_state_lock);
    s_camera_ready = camera_ready;
    s_sensor_pid = sensor_pid;
    s_camera_state = state != NULL ? state : "unknown";
    portEXIT_CRITICAL(&s_state_lock);
}
