# UI 动画开发约束

## 目录和文件

1. 后续新增或拆分的动画必须放在 `LVGL/main/ui/animation/`。
2. 每个动画必须成对创建一个 `.c` 和一个 `.h` 文件，文件名使用同一动画前缀，例如：
   - `ui_xxx_animation.c`
   - `ui_xxx_animation.h`
3. 新建的 C/C++ 文件头部必须注明 `@author lichunjiang`，非沃莱科技项目不添加公司 Copyright 行。

## 接口和职责

1. 动画 `.h` 只声明该动画对外需要的回调或控制接口；动画 `.c` 实现对应逻辑。
2. 页面组件文件（例如 `ui_loading.c`、`ui_category_scene.c`）不得继续保留名为 `update` 的静态动画回调。
3. 页面组件必须通过头文件引入动画接口，例如：

   ```c
   #include "animation/ui_xxx_animation.h"
   timer = lv_timer_create(ui_xxx_animation_update, period_ms, owner);
   ```

4. 页面组件负责对象创建、timer 生命周期和页面销毁；动画模块负责时间计算、关键帧读取和对象状态更新。
5. 动画回调通过 `lv_timer_get_user_data()` 获取所属状态，禁止使用跨页面共享的可变全局状态。
6. 关键帧数据可以放在动画专属的头文件中；动画状态必须保存在页面状态结构体中。

## 生命周期和安全

1. 删除页面对象前必须先删除对应 timer，确保回调不会访问失效的 `user_data` 或 LVGL 对象。
2. 重播动画必须通过页面提供的 restart 接口重置开始时间、帧号和结束状态。
3. 动画回调只在 LVGL 所属 UI 任务中执行；其他任务不得直接修改动画对象。
4. 新动画不得绕过 LVGL 页面层直接操作 SPI、GPIO 或 LCD 驱动。

## 构建与测试

1. 新增动画 `.c` 必须加入 `LVGL/main/CMakeLists.txt` 的 `SRCS`；对应 `.h` 通过页面源文件的 `#include` 引入。
2. 每个动画至少提供一个可重复的边界检查：起始帧、结束帧、重播或 timer 删除。
3. 修改后先运行对应的 `LVGL/tests/` 检查脚本，再执行 ESP-IDF 构建。
4. 报告动画帧数时必须区分“参考关键帧采样数”和“实际应用/刷新帧数”。
