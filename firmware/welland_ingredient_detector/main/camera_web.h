#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

esp_err_t camera_web_start(void);
esp_err_t camera_web_start_pipeline(void);
void camera_web_set_camera_status(bool camera_ready,
                                  uint16_t sensor_pid,
                                  const char *state);
