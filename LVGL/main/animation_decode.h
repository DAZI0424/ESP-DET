#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Allocation-free decoder; frame 0 must be applied before any delta frame. */
bool animation_decode(uint8_t *pixels, size_t pixel_count,
                      const uint8_t *packet, size_t packet_size);
