#include "animation_decode.h"
#include <string.h>

static uint32_t read_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

bool animation_decode(uint8_t *pixels, size_t pixel_count,
                      const uint8_t *packet, size_t packet_size)
{
    if (!pixels || !packet || packet_size < 4 || pixel_count > SIZE_MAX / 3) return false;
    uint32_t count = read_u32(packet);
    size_t pos = 4;
    for (uint32_t i = 0; i < count; ++i) {
        if (packet_size - pos < 8) return false;
        const uint32_t offset = read_u32(packet + pos);
        const uint32_t length = read_u32(packet + pos + 4);
        pos += 8;
        if (offset > pixel_count || length > pixel_count - offset ||
            length > (packet_size - pos) / 3) return false;
        memcpy(pixels + (size_t)offset * 3, packet + pos, (size_t)length * 3);
        pos += (size_t)length * 3;
    }
    return pos == packet_size;
}
