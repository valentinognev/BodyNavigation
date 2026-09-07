#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline int cadac_system(const char *) { return 0; }
#define system(arg) cadac_system(arg)

static inline char *_itoa(int value, char *buffer, int) {
    std::sprintf(buffer, "%d", value);
    return buffer;
}
