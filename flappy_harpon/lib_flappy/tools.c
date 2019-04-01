#include <stdio.h>
#include <stdlib.h>

#include "tools.h"

int min(int v1, int v2) {
    if (v1 <= v2) {
        return v1;
    } else {
        return v2;
    }
}

int max(int v1, int v2) {
    if (v1 >= v2) {
        return v1;
    } else {
        return v2;
    }
}

int randCenter(void) {
    return random() % (TROU_CENTRE_YMAX - TROU_CENTRE_YMIN) + TROU_CENTRE_YMIN;
}
