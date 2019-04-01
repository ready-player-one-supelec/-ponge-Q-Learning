#include <stdio.h>
#include <stdlib.h>

#include "lib_flappy/flappy.h"

int main (int argc, char *argv[]) {

    init_flappy(1);
    run_flappy();
    exit_flappy();
    return EXIT_SUCCESS;
}
