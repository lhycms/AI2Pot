#ifndef CORE_HARDWARE_H
#define CORE_HARDWARE_H

#include <unistd.h>

/**
 * @brief Get the number of processors (threads)
 * 
 */
static int getNumProcessorsOnln() {
    long nProcessorsOnln = sysconf(_SC_NPROCESSORS_ONLN);
    if (nProcessorsOnln == -1) {
        return 1;
    } else {
        return (int)nProcessorsOnln;
    }
}

#endif