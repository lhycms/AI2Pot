/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

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