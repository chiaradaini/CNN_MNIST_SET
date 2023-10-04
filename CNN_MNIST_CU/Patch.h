#ifndef PATCH_H
#define PATCH_H

#include <vector>
#include "Image.h"

using std::vector;

struct Patch {
    const image3D patch;
    const int x;
    const int y;
    const int z;
};

typedef vector<Patch> patches;

#endif
