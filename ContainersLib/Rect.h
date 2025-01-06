#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "SystemLib/BasicTypes.h"

namespace Selas
{
    struct Rect
    {
        sint left;
        sint top;
        sint right;
        sint bottom;
    };

    struct FloatRect
    {
        float left;
        float top;
        float right;
        float bottom;
    };
}