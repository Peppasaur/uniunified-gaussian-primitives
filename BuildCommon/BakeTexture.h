#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "SystemLib/Error.h"
#include "SystemLib/BasicTypes.h"

namespace Selas
{
    struct TextureResourceData;
    struct BuildProcessorContext;

    //=============================================================================================================================
    Error BakeTexture(BuildProcessorContext* context, TextureResourceData* data);

}