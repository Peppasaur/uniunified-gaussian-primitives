#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "MathLib/FloatStructs.h"
#include "SystemLib/Error.h"
#include "SystemLib/BasicTypes.h"

namespace Selas
{
    class CSerializer;

    struct TextureResourceData
    {
        enum TextureDataType
        {
            // -- Convert to something more like d3d formats?
            Float,
            Float2,
            Float3,
            Float4
        };

        static const uint MaxMipCount = 16;

        uint32 mipCount;
        uint32 dataSize;

        uint32 mipWidths[MaxMipCount];
        uint32 mipHeights[MaxMipCount];
        uint64 mipOffsets[MaxMipCount];

        TextureDataType format;
        uint32 pad;

        uint8* texture;
    };
    void Serialize(CSerializer* serializer, TextureResourceData& data);

    struct TextureResource
    {
        static cpointer kDataType;
        static const uint64 kDataVersion;

        TextureResourceData* data;
    };

    Error ReadTextureResource(cpointer filepath, TextureResource* texture);
    void ShutdownTextureResource(TextureResource* texture);
    void DebugWriteTextureMips(TextureResource* texture, cpointer folder, cpointer name);
}