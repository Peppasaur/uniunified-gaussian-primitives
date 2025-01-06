//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "TextureLib/TextureResource.h"
#include "TextureLib/StbImage.h"
#include "Assets/AssetFileUtils.h"
#include "StringLib/FixedString.h"
#include "StringLib/StringUtil.h"
#include "IoLib/BinaryStreamSerializer.h"
#include "IoLib/File.h"
#include "IoLib/Directory.h"
#include "SystemLib/BasicTypes.h"

#include <stdio.h>

namespace Selas
{
    cpointer TextureResource::kDataType = "Textures";
    const uint64 TextureResource::kDataVersion = 1534887410ul;

    //=============================================================================================================================
    void Serialize(CSerializer* serializer, TextureResourceData& data)
    {
        Serialize(serializer, data.mipCount);
        Serialize(serializer, data.dataSize);

        for(uint scan = 0; scan < TextureResourceData::MaxMipCount; ++scan) {
            Serialize(serializer, data.mipWidths[scan]);
        }
        for(uint scan = 0; scan < TextureResourceData::MaxMipCount; ++scan) {
            Serialize(serializer, data.mipHeights[scan]);
        }
        for(uint scan = 0; scan < TextureResourceData::MaxMipCount; ++scan) {
            Serialize(serializer, data.mipOffsets[scan]);
        }

        Serialize(serializer, (uint32&)data.format);
        Serialize(serializer, data.pad);

        serializer->SerializePtr((void*&)data.texture, data.dataSize, 0);
    }

    //=============================================================================================================================
    //Error ReadPtexTexture(cpointer filepath, TextureResource* resource)
    //{
    //    void* fileData = nullptr;
    //    uint64 fileSize = 0;
    //    ReturnError_(File::ReadWholeFile(filepath, &fileData, &fileSize));

    //    AttachToBinary(resource->data, (uint8*)fileData, fileSize);

    //    return Success_;
    //}

    //=============================================================================================================================
    Error ReadTextureResource(cpointer textureName, TextureResource* resource)
    {
        FilePathString filepath;
        AssetFileUtils::AssetFilePath(TextureResource::kDataType, TextureResource::kDataVersion, textureName, filepath);

        void* fileData = nullptr;
        uint64 fileSize = 0;
        ReturnError_(File::ReadWholeFile(filepath.Ascii(), &fileData, &fileSize));

        AttachToBinary(resource->data, (uint8*)fileData, fileSize);

        return Success_;
    }

    //=============================================================================================================================
    void ShutdownTextureResource(TextureResource* texture)
    {
        SafeFreeAligned_(texture->data);
    }

    //=============================================================================================================================
    static void DebugWriteTextureMip(TextureResource* texture, uint level, cpointer filepath)
    {
        uint channels = (uint)texture->data->format + 1;

        uint64 mipOffset = texture->data->mipOffsets[level];
        uint32 mipWidth  = texture->data->mipWidths[level];
        uint32 mipHeight = texture->data->mipHeights[level];
        void*  mip       = &texture->data->texture[mipOffset];
        StbImageWrite(filepath, mipWidth, mipHeight, channels, HDR, (void*)mip);
    }

    //=============================================================================================================================
    void DebugWriteTextureMips(TextureResource* texture, cpointer folder, cpointer name)
    {
        for(uint scan = 0, count = texture->data->mipCount; scan < count; ++scan) {
            FixedString256 path;
            #if IsWindows_
                sprintf_s(path.Ascii(), path.Capacity(), "%s/%s_mip_%llu.hdr", folder, name, scan);
            #else
                sprintf(path.Ascii(), "%s/%s_mip_%llu.hdr", folder, name, scan);
            #endif

            Directory::EnsureDirectoryExists(path.Ascii());

            DebugWriteTextureMip(texture, scan, path.Ascii());
        }
    }
}