#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "MathLib/FloatStructs.h"
#include "SystemLib/BasicTypes.h"
#include "SystemLib/JsAssert.h"

namespace Selas
{
    namespace Random
    {
        struct MersenneTwisterData;
        struct MersenneTwister
        {
            MersenneTwister() : data(nullptr) { }
            ~MersenneTwister() { Assert_(data == nullptr); }

            // this uses a pimpl implementation since it is backed by mt19937 from std
            MersenneTwisterData* data;
        };

        void MersenneTwisterInitialize(MersenneTwister* twister, uint32 seed);
        void MersenneTwisterReseed(MersenneTwister* twister, uint32 seed);
        void MersenneTwisterShutdown(MersenneTwister* twister);
        float MersenneTwisterFloat(MersenneTwister* twister);
        uint32 MersenneTwisterUint32(MersenneTwister* twister);

        uint  RandUint(uint max);
        float RandFloat0_1(void); // [0.0f, 1.0f]
    }
}