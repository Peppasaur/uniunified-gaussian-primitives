
// -- Make sure this is only included once per compilation unit.
// -- Aka: Only include this in cpp files
// -- This is done because we include a middleware library rapidjson.

#ifdef IncludeJsonUtilities_
#pragma error("Please make sure JsonUtilities.h is only included in cpp files.");
#endif

#define IncludeJsonUtilities_

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "StringLib/FixedString.h"
#include "MathLib/FloatStructs.h"
#include "SystemLib/Error.h"
#include "SystemLib/BasicTypes.h"

// -- middleware
#include "rapidjson/document.h"

namespace Selas
{
    namespace Json
    {
        Error OpenJsonDocument(cpointer filepath, rapidjson::Document& document);

        bool IsStringAttribute(const rapidjson::Value& element, cpointer key);
        bool IsFloatAttribute(const rapidjson::Value& element, cpointer key);
        bool IsFloat3Attribute(const rapidjson::Value& element, cpointer key);

        bool ReadBool(const rapidjson::Value& element, cpointer key, bool& value, bool defaultValue);
        bool ReadInt32(const rapidjson::Value& element, cpointer key, int32& value, int32 defaultValue);
        bool ReadFloat(const rapidjson::Value& element, cpointer key, float& value, float defaultValue);
        bool ReadFloat2(const rapidjson::Value& element, cpointer key, float2& value, float2 defaultValue);
        bool ReadFloat3(const rapidjson::Value& element, cpointer key, float3& value, float3 defaultValue);
        bool ReadFloat4(const rapidjson::Value& element, cpointer key, float4& value, float4 defaultValue);

        bool ReadMatrix4x4(const rapidjson::Value& element, float4x4& value);
        bool ReadMatrix4x4(const rapidjson::Value& element, cpointer key, float4x4& value);

        // default is "". Returns false when the string does not fit.
        template<int size>
        bool ReadFixedString(const rapidjson::Value& element, cpointer key, cpointer prefix, FixedString<size>& value)
        {
            value.Clear();

            if(element.HasMember(key) == false) {
                return false;
            }
            if(element[key].IsString() == false) {
                return false;
            }

            uint prefixLen = StringUtil::Length(prefix);
            if(element[key].GetStringLength() + prefixLen > value.Capacity()) {
                return false;
            }

            StringUtil::Copy(value.Ascii(), (uint32)value.Capacity(), prefix);
            StringUtil::Copy(value.Ascii() + prefixLen, (uint32)value.Capacity() - (int32)prefixLen, element[key].GetString());

            return true;
        }

        // default is "". Returns false when the string does not fit.
        template<int size>
        bool ReadFixedString(const rapidjson::Value& element, cpointer key, FixedString<size>& value)
        {
            return ReadFixedString(element, key, "", value);
        }

    }
}
