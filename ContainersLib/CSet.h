#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "SystemLib/MemoryAllocation.h"
#include "SystemLib/Memory.h"
#include "SystemLib/JsAssert.h"

namespace Selas
{
    template <typename Type_>
    class CSet
    {
    public:
        CSet(void);
        ~CSet(void);

        void Shutdown(void);
        void Clear(void);
        void Reserve(uint64 capacity);

        const Type_* DataPointer(void) const { return _data; }
        Type_* DataPointer(void) { return _data; }

        inline Type_&       operator[] (uint index) { return _data[index]; }
        inline const Type_& operator[] (uint index) const { return _data[index]; }

        inline uint64 Count(void) const { return _count; }
        inline uint64 Capacity(void) const { return _capacity; }
        inline uint64 DataSize(void) const { return _count * sizeof(Type_); }

        uint64 Add(const Type_& element);

        template<typename OtherType_>
        void   Append(const OtherType_& addend);

        bool Remove(const Type_& item);
        void RemoveFast(uint index);

    private:
        void ReallocateArray(uint64 newLength, uint64 newCapacity);
        void GrowArray(void);

    private:
        Type_ * _data;
        uint64  _count;
        uint64  _capacity;
    };

    template<typename Type_>
    CSet<Type_>::CSet(void)
        : _data(nullptr)
        , _count(0)
        , _capacity(0)
    {
    }

    template<typename Type_>
    CSet<Type_>::~CSet(void)
    {
        Shutdown();
    }

    template<typename Type_>
    void CSet<Type_>::Shutdown(void)
    {
        if(_data) {
            Free_(_data);
        }

        _data = nullptr;
        _count = 0;
        _capacity = 0;
    }

    template<typename Type_>
    void CSet<Type_>::Clear(void)
    {
        _count = 0;
    }

    template<typename Type_>
    void CSet<Type_>::Reserve(uint64 capacity)
    {
        if(capacity > _capacity) {
            ReallocateArray(_count, capacity);
        }
    }

    template<typename Type_>
    uint64 CSet<Type_>::Add(const Type_& element)
    {
        // JSTODO - Slow...
        for(uint64 scan = 0; scan < _count; ++scan) {
            if(_data[scan] == element) {
                return scan;
            }
        }

        if(_count == _capacity) {
            GrowArray();
        }

        Assert_(_count < _capacity);
        _data[_count] = element;
        return _count++;
    }

    template<typename Type_>
    template<typename OtherType_>
    void CSet<Type_>::Append(const OtherType_& addend)
    {
        uint64 newLength = _count + addend.Length();

        if(_capacity < newLength)
            ReallocateArray(_count, newLength);

        for(uint scan = 0, count = addend.Length(); scan < count; ++scan) {
            Add(addend[scan]);
        }
    }

    template<typename Type_>
    bool CSet<Type_>::Remove(const Type_& item)
    {
        uint64 index = 0;
        for(; index < _count; ++index) {
            if(_data[index] == item) {
                break;
            }
        }

        if(index == _count) {
            return false;
        }

        for(; index < _count; ++index) {
            _data[index] = _data[index + 1];
        }

        --_count;

        return true;
    }

    template<typename Type_>
    void CSet<Type_>::RemoveFast(uint index)
    {
        Assert_(index >= 0);
        Assert_(index < _count);

        _data[index] = _data[_count - 1];
        _count--;
    }

    template<typename Type_>
    void CSet<Type_>::ReallocateArray(uint64 newLength, uint64 newCapacity)
    {
        Type_* newList = AllocArray_(Type_, newCapacity);

        if(_data) {
            uint64 lengthToCopy = (_count < newLength) ? _count : newLength;
            if(lengthToCopy > 0) {
                Memory::Copy(newList, _data, lengthToCopy * sizeof(Type_));
            }

            Free_(_data);
        }

        _data = newList;
        _count = newLength;
        _capacity = newCapacity;
    }

    template<typename Type_>
    void CSet<Type_>::GrowArray(void)
    {
        // Idea from old BHG code; seems very sensible.
        if(_capacity < 64) {
            ReallocateArray(_count, _capacity + 16);
        }
        else if(_capacity < 256) {
            ReallocateArray(_count, _capacity + 32);
        }
        else {
            ReallocateArray(_count, _capacity + 128);
        }
    }
}