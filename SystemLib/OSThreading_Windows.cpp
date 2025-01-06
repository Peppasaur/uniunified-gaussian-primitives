//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#if IsWindows_

#include "SystemLib/OSThreading.h"
#include "SystemLib/JsAssert.h"
#include "SystemLib/MemoryAllocation.h"
#include "SystemLib/Memory.h"

#include <windows.h>

namespace Selas
{
    //=============================================================================================================================
    // Events
    //=============================================================================================================================

    //=============================================================================================================================
    // void* CreateEvent(bool manualReset, bool initialState)
    // {
    //     return ::CreateEvent(nullptr, manualReset, initialState, nullptr);
    // }

    // //==========================================================================================================================
    // bool CloseEvent(void* handle)
    // {
    //     return CloseHandle(reinterpret_cast<HANDLE>(handle)) ? true : false;
    // }

    // //==========================================================================================================================
    // bool SetEvent(void* handle)
    // {
    //     return ::SetEvent(reinterpret_cast<HANDLE>(handle)) ? true : false;
    // }

    // //==========================================================================================================================
    // bool ResetEvent(void* handle)
    // {
    //     return ::ResetEvent(reinterpret_cast<HANDLE>(handle)) ? true : false;
    // }

    // //==========================================================================================================================
    // bool WaitForSingleObject(void* handle, uint32 milliseconds)
    // {
    //     uint32 retval = ::WaitForSingleObject(reinterpret_cast<HANDLE>(handle), milliseconds);
    //     return (retval == WAIT_OBJECT_0);
    // }

    // //==========================================================================================================================
    // bool WaitForAllObjects(uint32 numHandles, void** handles, uint32 milliseconds)
    // {
    //     uint32 retval = ::WaitForMultipleObjects(numHandles, static_cast<const HANDLE*>(handles), /*wait_for_all*/true,
    //                                              milliseconds);
    //     return (retval == 0);
    // }

    //=============================================================================================================================
    // Semaphore
    //=============================================================================================================================

    //=============================================================================================================================
    void* CreateOSSemaphore(uint32 initialCount, uint32 maxCount)
    {
        return ::CreateSemaphore(nullptr, initialCount, maxCount, nullptr);
    }

    //=============================================================================================================================
    void CloseOSSemaphore(void* semaphore)
    {
        ::CloseHandle(semaphore);
    }

    //=============================================================================================================================
    void PostSemaphore(void* semaphore, uint32 count)
    {
        ::ReleaseSemaphore(semaphore, count, nullptr);
    }

    //=============================================================================================================================
    bool WaitForSemaphore(void* semaphore, uint32 milliseconds)
    {
        return (::WaitForSingleObject(semaphore, milliseconds) == 0);
    }

    //=============================================================================================================================
    // Spinlocks
    //=============================================================================================================================
    void* CreateSpinLock(void)
    {
        uint8* spin = AllocArrayAligned_(uint8, CacheLineSize_, CacheLineSize_);
        Memory::Zero(spin, CacheLineSize_);
        return spin;
    }

    //=============================================================================================================================
    void CreateSpinLock(uint8 spin[CacheLineSize_])
    {
        Memory::Zero(spin, CacheLineSize_);
    }

    //=============================================================================================================================
    void CloseSpinlock(void* spinlock)
    {
        FreeAligned_(spinlock);
    }

    //=============================================================================================================================
    bool TryEnterSpinLock(void* spinlock)
    {
        volatile LONG64* atom = reinterpret_cast<volatile LONG64*>(spinlock);
        return InterlockedCompareExchange64(atom, 1, 0) == 0;
    }

    //=============================================================================================================================
    void EnterSpinLock(void* spinlock)
    {
        volatile LONG64* atom = reinterpret_cast<volatile LONG64*>(spinlock);
        while(InterlockedCompareExchange64(atom, 1, 0) != 0) {};
    }

    //=============================================================================================================================
    void LeaveSpinLock(void* spinlock)
    {
        volatile LONG64* atom = reinterpret_cast<volatile LONG64*>(spinlock);
        *atom = 0;
    }

    //=============================================================================================================================
    void Sleep(uint sleepTimeMs)
    {
        ::Sleep((DWORD)sleepTimeMs);
    }
}

#endif
