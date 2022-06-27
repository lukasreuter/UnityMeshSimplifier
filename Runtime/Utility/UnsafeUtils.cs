using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Assertions;

namespace UnityMeshSimplifier
{
public static class UnsafeUtils
{
    [NotBurstCompatible]
    // ReSharper disable once InconsistentNaming
    public static T[] ToArrayNBC<T>(this UnsafeList<T> list) where T : unmanaged
    {
        if (list.IsEmpty)
        {
            return Array.Empty<T>();
        }

        var array = new T[list.Length];
        var length = list.Length;

        for (var i = 0; i < length; ++i)
        {
            array[i] = list[i];
        }

        return array;
    }

    [NotBurstCompatible]
    // ReSharper disable once InconsistentNaming
    public static void CopyFromNBC<T>(this UnsafeList<T> list, T[] array) where T : unmanaged
    {
        // GCHandle gcHandle = GCHandle.Alloc((object) array, GCHandleType.Pinned);
        // UnsafeUtility.MemCpy((void*) ((IntPtr) (void*) gcHandle.AddrOfPinnedObject() + dstIndex * UnsafeUtility.SizeOf<T>()), (void*) ((IntPtr) src.m_Buffer + srcIndex * UnsafeUtility.SizeOf<T>()), (long) (length * UnsafeUtility.SizeOf<T>()));
        // gcHandle.Free();

        if (list.Length != array.Length)
        {
            throw new ArgumentException("lengths are not equal");
        }

        for (var i = 0; i < list.Length; ++i)
        {
            list[i] = array[i];
        }
    }

    [NotBurstCompatible]
    // ReSharper disable once InconsistentNaming
    public static void CopyFromNBC<T>(this UnsafeList<T> list, IList<T> array) where T : unmanaged
    {
        // GCHandle gcHandle = GCHandle.Alloc((object) array, GCHandleType.Pinned);
        // UnsafeUtility.MemCpy((void*) ((IntPtr) (void*) gcHandle.AddrOfPinnedObject() + dstIndex * UnsafeUtility.SizeOf<T>()), (void*) ((IntPtr) src.m_Buffer + srcIndex * UnsafeUtility.SizeOf<T>()), (long) (length * UnsafeUtility.SizeOf<T>()));
        // gcHandle.Free();

        Check(list.Length, array.Count);

        for (var i = 0; i < list.Length; ++i)
        {
            list[i] = array[i];
        }

        [Conditional("ENABLE_UNITY_COLLECTIONS_CHECKS")]
        static void Check(int listLength, int arrayLength)
        {
            if (listLength != arrayLength)
            {
                throw new ArgumentException("lengths are not equal");
            }
        }
    }
}
}
