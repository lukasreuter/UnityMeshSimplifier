using System;
using System.Diagnostics;

namespace UnityMeshSimplifier
{
public struct FixedArray3<T> where T : unmanaged
{
    private T item1;
    private T item2;
    private T item3;

    public T this[int index]
    {
        get
        {
            CheckIndex(index);
            return index switch
            {
                0 => item1,
                1 => item2,
                2 => item3,
                _ => default,
            };
        }
        set
        {
            CheckIndex(index);
            switch (index)
            {
                case 0:
                    item1 = value;
                    break;
                case 1:
                    item2 = value;
                    break;
                case 2:
                    item3 = value;
                    break;
            }
        }
    }

    [Conditional("ENABLE_UNITY_COLLECTIONS_CHECKS")]
    private static void CheckIndex(int index)
    {
        if (index < 0 || index > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "index is not inside of the allowed range");
        }
    }
}
}
