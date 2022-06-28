#region License
/*
MIT License

Copyright(c) 2017-2020 Mattias Edlund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#endregion

#if UNITY_2018_2_OR_NEWER
#define UNITY_8UV_SUPPORT
#endif

using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Mathematics;

namespace UnityMeshSimplifier.Internal
{
    internal struct UVChannels<TVec> : IDisposable where TVec : unmanaged
    {
        private NativeList<TVec> channel0;
        private NativeList<TVec> channel1;
        private NativeList<TVec> channel2;
        private NativeList<TVec> channel3;
#if UNITY_8UV_SUPPORT
        private NativeList<TVec> channel4;
        private NativeList<TVec> channel5;
        private NativeList<TVec> channel6;
        private NativeList<TVec> channel7;
#endif

        private bool4 first;
#if UNITY_8UV_SUPPORT
        private bool4 second;
#endif

        public bool IsUsed => math.any(first)
#if UNITY_8UV_SUPPORT
                              || math.any(second);
#endif

        /// <summary>
        /// Gets or sets a specific channel by index.
        /// </summary>
        /// <param name="index">The channel index.</param>
        public NativeList<TVec> this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                CheckAccess();
                CheckIndex(index);
                return index switch
                {
                    0 => channel0,
                    1 => channel1,
                    2 => channel2,
                    3 => channel3,
#if UNITY_8UV_SUPPORT
                    4 => channel4,
                    5 => channel5,
                    6 => channel6,
                    7 => channel7,
#endif
                    _ => throw new ArgumentOutOfRangeException(nameof(index), index, null),
                };
            }
        }

        //TODO: does it ever happen that we have holes in the channels aka. uv4 but no uv3? If not then why do we not just give a max index instead?
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Get(int index, out NativeList<TVec> uvs)
        {
            CheckAccess();
            CheckIndex(index);

            var (list, res) = index switch
            {
                0 when first.x => (channel0, true),
                1 when first.y => (channel1, true),
                2 when first.z => (channel2, true),
                3 when first.w => (channel3, true),
#if UNITY_8UV_SUPPORT
                4 when second.x => (channel4, true),
                5 when second.y => (channel5, true),
                6 when second.z => (channel6, true),
                7 when second.w => (channel7, true),
#endif
                _ => (default, false),
            };

            uvs = list;
            return res;
        }

        public UVChannels(Allocator allocator)
        {
            channel0 = new NativeList<TVec>(allocator);
            channel1 = new NativeList<TVec>(allocator);
            channel2 = new NativeList<TVec>(allocator);
            channel3 = new NativeList<TVec>(allocator);
#if UNITY_8UV_SUPPORT
            channel4 = new NativeList<TVec>(allocator);
            channel5 = new NativeList<TVec>(allocator);
            channel6 = new NativeList<TVec>(allocator);
            channel7 = new NativeList<TVec>(allocator);
#endif

            first = default;
            second = default;
        }

        /// <summary>
        /// Resizes all channels at once.
        /// </summary>
        /// <param name="capacity">The new capacity.</param>
        /// <param name="trimExcess">If excess memory should be trimmed.</param>
        [SuppressMessage("ReSharper", "EnforceIfStatementBraces")]
        public void Resize(int capacity, bool trimExcess = false)
        {
            CheckAccess();
            // only resize if the channel is used
            if (first.x) channel0.Resize(capacity, NativeArrayOptions.ClearMemory);
            if (first.y) channel1.Resize(capacity, NativeArrayOptions.ClearMemory);
            if (first.z) channel2.Resize(capacity, NativeArrayOptions.ClearMemory);
            if (first.w) channel3.Resize(capacity, NativeArrayOptions.ClearMemory);
#if UNITY_8UV_SUPPORT
            if (second.x) channel4.Resize(capacity, NativeArrayOptions.ClearMemory);
            if (second.y) channel5.Resize(capacity, NativeArrayOptions.ClearMemory);
            if (second.z) channel6.Resize(capacity, NativeArrayOptions.ClearMemory);
            if (second.w) channel7.Resize(capacity, NativeArrayOptions.ClearMemory);
#endif
        }

        public void MarkAsUsed(int index)
        {
            CheckIndex(index);
            switch (index)
            {
               case 0: first.x = true; break;
               case 1: first.y = true; break;
               case 2: first.z = true; break;
               case 3: first.w = true; break;
#if UNITY_8UV_SUPPORT
               case 4: second.x = true; break;
               case 5: second.y = true; break;
               case 6: second.z = true; break;
               case 7: second.w = true; break;
#endif
            }
        }

        public void ClearChannel(int index)
        {
            CheckAccess();
            CheckIndex(index);
            switch (index)
            {
                case 0: first.x = false; channel0.Clear(); break;
                case 1: first.y = false; channel1.Clear(); break;
                case 2: first.z = false; channel2.Clear(); break;
                case 3: first.w = false; channel3.Clear(); break;
#if UNITY_8UV_SUPPORT
                case 4: second.x = false; channel4.Clear(); break;
                case 5: second.y = false; channel5.Clear(); break;
                case 6: second.z = false; channel6.Clear(); break;
                case 7: second.w = false; channel7.Clear(); break;
#endif
            }
        }

        public void Dispose()
        {
            channel0.Dispose();
            channel1.Dispose();
            channel2.Dispose();
            channel3.Dispose();
#if UNITY_8UV_SUPPORT
            channel4.Dispose();
            channel5.Dispose();
            channel6.Dispose();
            channel7.Dispose();
#endif
        }

        [Conditional("ENABLE_UNITY_COLLECTIONS_CHECKS")]
        private void CheckAccess()
        {
            if (!IsUsed)
            {
                throw new AccessViolationException("UVChannels is not used, do not access it in code!");
            }
        }

        [Conditional("ENABLE_UNITY_COLLECTIONS_CHECKS")]
        private static void CheckIndex(int index)
        {
            if (index < 0)
            {
                throw new ArgumentException("index must be greater than or equal to zero");
            }

            if (index >= MeshUtils.UVChannelCount)
            {
                throw new ArgumentException($"index must be less than the maximum UV Channel count of {MeshUtils.UVChannelCount}");
            }
        }
    }
}
