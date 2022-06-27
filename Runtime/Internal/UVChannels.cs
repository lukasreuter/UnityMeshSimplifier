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
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace UnityMeshSimplifier.Internal
{
#warning add a bool to indicate that this struct is unused
    internal struct UVChannels<TVec> : IDisposable where TVec : unmanaged
    {
        // private static readonly int UVChannelCount = MeshUtils.UVChannelCount;

        // private NativeArray<UnsafeList<TVec>> channels;
        // private TVec[][] channelsData = null;

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

        private bool valid;

        public bool IsValid => valid;

        /*public TVec[][] Data
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                for (int i = 0; i < UVChannelCount; i++)
                {
                    if (channels[i] != null)
                    {
                        channelsData[i] = channels[i].Data;
                    }
                    else
                    {
                        channelsData[i] = null;
                    }
                }
                return channelsData;
            }
        }*/

        /// <summary>
        /// Gets or sets a specific channel by index.
        /// </summary>
        /// <param name="index">The channel index.</param>
        public NativeList<TVec> this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                CheckAccess(index);
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
/*
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                CheckAccess(index);
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
                channels[index] = value;
            }
*/
        }

        public UVChannels(Allocator allocator)
        {
            // channels = new NativeArray<UnsafeList<TVec>>(UVChannelCount, allocator);

            // for (var i = 0; i < channels.Length; ++i)
            // {
            //     channels[i] = new UnsafeList<TVec>(0, allocator);
            // }

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

            // new ResizableArray<TVec>[UVChannelCount];
            // channelsData = new TVec[UVChannelCount][];
            valid = false;
        }

        /// <summary>
        /// Resizes all channels at once.
        /// </summary>
        /// <param name="capacity">The new capacity.</param>
        /// <param name="trimExcess">If excess memory should be trimmed.</param>
        public void Resize(int capacity, bool trimExcess = false)
        {
            channel0.Resize(capacity, NativeArrayOptions.ClearMemory);
            channel1.Resize(capacity, NativeArrayOptions.ClearMemory);
            channel2.Resize(capacity, NativeArrayOptions.ClearMemory);
            channel3.Resize(capacity, NativeArrayOptions.ClearMemory);
#if UNITY_8UV_SUPPORT
            channel4.Resize(capacity, NativeArrayOptions.ClearMemory);
            channel5.Resize(capacity, NativeArrayOptions.ClearMemory);
            channel6.Resize(capacity, NativeArrayOptions.ClearMemory);
            channel7.Resize(capacity, NativeArrayOptions.ClearMemory);
#endif

            // for (int i = 0; i < UVChannelCount; i++)
            // {
            //     // if (channels[i] != null)
            //     {
            //         channels[i].Resize(capacity, NativeArrayOptions.ClearMemory);
            //     }
            // }
        }

        public void MarkAsValid()
        {
            valid = true;
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

            // for (var i = 0; i < channels.Length; ++i)
            // {
            //     channels[i].Dispose();
            // }

            // channels.Dispose();
        }

        private void CheckAccess(int index)
        {
            if (!valid)
            {
                throw new AccessViolationException("UVChannels is not used, do not access it in code!");
            }

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
