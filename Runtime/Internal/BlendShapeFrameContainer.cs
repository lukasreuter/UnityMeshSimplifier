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

using System;
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace UnityMeshSimplifier.Internal
{
    internal struct BlendShapeFrameContainer : IDisposable
    {
        private readonly float frameWeight;
        private UnsafeList<Vector3> deltaVertices;
        private UnsafeList<Vector3> deltaNormals;
        private UnsafeList<Vector3> deltaTangents;

        public BlendShapeFrameContainer(BlendShapeFrame frame, Allocator allocator)
        {
            frameWeight = frame.FrameWeight;

            deltaVertices = new UnsafeList<Vector3>(frame.DeltaVertices.Length, allocator);
            foreach (var deltaVertex in deltaVertices)
            {
                deltaVertices.Add(deltaVertex);
            }

            deltaNormals = new UnsafeList<Vector3>(frame.DeltaNormals.Length, allocator);
            foreach (var deltaNormal in deltaNormals)
            {
                deltaNormals.Add(deltaNormal);
            }

            deltaTangents = new UnsafeList<Vector3>(frame.DeltaTangents.Length, allocator);
            foreach (var deltaTangent in deltaTangents)
            {
                deltaTangents.Add(deltaTangent);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void MoveVertexElement(int dst, int src)
        {
            deltaVertices[dst] = deltaVertices[src];
            deltaNormals[dst] = deltaNormals[src];
            deltaTangents[dst] = deltaTangents[src];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void InterpolateVertexAttributes(int dst, int i0, int i1, int i2, in Vector3 barycentricCoord)
        {
            deltaVertices[dst] = (deltaVertices[i0] * barycentricCoord.x) + (deltaVertices[i1] * barycentricCoord.y) + (deltaVertices[i2] * barycentricCoord.z);
            deltaNormals[dst] = Vector3.Normalize((deltaNormals[i0] * barycentricCoord.x) + (deltaNormals[i1] * barycentricCoord.y) + (deltaNormals[i2] * barycentricCoord.z));
            deltaTangents[dst] = Vector3.Normalize((deltaTangents[i0] * barycentricCoord.x) + (deltaTangents[i1] * barycentricCoord.y) + (deltaTangents[i2] * barycentricCoord.z));
        }

        public void Resize(int length, bool trimExcess = false)
        {
            deltaVertices.Resize(length, NativeArrayOptions.ClearMemory);
            deltaNormals.Resize(length, NativeArrayOptions.ClearMemory);
            deltaTangents.Resize(length, NativeArrayOptions.ClearMemory);
        }

        public BlendShapeFrame ToBlendShapeFrame()
        {
            var resultVertices = deltaVertices.ToArrayNBC();
            var resultNormals = deltaNormals.ToArrayNBC();
            var resultTangents = deltaTangents.ToArrayNBC();
            return new BlendShapeFrame(frameWeight, resultVertices, resultNormals, resultTangents);
        }

        public void Dispose()
        {
            deltaVertices.Dispose();
            deltaNormals.Dispose();
            deltaTangents.Dispose();
        }
    }
}
