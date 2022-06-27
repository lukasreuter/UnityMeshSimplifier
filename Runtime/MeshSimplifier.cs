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

#region Original License
/////////////////////////////////////////////
//
// Mesh Simplification Tutorial
//
// (C) by Sven Forstmann in 2014
//
// License : MIT
// http://opensource.org/licenses/MIT
//
//https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
#endregion

#if UNITY_2018_2_OR_NEWER
#define UNITY_8UV_SUPPORT
#endif

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.NotBurstCompatible;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityMeshSimplifier.Internal;
using static Unity.Mathematics.math;

namespace UnityMeshSimplifier
{
    /// <summary>
    /// The mesh simplifier.
    /// Deeply based on https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification but rewritten completely in C#.
    /// </summary>
    [BurstCompile]
    public struct MeshSimplifier : IJob, IDisposable
    {
        #region Consts & Static Read-Only
        private const int TriangleEdgeCount = 3;
        private const int TriangleVertexCount = 3;
        private const double DoubleEpsilon = 1.0E-3;
        private const double DenomEpilson = 0.00000001;
        private static readonly int UVChannelCount = MeshUtils.UVChannelCount;
        #endregion

        #region Fields
        private SimplificationOptions simplificationOptions;
        private float quality;
        private bool verbose;

        private int subMeshCount;
        private NativeList<int> subMeshOffsets;
        private NativeList<Triangle> triangles;
        private NativeList<Vertex> vertices;
        private NativeList<Ref> refs;

        private NativeList<Vector3> vertNormals;
        private NativeList<Vector4> vertTangents;
        private UVChannels<Vector2> vertUV2D;
        private UVChannels<Vector3> vertUV3D;
        private UVChannels<Vector4> vertUV4D;
        private NativeList<Color> vertColors;
        private NativeList<BoneWeight> vertBoneWeights;
        private NativeList<BlendShapeContainer> blendShapes;

        private NativeArray<Matrix4x4> bindposes;

        // Pre-allocated buffers
        private FixedArray3<double> errArr;
        private FixedArray3<int> attributeIndexArr;
        private NativeParallelHashSet<Triangle> triangleHashSet1;
        private NativeParallelHashSet<Triangle> triangleHashSet2;
        #endregion

        #region Properties
        /// <summary>
        /// Gets or sets all of the simplification options as a single block.
        /// Default value: SimplificationOptions.Default
        /// </summary>
        public SimplificationOptions SimplificationOptions
        {
            get => simplificationOptions;
            set
            {
                ValidateOptions(value);
                simplificationOptions = value;
            }
        }

        public float Quality
        {
            get => quality;
            set => quality = value;
        }

        /// <summary>
        /// Gets or sets if verbose information should be printed to the console.
        /// Default value: false
        /// </summary>
        public bool Verbose
        {
            get => verbose;
            set => verbose = value;
        }

        /// <summary>
        /// Gets or sets the vertex positions.
        /// </summary>
        public Vector3[] Vertices
        {
            get
            {
                int vertexCount = this.vertices.Length;
                var vertices = new Vector3[vertexCount];
                var vertArr = this.vertices;
                for (int i = 0; i < vertexCount; i++)
                {
                    vertices[i] = (Vector3) vertArr[i].p;
                }
                return vertices;
            }
        }

        /// <summary>
        /// Gets the count of sub-meshes.
        /// </summary>
        public int SubMeshCount => subMeshCount;

        /// <summary>
        /// Gets the count of blend shapes.
        /// </summary>
        public int BlendShapeCount => (blendShapes.IsCreated ? blendShapes.Length : 0);

        /// <summary>
        /// Gets or sets the vertex normals.
        /// </summary>
        public Vector3[] Normals => (vertNormals.IsCreated ? vertNormals.ToArray() : Array.Empty<Vector3>());

        // set
        // {
        // InitializeVertexAttribute(value, ref vertNormals, "normals");
        // }
        /// <summary>
        /// Gets or sets the vertex tangents.
        /// </summary>
        public Vector4[] Tangents => (vertTangents.IsCreated ? vertTangents.ToArray() : Array.Empty<Vector4>());

        // set
        // {
        // InitializeVertexAttribute(value, ref vertTangents, "tangents");
        // }
        /// <summary>
        /// Gets or sets the vertex 2D UV set 1.
        /// </summary>
        public Vector2[] UV1
        {
            get => GetUVs2D(0);
            set => SetUVs(0, value);
        }

        /// <summary>
        /// Gets or sets the vertex 2D UV set 2.
        /// </summary>
        public Vector2[] UV2
        {
            get => GetUVs2D(1);
            set => SetUVs(1, value);
        }

        /// <summary>
        /// Gets or sets the vertex 2D UV set 3.
        /// </summary>
        public Vector2[] UV3
        {
            get => GetUVs2D(2);
            set => SetUVs(2, value);
        }

        /// <summary>
        /// Gets or sets the vertex 2D UV set 4.
        /// </summary>
        public Vector2[] UV4
        {
            get => GetUVs2D(3);
            set => SetUVs(3, value);
        }

#if UNITY_8UV_SUPPORT
        /// <summary>
        /// Gets or sets the vertex 2D UV set 5.
        /// </summary>
        public Vector2[] UV5
        {
            get => GetUVs2D(4);
            set => SetUVs(4, value);
        }

        /// <summary>
        /// Gets or sets the vertex 2D UV set 6.
        /// </summary>
        public Vector2[] UV6
        {
            get => GetUVs2D(5);
            set => SetUVs(5, value);
        }

        /// <summary>
        /// Gets or sets the vertex 2D UV set 7.
        /// </summary>
        public Vector2[] UV7
        {
            get => GetUVs2D(6);
            set => SetUVs(6, value);
        }

        /// <summary>
        /// Gets or sets the vertex 2D UV set 8.
        /// </summary>
        public Vector2[] UV8
        {
            get => GetUVs2D(7);
            set => SetUVs(7, value);
        }
#endif

        /// <summary>
        /// Gets or sets the vertex colors.
        /// </summary>
        public Color[] Colors =>
            (vertColors.IsCreated ? vertColors.ToArray() : Array.Empty<Color>());

        /// <summary>
        /// Gets or sets the vertex bone weights.
        /// </summary>
        public BoneWeight[] BoneWeights =>
            (vertBoneWeights.IsCreated ? vertBoneWeights.ToArray() : Array.Empty<BoneWeight>());

        #endregion

        #region Constructors
/*
        /// <summary>
        /// Creates a new mesh simplifier.
        /// </summary>
        public MeshSimplifier(int capacity = 0)
        {
            triangles = new ResizableArray<Triangle>(capacity);
            vertices = new ResizableArray<Vertex>(capacity);
            refs = new ResizableArray<Ref>(capacity);
        }
*/

        /// <summary>
        /// Creates a new mesh simplifier.
        /// </summary>
        /// <param name="mesh">The original mesh to simplify.</param>
        public MeshSimplifier(Mesh mesh, Allocator allocator) : this()
        {
            if (mesh == null)
            {
                #warning throw an error instead
                return;
            }

            Initialize(mesh, allocator);
        }
        #endregion

        #region Private Methods
        #region Initialize Vertex Attribute
/*
        private void InitializeVertexAttribute<T>(T[] attributeValues, ref ResizableArray<T> attributeArray, string attributeName)
        {
            if (attributeValues != null && attributeValues.Length == vertices.Length)
            {
                if (attributeArray == null)
                {
                    attributeArray = new ResizableArray<T>(attributeValues.Length, attributeValues.Length);
                }
                else
                {
                    attributeArray.Resize(attributeValues.Length);
                }

                var arrayData = attributeArray.Data;
                Array.Copy(attributeValues, 0, arrayData, 0, attributeValues.Length);
            }
            else
            {
                if (attributeValues != null && attributeValues.Length > 0)
                {
                    Debug.LogErrorFormat("Failed to set vertex attribute '{0}' with {1} length of array, when {2} was needed.", attributeName, attributeValues.Length, vertices.Length);
                }
                attributeArray = null;
            }
        }
*/
        #endregion

        #region Calculate Error
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double VertexError(ref SymmetricMatrix q, double x, double y, double z) =>
            q.m0 * x * x + 2 * q.m1 * x * y + 2 * q.m2 * x * z + 2 * q.m3 * x + q.m4 * y * y
            + 2 * q.m5 * y * z + 2 * q.m6 * y + q.m7 * z * z + 2 * q.m8 * z + q.m9;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double CurvatureError(in Vertex vert0, in Vertex vert1)
        {
            double diffVector = (vert0.p - vert1.p).Magnitude;

            var trianglesWithViOrVjOrBoth = triangleHashSet1;
            trianglesWithViOrVjOrBoth.Clear();
            GetTrianglesContainingVertex(in vert0, trianglesWithViOrVjOrBoth);
            GetTrianglesContainingVertex(in vert1, trianglesWithViOrVjOrBoth);

            var trianglesWithViAndVjBoth = triangleHashSet2;
            trianglesWithViAndVjBoth.Clear();
            GetTrianglesContainingBothVertices(in vert0, in vert1, trianglesWithViAndVjBoth);

            double maxDotOuter = 0;
            foreach (var triangleWithViOrVjOrBoth in trianglesWithViOrVjOrBoth)
            {
                double maxDotInner = 0;
                Vector3d normVecTriangleWithViOrVjOrBoth = triangleWithViOrVjOrBoth.n;

                foreach (var triangleWithViAndVjBoth in trianglesWithViAndVjBoth)
                {
                    Vector3d normVecTriangleWithViAndVjBoth = triangleWithViAndVjBoth.n;
                    double dot = Vector3d.Dot(ref normVecTriangleWithViOrVjOrBoth, ref normVecTriangleWithViAndVjBoth);

                    if (dot > maxDotInner)
                        maxDotInner = dot;
                }

                if (maxDotInner > maxDotOuter)
                    maxDotOuter = maxDotInner;
            }

            return diffVector * maxDotOuter;
        }

        private double CalculateError(in Vertex vert0, in Vertex vert1, out Vector3d result)
        {
            // compute interpolated vertex
            SymmetricMatrix q = (vert0.q + vert1.q);
            bool borderEdge = (vert0.borderEdge && vert1.borderEdge);
            double error = 0.0;
            double det = q.Determinant1();
            if (det != 0.0 && !borderEdge)
            {
                // q_delta is invertible
                result = new Vector3d(
                    -1.0 / det * q.Determinant2(),  // vx = A41/det(q_delta)
                    1.0 / det * q.Determinant3(),   // vy = A42/det(q_delta)
                    -1.0 / det * q.Determinant4()); // vz = A43/det(q_delta)

                double curvatureError = 0;
                if (simplificationOptions.PreserveSurfaceCurvature)
                {
                    curvatureError = CurvatureError(in vert0, in vert1);
                }

                error = VertexError(ref q, result.x, result.y, result.z) + curvatureError;
            }
            else
            {
                // det = 0 -> try to find best result
                Vector3d p1 = vert0.p;
                Vector3d p2 = vert1.p;
                Vector3d p3 = (p1 + p2) * 0.5f;
                double error1 = VertexError(ref q, p1.x, p1.y, p1.z);
                double error2 = VertexError(ref q, p2.x, p2.y, p2.z);
                double error3 = VertexError(ref q, p3.x, p3.y, p3.z);

                if (error1 < error2)
                {
                    if (error1 < error3)
                    {
                        error = error1;
                        result = p1;
                    }
                    else
                    {
                        error = error3;
                        result = p3;
                    }
                }
                else if (error2 < error3)
                {
                    error = error2;
                    result = p2;
                }
                else
                {
                    error = error3;
                    result = p3;
                }
            }
            return error;
        }
        #endregion

        #region Calculate Barycentric Coordinates
        private static void CalculateBarycentricCoords(in Vector3d point, in Vector3d a, in Vector3d b, in Vector3d c, out Vector3 result)
        {
            Vector3d v0 = (b - a), v1 = (c - a), v2 = (point - a);
            double d00 = Vector3d.Dot(ref v0, ref v0);
            double d01 = Vector3d.Dot(ref v0, ref v1);
            double d11 = Vector3d.Dot(ref v1, ref v1);
            double d20 = Vector3d.Dot(ref v2, ref v0);
            double d21 = Vector3d.Dot(ref v2, ref v1);
            double denom = d00 * d11 - d01 * d01;

            // Make sure the denominator is not too small to cause math problems
            if (Math.Abs(denom) < DenomEpilson)
            {
                denom = DenomEpilson;
            }

            double v = (d11 * d20 - d01 * d21) / denom;
            double w = (d00 * d21 - d01 * d20) / denom;
            double u = 1.0 - v - w;
            result = new Vector3((float)u, (float)v, (float)w);
        }
        #endregion

        #region Normalize Tangent
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector4 NormalizeTangent(Vector4 tangent)
        {
            var tangentVec = new Vector3(tangent.x, tangent.y, tangent.z);
            tangentVec.Normalize();
            return new Vector4(tangentVec.x, tangentVec.y, tangentVec.z, tangent.w);
        }
        #endregion

        #region Flipped
        /// <summary>
        /// Check if a triangle flips when this edge is removed
        /// </summary>
        private bool Flipped(in Vector3d p, int i0, int i1, in Vertex v0, NativeList<bool> deleted)
        {
            int tcount = v0.tcount;
            var refs = this.refs;
            var triangles = this.triangles;
            var vertices = this.vertices;
            for (int k = 0; k < tcount; k++)
            {
                Ref r = refs[v0.tstart + k];
                if (triangles[r.tid].deleted)
                    continue;

                int s = r.tvertex;
                int id1 = triangles[r.tid][(s + 1) % 3];
                int id2 = triangles[r.tid][(s + 2) % 3];
                if (id1 == i1 || id2 == i1)
                {
                    deleted[k] = true;
                    continue;
                }

                Vector3d d1 = vertices[id1].p - p;
                d1.Normalize();
                Vector3d d2 = vertices[id2].p - p;
                d2.Normalize();
                double dot = Vector3d.Dot(ref d1, ref d2);
                if (Math.Abs(dot) > 0.999)
                    return true;

                Vector3d n;
                Vector3d.Cross(ref d1, ref d2, out n);
                n.Normalize();
                deleted[k] = false;
                var triangle = triangles[r.tid]; //TODO: is just here because of the ref, remove the local var
                dot = Vector3d.Dot(ref n, ref triangle.n);
                if (dot < 0.2)
                    return true;
            }

            return false;
        }
        #endregion

        #region Update Triangles
        /// <summary>
        /// Update triangle connections and edge error after a edge is collapsed.
        /// </summary>
        private void UpdateTriangles(int i0, int ia0, in Vertex v, NativeList<bool> deleted, ref int deletedTriangles)
        {
            Vector3d p;
            int tcount = v.tcount;
            var triangles = this.triangles;
            var vertices = this.vertices;
            for (int k = 0; k < tcount; k++)
            {
                Ref r = refs[v.tstart + k];
                int tid = r.tid;
                Triangle t = triangles[tid];
                if (t.deleted)
                    continue;

                if (deleted[k])
                {
                    ref var triangle = ref triangles.ElementAt(tid);
                    triangle.deleted = true;
                    ++deletedTriangles;
                    continue;
                }

                t[r.tvertex] = i0;
                if (ia0 != -1)
                {
                    t.SetAttributeIndex(r.tvertex, ia0);
                }

                ref var v0 = ref vertices.ElementAt(t.v0);
                ref var v1 = ref vertices.ElementAt(t.v1);
                ref var v2 = ref vertices.ElementAt(t.v2);
                t.dirty = true;
                t.err0 = CalculateError(in v0, in v1, out p);
                t.err1 = CalculateError(in v1, in v2, out p);
                t.err2 = CalculateError(in v2, in v0, out p);
                t.err3 = MathHelper.Min(t.err0, t.err1, t.err2);
                triangles[tid] = t;
                refs.Add(r);
            }
        }
        #endregion

        #region Interpolate Vertex Attributes
        private void InterpolateVertexAttributes(int dst, int i0, int i1, int i2, in Vector3 barycentricCoord)
        {
            vertNormals[dst] = Vector3.Normalize((vertNormals[i0] * barycentricCoord.x) + (vertNormals[i1] * barycentricCoord.y) +
                                                 (vertNormals[i2] * barycentricCoord.z));

            vertTangents[dst] = NormalizeTangent((vertTangents[i0] * barycentricCoord.x) + (vertTangents[i1] * barycentricCoord.y) +
                                                 (vertTangents[i2] * barycentricCoord.z));

            if (vertUV2D.IsValid)
            {
                for (int i = 0; i < UVChannelCount; i++)
                {
                    var vertUV = vertUV2D[i];
                    if (!vertUV.IsEmpty)
                    {
                        vertUV[dst] = (vertUV[i0] * barycentricCoord.x) +
                                      (vertUV[i1] * barycentricCoord.y) +
                                      (vertUV[i2] * barycentricCoord.z);
                    }
                }
            }

            if (vertUV3D.IsValid)
            {
                for (int i = 0; i < UVChannelCount; i++)
                {
                    var vertUV = vertUV3D[i];
                    if (!vertUV.IsEmpty)
                    {
                        vertUV[dst] = (vertUV[i0] * barycentricCoord.x) +
                                      (vertUV[i1] * barycentricCoord.y) +
                                      (vertUV[i2] * barycentricCoord.z);
                    }
                }
            }

            if (vertUV4D.IsValid)
            {
                for (int i = 0; i < UVChannelCount; i++)
                {
                    var vertUV = vertUV4D[i];
                    if (!vertUV.IsEmpty)
                    {
                        vertUV[dst] = (vertUV[i0] * barycentricCoord.x) +
                                      (vertUV[i1] * barycentricCoord.y) +
                                      (vertUV[i2] * barycentricCoord.z);
                    }
                }
            }

            if (!vertColors.IsEmpty)
            {
                vertColors[dst] = (vertColors[i0] * barycentricCoord.x) +
                                  (vertColors[i1] * barycentricCoord.y) +
                                  (vertColors[i2] * barycentricCoord.z);
            }

            for (int i = 0; i < blendShapes.Length; i++)
            {
                blendShapes[i].InterpolateVertexAttributes(dst, i0, i1, i2, in barycentricCoord);
            }

            // TODO: How do we interpolate the bone weights? Do we have to?
        }
        #endregion

        #region Are UVs The Same
        private bool AreUVsTheSame(int channel, int indexA, int indexB)
        {
            if (vertUV2D.IsValid)
            {
                var vertUV = vertUV2D[channel];
                var uvA = vertUV[indexA];
                var uvB = vertUV[indexB];
                return uvA == uvB;
            }

            if (vertUV3D.IsValid)
            {
                var vertUV = vertUV3D[channel];
                var uvA = vertUV[indexA];
                var uvB = vertUV[indexB];
                return uvA == uvB;
            }

            if (vertUV4D.IsValid)
            {
                var vertUV = vertUV4D[channel];
                var uvA = vertUV[indexA];
                var uvB = vertUV[indexB];
                return uvA == uvB;
            }

            return false;
        }
        #endregion

        #region Remove Vertex Pass
        /// <summary>
        /// Remove vertices and mark deleted triangles
        /// </summary>
        private void RemoveVertexPass(
            int startTrisCount,
            int targetTrisCount,
            double threshold,
            NativeList<bool> deleted0,
            NativeList<bool> deleted1,
            ref int deletedTris)
        {
            var triangles = this.triangles;
            int triangleCount = this.triangles.Length;
            var vertices = this.vertices;

            Vector3d p;
            Vector3 barycentricCoord;
            for (int tid = 0; tid < triangleCount; tid++)
            {
                if (triangles[tid].dirty || triangles[tid].deleted || triangles[tid].err3 > threshold)
                    continue;

                triangles[tid].GetErrors(errArr);
                triangles[tid].GetAttributeIndices(attributeIndexArr);
                for (int edgeIndex = 0; edgeIndex < TriangleEdgeCount; edgeIndex++)
                {
                    if (errArr[edgeIndex] > threshold)
                        continue;

                    int nextEdgeIndex = ((edgeIndex + 1) % TriangleEdgeCount);
                    int i0 = triangles[tid][edgeIndex];
                    int i1 = triangles[tid][nextEdgeIndex];

                    // Border check
                    if (vertices[i0].borderEdge != vertices[i1].borderEdge)
                        continue;
                    // Seam check
                    else if (vertices[i0].uvSeamEdge != vertices[i1].uvSeamEdge)
                        continue;
                    // Foldover check
                    else if (vertices[i0].uvFoldoverEdge != vertices[i1].uvFoldoverEdge)
                        continue;
                    // If borders should be preserved
                    else if (simplificationOptions.PreserveBorderEdges && vertices[i0].borderEdge)
                        continue;
                    // If seams should be preserved
                    else if (simplificationOptions.PreserveUVSeamEdges && vertices[i0].uvSeamEdge)
                        continue;
                    // If foldovers should be preserved
                    else if (simplificationOptions.PreserveUVFoldoverEdges && vertices[i0].uvFoldoverEdge)
                        continue;

                    // Compute vertex to collapse to
                    CalculateError(vertices[i0], vertices[i1], out p);
                    deleted0.Resize(vertices[i0].tcount, NativeArrayOptions.ClearMemory); // normals temporarily
                    deleted1.Resize(vertices[i1].tcount, NativeArrayOptions.ClearMemory); // normals temporarily

                    // Don't remove if flipped
                    if (Flipped(in p, i0, i1, vertices[i0], deleted0))
                        continue;
                    if (Flipped(in p, i1, i0, vertices[i1], deleted1))
                        continue;

                    // Calculate the barycentric coordinates within the triangle
                    int nextNextEdgeIndex = ((edgeIndex + 2) % 3);
                    int i2 = triangles[tid][nextNextEdgeIndex];
                    CalculateBarycentricCoords(in p, vertices[i0].p, vertices[i1].p, vertices[i2].p, out barycentricCoord);

                    // Not flipped, so remove edge
                    {
                        ref var vert = ref vertices.ElementAt(i0);
                        vert.p = p;
                        vert.q += vertices[i1].q;
                    }

                    // Interpolate the vertex attributes
                    int ia0 = attributeIndexArr[edgeIndex];
                    int ia1 = attributeIndexArr[nextEdgeIndex];
                    int ia2 = attributeIndexArr[nextNextEdgeIndex];
                    InterpolateVertexAttributes(ia0, ia0, ia1, ia2, in barycentricCoord);

                    if (vertices[i0].uvSeamEdge)
                    {
                        ia0 = -1;
                    }

                    int tStart = refs.Length;
                    UpdateTriangles(i0, ia0, vertices[i0], deleted0, ref deletedTris);
                    UpdateTriangles(i0, ia0, vertices[i1], deleted1, ref deletedTris);

                    int tCount = refs.Length - tStart;
                    if (tCount <= vertices[i0].tcount)
                    {
                        // save ram
                        if (tCount > 0)
                        {
                            // var refsArr = refs;

                            NativeArray<Ref>.Copy(refs.AsArray(), tStart, refs.AsArray(), vertices[i0].tstart, tCount);
                            // Array.Copy(refsArr, tStart, refsArr, vertices[i0].tstart, tCount);
                        }
                    }
                    else
                    {
                        // append
                        ref var vert = ref vertices.ElementAt(i0);
                        vert.tstart = tStart;
                    }

                    ref var vert0 = ref vertices.ElementAt(i0);
                    vert0.tcount = tCount;
                    break;
                }

                // Check if we are already done
                if ((startTrisCount - deletedTris) <= targetTrisCount)
                    break;
            }
        }
        #endregion

        #region Update Mesh
        /// <summary>
        /// Compact triangles, compute edge error and build reference list.
        /// </summary>
        /// <param name="iteration">The iteration index.</param>
        private void UpdateMesh(int iteration)
        {
            var triangles = this.triangles;
            var vertices = this.vertices;

            int triangleCount = this.triangles.Length;
            int vertexCount = this.vertices.Length;
            if (iteration > 0) // compact triangles
            {
                int dst = 0;
                for (int i = 0; i < triangleCount; i++)
                {
                    if (!triangles[i].deleted)
                    {
                        if (dst != i)
                        {
                            var triangle = triangles[i];
                            triangle.index = dst;
                            triangles[dst] = triangle;
                        }
                        dst++;
                    }
                }
                this.triangles.Resize(dst, NativeArrayOptions.ClearMemory);
                triangles = this.triangles;
                triangleCount = dst;
            }

            UpdateReferences();

            // Identify boundary : vertices[].border=0,1
            if (iteration == 0)
            {
                var refs = this.refs;

                var vcount = new NativeList<int>(8, Allocator.Temp);
                var vids = new NativeList<int>(8, Allocator.Temp);
                int vsize = 0;
                for (int i = 0; i < vertexCount; ++i)
                {
                    ref var vertex = ref vertices.ElementAt(i);
                    vertex.borderEdge = false;
                    vertex.uvSeamEdge = false;
                    vertex.uvFoldoverEdge = false;
                }

                int ofs;
                int id;
                int borderVertexCount = 0;
                double borderMinX = double.MaxValue;
                double borderMaxX = double.MinValue;
                var vertexLinkDistanceSqr = simplificationOptions.VertexLinkDistance * simplificationOptions.VertexLinkDistance;
                for (int i = 0; i < vertexCount; i++)
                {
                    int tstart = vertices[i].tstart;
                    int tcount = vertices[i].tcount;
                    vcount.Clear();
                    vids.Clear();
                    vsize = 0;

                    for (int j = 0; j < tcount; j++)
                    {
                        int tid = refs[tstart + j].tid;
                        for (int k = 0; k < TriangleVertexCount; k++)
                        {
                            ofs = 0;
                            id = triangles[tid][k];
                            while (ofs < vsize)
                            {
                                if (vids[ofs] == id)
                                    break;

                                ++ofs;
                            }

                            if (ofs == vsize)
                            {
                                vcount.Add(1);
                                vids.Add(id);
                                ++vsize;
                            }
                            else
                            {
                                ++vcount[ofs];
                            }
                        }
                    }

                    for (int j = 0; j < vsize; j++)
                    {
                        if (vcount[j] == 1)
                        {
                            id = vids[j];
                            ref var vertex = ref vertices.ElementAt(id);
                            vertex.borderEdge = true;
                            ++borderVertexCount;

                            if (simplificationOptions.EnableSmartLink)
                            {
                                if (vertex.p.x < borderMinX)
                                {
                                    borderMinX = vertex.p.x;
                                }
                                if (vertex.p.x > borderMaxX)
                                {
                                    borderMaxX = vertex.p.x;
                                }
                            }
                        }
                    }
                }

                if (simplificationOptions.EnableSmartLink)
                {
                    // First find all border vertices
                    var borderVertices = new NativeArray<BorderVertex>(borderVertexCount, Allocator.Temp);
                    int borderIndexCount = 0;
                    double borderAreaWidth = borderMaxX - borderMinX;
                    for (int i = 0; i < vertexCount; i++)
                    {
                        if (vertices[i].borderEdge)
                        {
                            int vertexHash = (int)(((((vertices[i].p.x - borderMinX) / borderAreaWidth) * 2.0) - 1.0) * int.MaxValue);
                            borderVertices[borderIndexCount] = new BorderVertex(i, vertexHash);
                            ++borderIndexCount;
                        }
                    }

                    #warning fix this
                    // Sort the border vertices by hash
                    borderVertices.Sort(new BorderVertexComparer());
                    // Array.Sort(borderVertices, 0, borderIndexCount, BorderVertexComparer.instance);

                    // Calculate the maximum hash distance based on the maximum vertex link distance
                    double vertexLinkDistance = Math.Sqrt(vertexLinkDistanceSqr);
                    int hashMaxDistance = Math.Max((int)((vertexLinkDistance / borderAreaWidth) * int.MaxValue), 1);

                    // Then find identical border vertices and bind them together as one
                    for (int i = 0; i < borderIndexCount; i++)
                    {
                        int myIndex = borderVertices[i].index;
                        if (myIndex == -1)
                            continue;

                        ref var myVert = ref vertices.ElementAt(myIndex);
                        var myPoint = vertices[myIndex].p;
                        for (int j = i + 1; j < borderIndexCount; j++)
                        {
                            var borderVertex = borderVertices[j];
                            int otherIndex = borderVertex.index;
                            if (otherIndex == -1)
                            {
                                continue;
                            }
                            else if ((borderVertex.hash - borderVertices[i].hash) > hashMaxDistance) // There is no point to continue beyond this point
                            {
                                break;
                            }

                            ref var otherVert = ref vertices.ElementAt(otherIndex);
                            var otherPoint = otherVert.p;
                            var sqrX = ((myPoint.x - otherPoint.x) * (myPoint.x - otherPoint.x));
                            var sqrY = ((myPoint.y - otherPoint.y) * (myPoint.y - otherPoint.y));
                            var sqrZ = ((myPoint.z - otherPoint.z) * (myPoint.z - otherPoint.z));
                            var sqrMagnitude = sqrX + sqrY + sqrZ;

                            if (sqrMagnitude <= vertexLinkDistanceSqr)
                            {
                                borderVertex.index = -1; // NOTE: This makes sure that the "other" vertex is not processed again
                                borderVertices[j] = borderVertex;

                                myVert.borderEdge = false;
                                otherVert.borderEdge = false;

                                if (AreUVsTheSame(0, myIndex, otherIndex))
                                {
                                    myVert.uvFoldoverEdge = true;
                                    otherVert.uvFoldoverEdge = true;
                                }
                                else
                                {
                                    myVert.uvSeamEdge = true;
                                    otherVert.uvSeamEdge = true;
                                }

                                int otherTriangleCount = otherVert.tcount;
                                int otherTriangleStart = otherVert.tstart;
                                for (int k = 0; k < otherTriangleCount; k++)
                                {
                                    var r = refs[otherTriangleStart + k];
                                    ref var triangle = ref triangles.ElementAt(r.tid);
                                    triangle[r.tvertex] = myIndex;
                                }
                            }
                        }
                    }

                    // Update the references again
                    UpdateReferences();
                }

                // Init Quadrics by Plane & Edge Errors
                //
                // required at the beginning ( iteration == 0 )
                // recomputing during the simplification is not required,
                // but mostly improves the result for closed meshes
                for (int i = 0; i < vertexCount; i++)
                {
                    ref var vertex = ref vertices.ElementAt(i);
                    vertex.q = new SymmetricMatrix();
                }

                for (int i = 0; i < triangleCount; i++)
                {
                    ref var triangle = ref triangles.ElementAt(i);
                    ref var v0 = ref vertices.ElementAt(triangle.v0);
                    ref var v1 = ref vertices.ElementAt(triangle.v1);
                    ref var v2 = ref vertices.ElementAt(triangle.v2);

                    var p0 = v0.p;
                    var p1 = v1.p;
                    var p2 = v2.p;
                    var p10 = p1 - p0;
                    var p20 = p2 - p0;
                    Vector3d.Cross(ref p10, ref p20, out var n);
                    n.Normalize();
                    triangle.n = n;

                    var sm = new SymmetricMatrix(n.x, n.y, n.z, -Vector3d.Dot(ref n, ref p0));
                    v0.q += sm;
                    v1.q += sm;
                    v2.q += sm;
                }

                for (int i = 0; i < triangleCount; i++)
                {
                    // Calc Edge Error
                    ref var triangle = ref triangles.ElementAt(i);
                    ref readonly var v0 = ref vertices.ElementAt(triangle.v0);
                    ref readonly var v1 = ref vertices.ElementAt(triangle.v1);
                    ref readonly var v2 = ref vertices.ElementAt(triangle.v2);
                    triangle.err0 = CalculateError(in v0, in v1, out _);
                    triangle.err1 = CalculateError(in v1, in v2, out _);
                    triangle.err2 = CalculateError(in v2, in v0, out _);
                    triangle.err3 = MathHelper.Min(triangle.err0, triangle.err1, triangle.err2);
                }
            }
        }
        #endregion

        #region Update References
        private void UpdateReferences()
        {
            int triangleCount = this.triangles.Length;
            int vertexCount = this.vertices.Length;
            var triangles = this.triangles;
            var vertices = this.vertices;

            // Init Reference ID list
            for (int i = 0; i < vertexCount; i++)
            {
                ref var vertex = ref vertices.ElementAt(i);
                vertex.tstart = 0;
                vertex.tcount = 0;
            }

            for (int i = 0; i < triangleCount; i++)
            {
                var triangle = triangles[i];
                ref var v0 = ref vertices.ElementAt(triangle.v0);
                ref var v1 = ref vertices.ElementAt(triangle.v1);
                ref var v2 = ref vertices.ElementAt(triangle.v2);
                v0.tcount += 1;
                v1.tcount += 1;
                v2.tcount += 1;
            }

            int tStart = 0;
            for (int i = 0; i < vertexCount; i++)
            {
                ref var vertex = ref vertices.ElementAt(i);
                vertex.tstart = tStart;
                tStart += vertex.tcount;
                vertex.tcount = 0;
            }

            // Write References
            this.refs.Resize(tStart, NativeArrayOptions.ClearMemory);
            var refs = this.refs;
            for (int i = 0; i < triangleCount; i++)
            {
                ref var v0 = ref vertices.ElementAt(triangles[i].v0);
                ref var v1 = ref vertices.ElementAt(triangles[i].v1);
                ref var v2 = ref vertices.ElementAt(triangles[i].v2);
                int start0 = v0.tstart;
                int count0 = v0.tcount++;
                int start1 = v1.tstart;
                int count1 = v1.tcount++;
                int start2 = v2.tstart;
                int count2 = v2.tcount++;

                ref var ref0 = ref refs.ElementAt(start0 + count0);
                ref var ref1 = ref refs.ElementAt(start1 + count1);
                ref var ref2 = ref refs.ElementAt(start2 + count2);
                ref0.Set(i, 0);
                ref1.Set(i, 1);
                ref2.Set(i, 2);
            }
        }
        #endregion

        #region Compact Mesh
        /// <summary>
        /// Finally compact mesh before exiting.
        /// </summary>
        private void CompactMesh()
        {
            int dst = 0;
            var vertices = this.vertices;
            int vertexCount = this.vertices.Length;
            for (int i = 0; i < vertexCount; ++i)
            {
                ref var vertex = ref vertices.ElementAt(i);
                vertex.tcount = 0;
            }

            var vertNormals = this.vertNormals;
            var vertTangents = this.vertTangents;
            var vertUV2D = this.vertUV2D;
            var vertUV3D = this.vertUV3D;
            var vertUV4D = this.vertUV4D;
            var vertColors = this.vertColors;
            var vertBoneWeights = this.vertBoneWeights;
            var blendShapes = this.blendShapes;

            int lastSubMeshIndex = -1;
            subMeshOffsets.Clear();
            subMeshOffsets.Length = subMeshCount;

            var triangles = this.triangles;
            int triangleCount = this.triangles.Length;
            for (int i = 0; i < triangleCount; i++)
            {
                var triangle = triangles[i];
                if (!triangle.deleted)
                {
                    if (triangle.va0 != triangle.v0)
                    {
                        int iDest = triangle.va0;
                        int iSrc = triangle.v0;
                        ref var destVertex = ref vertices.ElementAt(iDest);
                        destVertex.p = vertices[iSrc].p;

                        if (!vertBoneWeights.IsEmpty)
                        {
                            vertBoneWeights[iDest] = vertBoneWeights[iSrc];
                        }

                        triangle.v0 = triangle.va0;
                    }
                    if (triangle.va1 != triangle.v1)
                    {
                        int iDest = triangle.va1;
                        int iSrc = triangle.v1;
                        ref var destVertex = ref vertices.ElementAt(iDest);
                        destVertex.p = vertices[iSrc].p;

                        if (!vertBoneWeights.IsEmpty)
                        {
                            vertBoneWeights[iDest] = vertBoneWeights[iSrc];
                        }

                        triangle.v1 = triangle.va1;
                    }
                    if (triangle.va2 != triangle.v2)
                    {
                        int iDest = triangle.va2;
                        int iSrc = triangle.v2;
                        ref var destVertex = ref vertices.ElementAt(iDest);
                        destVertex.p = vertices[iSrc].p;

                        if (!vertBoneWeights.IsEmpty)
                        {
                            vertBoneWeights[iDest] = vertBoneWeights[iSrc];
                        }

                        triangle.v2 = triangle.va2;
                    }
                    int newTriangleIndex = dst++;
                    ref var newTriangle = ref triangles.ElementAt(newTriangleIndex);
                    newTriangle = triangle;
                    newTriangle.index = newTriangleIndex;

                    ref var v0 = ref vertices.ElementAt(triangle.v0);
                    ref var v1 = ref vertices.ElementAt(triangle.v1);
                    ref var v2 = ref vertices.ElementAt(triangle.v2);
                    v0.tcount = 1;
                    v1.tcount = 1;
                    v2.tcount = 1;

                    if (triangle.subMeshIndex > lastSubMeshIndex)
                    {
                        for (int j = lastSubMeshIndex + 1; j < triangle.subMeshIndex; j++)
                        {
                            subMeshOffsets[j] = newTriangleIndex;
                        }
                        subMeshOffsets[triangle.subMeshIndex] = newTriangleIndex;
                        lastSubMeshIndex = triangle.subMeshIndex;
                    }
                }
            }

            triangleCount = dst;
            for (int i = lastSubMeshIndex + 1; i < subMeshCount; i++)
            {
                subMeshOffsets[i] = triangleCount;
            }

            this.triangles.Resize(triangleCount, NativeArrayOptions.ClearMemory);
            triangles = this.triangles;

            dst = 0;
            for (int i = 0; i < vertexCount; i++)
            {
                ref var vert = ref vertices.ElementAt(i);
                if (vert.tcount > 0)
                {
                    vert.tstart = dst;

                    if (dst != i)
                    {
                        ref var dstVertex = ref vertices.ElementAt(dst);
                        dstVertex.index = dst;
                        dstVertex.p = vert.p;

                        if (!vertNormals.IsEmpty)
                        {
                            vertNormals[dst] = vertNormals[i];
                        }

                        if (!vertTangents.IsEmpty)
                        {
                            vertTangents[dst] = vertTangents[i];
                        }

                        if (vertUV2D.IsValid)
                        {
                            for (int j = 0; j < UVChannelCount; j++)
                            {
                                var vertUV = vertUV2D[j];
                                if (!vertUV.IsEmpty)
                                {
                                    vertUV[dst] = vertUV[i];
                                }
                            }
                        }
                        if (vertUV3D.IsValid)
                        {
                            for (int j = 0; j < UVChannelCount; j++)
                            {
                                var vertUV = vertUV3D[j];
                                if (!vertUV.IsEmpty)
                                {
                                    vertUV[dst] = vertUV[i];
                                }
                            }
                        }
                        if (vertUV4D.IsValid)
                        {
                            for (int j = 0; j < UVChannelCount; j++)
                            {
                                var vertUV = vertUV4D[j];
                                if (!vertUV.IsEmpty)
                                {
                                    vertUV[dst] = vertUV[i];
                                }
                            }
                        }

                        if (!vertColors.IsEmpty)
                        {
                            vertColors[dst] = vertColors[i];
                        }

                        if (!vertBoneWeights.IsEmpty)
                        {
                            vertBoneWeights[dst] = vertBoneWeights[i];
                        }

                        for (int shapeIndex = 0; shapeIndex < this.blendShapes.Length; shapeIndex++)
                        {
                            blendShapes[shapeIndex].MoveVertexElement(dst, i);
                        }
                    }
                    ++dst;
                }
            }

            for (int i = 0; i < triangleCount; i++)
            {
                var triangle = triangles[i];
                triangle.v0 = vertices[triangle.v0].tstart;
                triangle.v1 = vertices[triangle.v1].tstart;
                triangle.v2 = vertices[triangle.v2].tstart;
                triangles[i] = triangle;
            }

            vertexCount = dst;
            this.vertices.Resize(vertexCount, NativeArrayOptions.ClearMemory);
            this.vertNormals.Resize(vertexCount, NativeArrayOptions.ClearMemory);
            this.vertTangents.Resize(vertexCount, NativeArrayOptions.ClearMemory);
            this.vertUV2D.Resize(vertexCount, true);
            this.vertUV3D.Resize(vertexCount, true);
            this.vertUV4D.Resize(vertexCount, true);
            this.vertColors.Resize(vertexCount, NativeArrayOptions.ClearMemory);
            this.vertBoneWeights.Resize(vertexCount, NativeArrayOptions.ClearMemory);

            for (int i = 0; i < this.blendShapes.Length; i++)
            {
                blendShapes[i].Resize(vertexCount, false);
            }
        }
        #endregion

        #region Calculate Sub Mesh Offsets
        private void CalculateSubMeshOffsets()
        {
            int lastSubMeshIndex = -1;
            subMeshOffsets.Clear();
            subMeshOffsets.Length = subMeshCount;

            var triangles = this.triangles;
            int triangleCount = this.triangles.Length;
            for (int i = 0; i < triangleCount; i++)
            {
                var triangle = triangles[i];
                if (triangle.subMeshIndex > lastSubMeshIndex)
                {
                    for (int j = lastSubMeshIndex + 1; j < triangle.subMeshIndex; j++)
                    {
                        subMeshOffsets[j] = i;
                    }
                    subMeshOffsets[triangle.subMeshIndex] = i;
                    lastSubMeshIndex = triangle.subMeshIndex;
                }
            }

            for (int i = lastSubMeshIndex + 1; i < subMeshCount; i++)
            {
                subMeshOffsets[i] = triangleCount;
            }
        }
        #endregion

        #region Triangle helper functions
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void GetTrianglesContainingVertex(in Vertex vert, NativeParallelHashSet<Triangle> tris)
        {
            int trianglesCount = vert.tcount;
            int startIndex = vert.tstart;

            for (int a = startIndex; a < startIndex + trianglesCount; a++)
            {
                tris.Add(triangles[refs[a].tid]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void GetTrianglesContainingBothVertices(in Vertex vert0, in Vertex vert1, NativeParallelHashSet<Triangle> tris)
        {
            int triangleCount = vert0.tcount;
            int startIndex = vert0.tstart;

            for (int refIndex = startIndex; refIndex < (startIndex + triangleCount); refIndex++)
            {
                int tid = refs[refIndex].tid;
                Triangle tri = triangles[tid];

                if (vertices[tri.v0].index == vert1.index ||
                    vertices[tri.v1].index == vert1.index ||
                    vertices[tri.v2].index == vert1.index)
                {
                    tris.Add(tri);
                }
            }
        }
        #endregion Triangle helper functions
        #endregion

        #region Public Methods
        #region Sub-Meshes
        /// <summary>
        /// Returns the triangle indices for all sub-meshes.
        /// </summary>
        /// <returns>The triangle indices for all sub-meshes.</returns>
        public int[][] GetAllSubMeshTriangles()
        {
            var indices = new int[subMeshCount][];
            for (int subMeshIndex = 0; subMeshIndex < subMeshCount; subMeshIndex++)
            {
                indices[subMeshIndex] = GetSubMeshTriangles(subMeshIndex);
            }
            return indices;
        }

        /// <summary>
        /// Returns the triangle indices for a specific sub-mesh.
        /// </summary>
        /// <param name="subMeshIndex">The sub-mesh index.</param>
        /// <returns>The triangle indices.</returns>
        public int[] GetSubMeshTriangles(int subMeshIndex)
        {
            if (subMeshIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(subMeshIndex), "The sub-mesh index is negative.");

            // First get the sub-mesh offsets
            if (subMeshOffsets.IsEmpty)
            {
                CalculateSubMeshOffsets();
            }

            if (subMeshIndex >= subMeshOffsets.Length)
                throw new ArgumentOutOfRangeException(nameof(subMeshIndex), "The sub-mesh index is greater than or equals to the sub mesh count.");
            else if (subMeshOffsets.Length != subMeshCount)
                throw new InvalidOperationException("The sub-mesh triangle offsets array is not the same size as the count of sub-meshes. This should not be possible to happen.");

            var triangles = this.triangles;
            int triangleCount = this.triangles.Length;

            int startOffset = subMeshOffsets[subMeshIndex];
            if (startOffset >= triangleCount)
                return new int[0];

            int endOffset = ((subMeshIndex + 1) < subMeshCount ? subMeshOffsets[subMeshIndex + 1] : triangleCount);
            int subMeshTriangleCount = endOffset - startOffset;
            if (subMeshTriangleCount < 0) subMeshTriangleCount = 0;
            int[] subMeshIndices = new int[subMeshTriangleCount * 3];

            Debug.AssertFormat(startOffset >= 0, "The start sub mesh offset at index {0} was below zero ({1}).", subMeshIndex, startOffset);
            Debug.AssertFormat(endOffset >= 0, "The end sub mesh offset at index {0} was below zero ({1}).", subMeshIndex + 1, endOffset);
            Debug.AssertFormat(startOffset < triangleCount, "The start sub mesh offset at index {0} was higher or equal to the triangle count ({1} >= {2}).", subMeshIndex, startOffset, triangleCount);
            Debug.AssertFormat(endOffset <= triangleCount, "The end sub mesh offset at index {0} was higher than the triangle count ({1} > {2}).", subMeshIndex + 1, endOffset, triangleCount);

            for (int triangleIndex = startOffset; triangleIndex < endOffset; triangleIndex++)
            {
                var triangle = triangles[triangleIndex];
                int offset = (triangleIndex - startOffset) * 3;
                subMeshIndices[offset] = triangle.v0;
                subMeshIndices[offset + 1] = triangle.v1;
                subMeshIndices[offset + 2] = triangle.v2;
            }

            return subMeshIndices;
        }

        /// <summary>
        /// Clears out all sub-meshes.
        /// </summary>
        public void ClearSubMeshes()
        {
            subMeshCount = 0;
            subMeshOffsets.Clear();
            triangles.Clear();
        }

        /// <summary>
        /// Adds a sub-mesh triangle indices for a specific sub-mesh.
        /// </summary>
        /// <param name="triangles">The triangle indices.</param>
        public void AddSubMeshTriangles(int[] triangles)
        {
            if (triangles == null)
                throw new ArgumentNullException(nameof(triangles));
            else if ((triangles.Length % TriangleVertexCount) != 0)
                throw new ArgumentException("The index array length must be a multiple of 3 in order to represent triangles.", nameof(triangles));

            int subMeshIndex = subMeshCount++;
            int triangleIndexStart = this.triangles.Length;
            int subMeshTriangleCount = triangles.Length / TriangleVertexCount;
            this.triangles.Resize(this.triangles.Length + subMeshTriangleCount, NativeArrayOptions.ClearMemory);
            var trisArr = this.triangles;
            for (int i = 0; i < subMeshTriangleCount; i++)
            {
                int offset = i * 3;
                int v0 = triangles[offset];
                int v1 = triangles[offset + 1];
                int v2 = triangles[offset + 2];
                int triangleIndex = triangleIndexStart + i;
                trisArr[triangleIndex] = new Triangle(triangleIndex, v0, v1, v2, subMeshIndex);
            }
        }

        /// <summary>
        /// Adds several sub-meshes at once with their triangle indices for each sub-mesh.
        /// </summary>
        /// <param name="triangles">The triangle indices for each sub-mesh.</param>
        public void AddSubMeshTriangles(int[][] triangles)
        {
            if (triangles == null)
                throw new ArgumentNullException(nameof(triangles));

            int totalTriangleCount = 0;
            for (int i = 0; i < triangles.Length; i++)
            {
                if (triangles[i] == null)
                    throw new ArgumentException(string.Format("The index array at index {0} is null.", i));
                else if ((triangles[i].Length % TriangleVertexCount) != 0)
                    throw new ArgumentException(string.Format("The index array length at index {0} must be a multiple of 3 in order to represent triangles.", i), nameof(triangles));

                totalTriangleCount += triangles[i].Length / TriangleVertexCount;
            }

            int triangleIndexStart = this.triangles.Length;
            this.triangles.Resize(this.triangles.Length + totalTriangleCount, NativeArrayOptions.ClearMemory);
            var trisArr = this.triangles;

            for (int i = 0; i < triangles.Length; i++)
            {
                int subMeshIndex = subMeshCount++;
                var subMeshTriangles = triangles[i];
                int subMeshTriangleCount = subMeshTriangles.Length / TriangleVertexCount;
                for (int j = 0; j < subMeshTriangleCount; j++)
                {
                    int offset = j * 3;
                    int v0 = subMeshTriangles[offset];
                    int v1 = subMeshTriangles[offset + 1];
                    int v2 = subMeshTriangles[offset + 2];
                    int triangleIndex = triangleIndexStart + j;
                    trisArr[triangleIndex] = new Triangle(triangleIndex, v0, v1, v2, subMeshIndex);
                }

                triangleIndexStart += subMeshTriangleCount;
            }
        }
        #endregion

        #region UV Sets
        #region Getting
        /// <summary>
        /// Returns the UVs (2D) from a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <returns>The UVs.</returns>
        public Vector2[] GetUVs2D(int channel)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));

            return vertUV2D[channel].ToArrayNBC();
        }

        /// <summary>
        /// Returns the UVs (3D) from a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <returns>The UVs.</returns>
        public Vector3[] GetUVs3D(int channel)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));

            return vertUV3D[channel].ToArrayNBC();
        }

        /// <summary>
        /// Returns the UVs (4D) from a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <returns>The UVs.</returns>
        public Vector4[] GetUVs4D(int channel)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));

            return vertUV4D[channel].ToArrayNBC();
        }

        /// <summary>
        /// Returns the UVs (2D) from a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        public void GetUVs(int channel, List<Vector2> uvs)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));
            else if (uvs == null)
                throw new ArgumentNullException(nameof(uvs));

            uvs.Clear();
            var uvData = vertUV2D[channel];
            for (var i = 0; i < uvData.Length; ++i)
            {
                uvs.Add(uvData[i]);
            }
        }

        /// <summary>
        /// Returns the UVs (3D) from a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        public void GetUVs(int channel, List<Vector3> uvs)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));
            else if (uvs == null)
                throw new ArgumentNullException(nameof(uvs));

            uvs.Clear();
            var uvData = vertUV3D[channel];
            for (var i = 0; i < uvData.Length; ++i)
            {
                uvs.Add(uvData[i]);
            }
        }

        /// <summary>
        /// Returns the UVs (4D) from a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        public void GetUVs(int channel, List<Vector4> uvs)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));
            else if (uvs == null)
                throw new ArgumentNullException(nameof(uvs));

            uvs.Clear();
            var uvData = vertUV4D[channel];
            for (var i = 0; i < uvData.Length; ++i)
            {
                uvs.Add(uvData[i]);
            }
        }
        #endregion

        #region Setting
        /// <summary>
        /// Sets the UVs (2D) for a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        public void SetUVs(int channel, IList<Vector2> uvs)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));

            if (uvs != null && uvs.Count > 0)
            {
                int uvCount = uvs.Count;
                vertUV2D.MarkAsValid();
                var uvSet = vertUV2D[channel];

                uvSet.Resize(uvCount, NativeArrayOptions.ClearMemory);
                uvSet.CopyFromNBC(uvs.ToArray());
            }
            else
            {
                vertUV2D[channel].Clear();
            }

            if (vertUV3D.IsValid)
            {
                vertUV3D[channel].Clear();
            }
            if (vertUV4D.IsValid)
            {
                vertUV4D[channel].Clear();
            }
        }

        /// <summary>
        /// Sets the UVs (3D) for a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        public void SetUVs(int channel, IList<Vector3> uvs)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));

            if (uvs != null && uvs.Count > 0)
            {
                int uvCount = uvs.Count;
                vertUV3D.MarkAsValid();
                var uvSet = vertUV3D[channel];

                uvSet.Resize(uvCount, NativeArrayOptions.ClearMemory);
                uvSet.CopyFromNBC(uvs.ToArray());
            }
            else
            {
                vertUV3D[channel].Clear();
            }

            if (vertUV2D.IsValid)
            {
                vertUV2D[channel].Clear();
            }
            if (vertUV4D.IsValid)
            {
                vertUV4D[channel].Clear();
            }
        }

        /// <summary>
        /// Sets the UVs (4D) for a specific channel.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        public void SetUVs(int channel, IList<Vector4> uvs)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));

            if (uvs != null && uvs.Count > 0)
            {
                int uvCount = uvs.Count;
                vertUV4D.MarkAsValid();
                var uvSet = vertUV4D[channel];

                uvSet.Resize(uvCount, NativeArrayOptions.ClearMemory);
                uvSet.CopyFromNBC(uvs.ToArray());
            }
            else
            {
                vertUV4D[channel].Clear();
            }

            if (vertUV2D.IsValid)
            {
                vertUV2D[channel].Clear();
            }
            if (vertUV3D.IsValid)
            {
                vertUV3D[channel].Clear();
            }
        }

        /// <summary>
        /// Sets the UVs for a specific channel with a specific count of UV components.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        /// <param name="uvComponentCount">The count of UV components.</param>
        public void SetUVs(int channel, IList<Vector4> uvs, int uvComponentCount)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));
            else if (uvComponentCount < 0 || uvComponentCount > 4)
                throw new ArgumentOutOfRangeException(nameof(uvComponentCount));

            if (uvs != null && uvs.Count > 0 && uvComponentCount > 0)
            {
                if (uvComponentCount <= 2)
                {
                    var uv2D = MeshUtils.ConvertUVsTo2D(uvs);
                    SetUVs(channel, uv2D);
                }
                else if (uvComponentCount == 3)
                {
                    var uv3D = MeshUtils.ConvertUVsTo3D(uvs);
                    SetUVs(channel, uv3D);
                }
                else
                {
                    SetUVs(channel, uvs);
                }
            }
            else
            {
                if (vertUV2D.IsValid)
                {
                    vertUV2D[channel].Clear();
                }
                if (vertUV3D.IsValid)
                {
                    vertUV3D[channel].Clear();
                }
                if (vertUV4D.IsValid)
                {
                    vertUV4D[channel].Clear();
                }
            }
        }

        /// <summary>
        /// Sets the UVs for a specific channel and automatically detects the used components.
        /// </summary>
        /// <param name="channel">The channel index.</param>
        /// <param name="uvs">The UVs.</param>
        public void SetUVsAuto(int channel, IList<Vector4> uvs)
        {
            if (channel < 0 || channel >= UVChannelCount)
                throw new ArgumentOutOfRangeException(nameof(channel));

            int uvComponentCount = MeshUtils.GetUsedUVComponents(uvs);
            SetUVs(channel, uvs, uvComponentCount);
        }
        #endregion
        #endregion

        #region Blend Shapes
        /// <summary>
        /// Returns all blend shapes.
        /// </summary>
        /// <returns>An array of all blend shapes.</returns>
        public BlendShape[] GetAllBlendShapes()
        {
            var results = new BlendShape[blendShapes.Length];
            for (int i = 0; i < results.Length; i++)
            {
                results[i] = blendShapes[i].ToBlendShape();
            }
            return results;
        }

        /// <summary>
        /// Returns a specific blend shape.
        /// </summary>
        /// <param name="blendShapeIndex">The blend shape index.</param>
        /// <returns>The blend shape.</returns>
        public BlendShape GetBlendShape(int blendShapeIndex)
        {
            if (blendShapes.IsEmpty || blendShapeIndex < 0 || blendShapeIndex >= blendShapes.Length)
                throw new ArgumentOutOfRangeException(nameof(blendShapeIndex));

            return blendShapes[blendShapeIndex].ToBlendShape();
        }

        /// <summary>
        /// Clears all blend shapes.
        /// </summary>
        public void ClearBlendShapes()
        {
            blendShapes.Clear();
        }

        /// <summary>
        /// Adds a blend shape.
        /// </summary>
        /// <param name="blendShape">The blend shape to add.</param>
        public void AddBlendShape(BlendShape blendShape, Allocator allocator)
        {
            var frames = blendShape.Frames;
            if (frames == null || frames.Length == 0)
                throw new ArgumentException("The frames cannot be null or empty.", nameof(blendShape));

            var container = new BlendShapeContainer(blendShape, allocator);
            this.blendShapes.Add(container);
        }

        /// <summary>
        /// Adds several blend shapes.
        /// </summary>
        /// <param name="blendShapes">The blend shapes to add.</param>
        public void AddBlendShapes(BlendShape[] blendShapes, Allocator allocator)
        {
            if (blendShapes == null)
                throw new ArgumentNullException(nameof(blendShapes));

            for (int i = 0; i < blendShapes.Length; i++)
            {
                var frames = blendShapes[i].Frames;
                if (frames == null || frames.Length == 0)
                    throw new ArgumentException(string.Format("The frames of blend shape at index {0} cannot be null or empty.", i), nameof(blendShapes));

                var container = new BlendShapeContainer(blendShapes[i], allocator);
                this.blendShapes.Add(container);
            }
        }
        #endregion

        #region Initialize

        /// <summary>
        /// Initializes the algorithm with the original mesh.
        /// </summary>
        /// <param name="mesh">The mesh.</param>
        /// <param name="allocator"></param>
        public void Initialize(Mesh mesh, Allocator allocator)
        {
            if (mesh == null)
            {
                throw new ArgumentNullException(nameof(mesh));
            }

            refs = new NativeList<Ref>(0, allocator);
            this.blendShapes = new NativeList<BlendShapeContainer>(0, allocator);
            subMeshOffsets = new NativeList<int>(0, allocator);
            triangles = new NativeList<Triangle>(0, allocator);
            triangleHashSet1 = new NativeParallelHashSet<Triangle>(10, allocator);
            triangleHashSet2 = new NativeParallelHashSet<Triangle>(10, allocator);

            vertices = new NativeList<Vertex>(mesh.vertices.Length, allocator);
            //TODO: make this nicer
            vertices.CopyFromNBC(mesh.vertices.Select((v, index) => new Vertex(index, v)).ToArray());

            vertNormals = new NativeList<Vector3>(mesh.normals.Length, allocator);
            vertNormals.CopyFromNBC(mesh.normals);

            vertTangents = new NativeList<Vector4>(mesh.tangents.Length, allocator);
            vertTangents.CopyFromNBC(mesh.tangents);

            vertColors = new NativeList<Color>(mesh.colors.Length, allocator);
            vertColors.CopyFromNBC(mesh.colors);

            vertBoneWeights = new NativeList<BoneWeight>(mesh.boneWeights.Length, allocator);
            vertBoneWeights.CopyFromNBC(mesh.boneWeights);

            bindposes = new NativeArray<Matrix4x4>(mesh.bindposes, allocator);

            vertUV2D = new UVChannels<Vector2>(allocator);
            vertUV3D = new UVChannels<Vector3>(allocator);
            vertUV4D = new UVChannels<Vector4>(allocator);

            for (int channel = 0; channel < UVChannelCount; channel++)
            {
                if (simplificationOptions.ManualUVComponentCount)
                {
                    switch (simplificationOptions.UVComponentCount)
                    {
                        case 1:
                        case 2:
                            {
                                var uvs = MeshUtils.GetMeshUVs2D(mesh, channel);
                                SetUVs(channel, uvs);
                                break;
                            }
                        case 3:
                            {
                                var uvs = MeshUtils.GetMeshUVs3D(mesh, channel);
                                SetUVs(channel, uvs);
                                break;
                            }
                        case 4:
                            {
                                var uvs = MeshUtils.GetMeshUVs(mesh, channel);
                                SetUVs(channel, uvs);
                                break;
                            }
                    }
                }
                else
                {
                    var uvs = MeshUtils.GetMeshUVs(mesh, channel);
                    SetUVsAuto(channel, uvs);
                }
            }

            var blendShapes = MeshUtils.GetMeshBlendShapes(mesh);
            if (blendShapes != null && blendShapes.Length > 0)
            {
                AddBlendShapes(blendShapes, allocator);
            }

            ClearSubMeshes();

            int subMeshCount = mesh.subMeshCount;
            var subMeshTriangles = new int[subMeshCount][];
            for (int i = 0; i < subMeshCount; i++)
            {
                subMeshTriangles[i] = mesh.GetTriangles(i);
            }
            AddSubMeshTriangles(subMeshTriangles);
        }

        public void Dispose()
        {
            subMeshOffsets.Dispose();
            triangles.Dispose();
            vertices.Dispose();
            refs.Dispose();
            vertNormals.Dispose();
            vertTangents.Dispose();
            vertUV2D.Dispose();
            vertUV3D.Dispose();
            vertUV4D.Dispose();
            vertColors.Dispose();
            vertBoneWeights.Dispose();
            blendShapes.Dispose();
            bindposes.Dispose();
            triangleHashSet1.Dispose();
            triangleHashSet2.Dispose();
        }

        #endregion

        public void Execute()
        {
            SimplifyMesh(quality);
        }

        #region Simplify Mesh
        /// <summary>
        /// Simplifies the mesh to a desired quality.
        /// </summary>
        /// <param name="quality">The target quality (between 0 and 1).</param>
        public void SimplifyMesh(float quality)
        {
            quality = math.clamp(quality, 0f, 1f);

            int deletedTris = 0;
            using var deleted0 = new NativeList<bool>(20, Allocator.Temp)
            {
                Length = 20,
            };
            using var deleted1 = new NativeList<bool>(20, Allocator.Temp)
            {
                Length = 20,
            };
            var triangles = this.triangles;
            int triangleCount = this.triangles.Length;
            int startTrisCount = triangleCount;
            var vertices = this.vertices;
            int targetTrisCount = (int) math.round(triangleCount * quality);

            for (int iteration = 0; iteration < simplificationOptions.MaxIterationCount; iteration++)
            {
                if ((startTrisCount - deletedTris) <= targetTrisCount)
                    break;

                // Update mesh once in a while
                if ((iteration % 5) == 0)
                {
#warning no2 heavy method
                    UpdateMesh(iteration);
                    //TODO: are these even necessary?
                    triangles = this.triangles;
                    triangleCount = this.triangles.Length;
                    vertices = this.vertices;
                }

                // Clear dirty flag
                for (int i = 0; i < triangleCount; i++)
                {
                    ref var triangle = ref triangles.ElementAt(i);
                    triangle.dirty = false;
                }

                // All triangles with edges below the threshold will be removed
                //
                // The following numbers works well for most models.
                // If it does not, try to adjust the 3 parameters
                double threshold = 0.000000001 * Math.Pow(iteration + 3, simplificationOptions.Agressiveness);

                if (verbose)
                {
                    Debug.Log($"iteration {iteration} - triangles {(startTrisCount - deletedTris)} threshold {threshold}");
                }

#warning no1 heavy method
                // Remove vertices & mark deleted triangles
                RemoveVertexPass(startTrisCount, targetTrisCount, threshold, deleted0, deleted1, ref deletedTris);
            }

            CompactMesh();

            if (verbose)
            {
                Debug.Log($"Finished simplification with triangle count {this.triangles.Length}");
            }
        }

        /// <summary>
        /// Simplifies the mesh without losing too much quality.
        /// </summary>
        public void SimplifyMeshLossless()
        {
            int deletedTris = 0;
            using var deleted0 = new NativeList<bool>(0, Allocator.Temp);
            using var deleted1 = new NativeList<bool>(0, Allocator.Temp);
            var triangles = this.triangles;
            int triangleCount = this.triangles.Length;
            int startTrisCount = triangleCount;
            var vertices = this.vertices;

            for (int iteration = 0; iteration < 9999; iteration++)
            {
                // Update mesh constantly
                UpdateMesh(iteration);
                //TODO: these should be unnecessary
                triangles = this.triangles;
                triangleCount = this.triangles.Length;
                vertices = this.vertices;

                // Clear dirty flag
                for (int i = 0; i < triangleCount; i++)
                {
                    ref var triangle = ref triangles.ElementAt(i);
                    triangle.dirty = false;
                }

                // All triangles with edges below the threshold will be removed
                //
                // The following numbers works well for most models.
                // If it does not, try to adjust the 3 parameters
                double threshold = DoubleEpsilon;

                if (verbose)
                {
                    Debug.Log($"Lossless iteration {iteration} - triangles {triangleCount}");
                }

                // Remove vertices & mark deleted triangles
                RemoveVertexPass(startTrisCount, 0, threshold, deleted0, deleted1, ref deletedTris);

                if (deletedTris <= 0)
                    break;

                deletedTris = 0;
            }

            CompactMesh();

            if (verbose)
            {
                Debug.Log($"Finished simplification with triangle count {this.triangles.Length}");
            }
        }
        #endregion

        #region To Mesh
        /// <summary>
        /// Returns the resulting mesh.
        /// </summary>
        /// <returns>The resulting mesh.</returns>
        public Mesh ToMesh()
        {
            var verticesLocal = this.Vertices;
            var normals = this.Normals;
            var tangents = this.Tangents;
            var colors = this.Colors;
            var boneWeights = this.BoneWeights;
            var indices = GetAllSubMeshTriangles();
            var blendShapesLocal = GetAllBlendShapes();
            var bindPoses = this.bindposes.ToArray();

            List<Vector2>[] uvs2D = null;
            List<Vector3>[] uvs3D = null;
            List<Vector4>[] uvs4D = null;

            if (vertUV2D.IsValid)
            {
                uvs2D = new List<Vector2>[UVChannelCount];
                for (int channel = 0; channel < UVChannelCount; channel++)
                {
                    if (!vertUV2D[channel].IsEmpty)
                    {
                        var uvs = new List<Vector2>(verticesLocal.Length);
                        GetUVs(channel, uvs);
                        uvs2D[channel] = uvs;
                    }
                }
            }

            if (vertUV3D.IsValid)
            {
                uvs3D = new List<Vector3>[UVChannelCount];
                for (int channel = 0; channel < UVChannelCount; channel++)
                {
                    if (!vertUV3D[channel].IsEmpty)
                    {
                        var uvs = new List<Vector3>(verticesLocal.Length);
                        GetUVs(channel, uvs);
                        uvs3D[channel] = uvs;
                    }
                }
            }

            if (vertUV4D.IsValid)
            {
                uvs4D = new List<Vector4>[UVChannelCount];
                for (int channel = 0; channel < UVChannelCount; channel++)
                {
                    if (!vertUV4D[channel].IsEmpty)
                    {
                        var uvs = new List<Vector4>(verticesLocal.Length);
                        GetUVs(channel, uvs);
                        uvs4D[channel] = uvs;
                    }
                }
            }

            return MeshUtils.CreateMesh(verticesLocal, indices, normals, tangents, colors, boneWeights, uvs2D, uvs3D, uvs4D, bindPoses,
                blendShapesLocal);
        }

        #endregion

        #region Validate Options
        /// <summary>
        /// Validates simplification options.
        /// Will throw an exception if the options are invalid.
        /// </summary>
        /// <param name="options">The simplification options to validate.</param>
        /// <exception cref="ValidateSimplificationOptionsException">The exception thrown in case of invalid options.</exception>
        public static void ValidateOptions(SimplificationOptions options)
        {
            if (options.EnableSmartLink && options.VertexLinkDistance < 0.0)
                throw new ValidateSimplificationOptionsException(nameof(options.VertexLinkDistance), "The vertex link distance cannot be negative when smart linking is enabled.");

            if (options.MaxIterationCount <= 0)
                throw new ValidateSimplificationOptionsException(nameof(options.MaxIterationCount), "The max iteration count cannot be zero or negative, since there would be nothing for the algorithm to do.");

            if (options.Agressiveness <= 0.0)
                throw new ValidateSimplificationOptionsException(nameof(options.Agressiveness), "The aggressiveness has to be above zero to make sense. Recommended is around 7.");

            if (options.ManualUVComponentCount)
            {
                if (options.UVComponentCount < 0 || options.UVComponentCount > 4)
                    throw new ValidateSimplificationOptionsException(nameof(options.UVComponentCount), "The UV component count cannot be below 0 or above 4 when manual UV component count is enabled.");
            }
        }
        #endregion
        #endregion
    }
}
