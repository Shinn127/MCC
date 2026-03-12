// IPathWalker.cs
using UnityEngine;

public interface IPathWalker
{
    void Initialize(Vector3 center);
    void UpdatePath(Vector3 currentPosition);
    Vector3 GetTargetDirection();
    Vector3 GetTargetVelocity();
    float GetTrajectoryCorrection();
    void DrawPath();
}