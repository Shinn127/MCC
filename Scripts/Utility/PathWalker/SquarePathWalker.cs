using UnityEngine;

public class SquarePathWalker : IPathWalker
{
    private float size;
    private Vector3 center;
    private Color color;
    private float speed;
    private bool initialized = false;
    private int currentCornerIndex = 0;
    private Vector3[] corners = new Vector3[4];
    private Vector3 targetDirection;

    public SquarePathWalker(float size, Vector3 center, Color color, float speed)
    {
        this.size = size;
        this.center = center;
        this.color = color;
        this.speed = speed;
    }

    public void Initialize(Vector3 currentPosition)
    {
        center = currentPosition;
        center.y = 0f;

        float halfSize = size / 2f;
        corners[0] = center + new Vector3(-halfSize, 0f, halfSize);
        corners[1] = center + new Vector3(halfSize, 0f, halfSize);
        corners[2] = center + new Vector3(halfSize, 0f, -halfSize);
        corners[3] = center + new Vector3(-halfSize, 0f, -halfSize);

        initialized = true;
    }

    public void UpdatePath(Vector3 currentPosition)
    {
        if (!initialized) Initialize(currentPosition);

        // Get current target corner
        Vector3 targetCorner = corners[currentCornerIndex];

        // Check if we need to switch to next corner
        if (Vector3.Distance(currentPosition, targetCorner) < size * 0.3f)
        {
            currentCornerIndex = (currentCornerIndex + 1) % 4;
            targetCorner = corners[currentCornerIndex];
        }

        // Calculate target direction
        targetDirection = (targetCorner - currentPosition).normalized;
        targetDirection.y = 0f;

        // Calculate edge correction
        Vector3 edgeVector = targetCorner - corners[(currentCornerIndex - 1 + 4) % 4];
        Vector3 edgeNormal = new Vector3(-edgeVector.z, 0f, edgeVector.x).normalized;

        float edgeDistance = Vector3.Dot(currentPosition - corners[(currentCornerIndex - 1 + 4) % 4], edgeNormal);
        if (Mathf.Abs(edgeDistance) > 0.01f)
        {
            targetDirection = (targetDirection - edgeNormal * edgeDistance * 0.75f).normalized;
        }
    }

    public Vector3 GetTargetDirection()
    {
        return targetDirection;
    }

    public Vector3 GetTargetVelocity()
    {
        return targetDirection * speed;
    }

    public float GetTrajectoryCorrection()
    {
        return 1f;
    }

    public void DrawPath()
    {
        float heightOffset = 0.05f;

        UltiDraw.Begin();

        for (int i = 0; i < 4; i++)
        {
            Vector3 start = corners[i] + Vector3.up * heightOffset;
            Vector3 end = corners[(i + 1) % 4] + Vector3.up * heightOffset;

            UltiDraw.DrawLine(
                start,
                end,
                0.03f,
                0.03f,
                color.Opacity(1.0f));
        }

        // Draw current target corner
        UltiDraw.DrawCircle(
            corners[currentCornerIndex] + Vector3.up * heightOffset,
            0.15f,
            UltiDraw.Green.Opacity(0.9f));

        UltiDraw.End();
    }
}