using UnityEngine;

public class CirclePathWalker : IPathWalker
{
    private float radius;
    private Vector3 center;
    private Color color;
    private float baseSpeed;
    private float currentSpeed;
    private bool initialized = false;
    private Vector3 targetDirection;

    public CirclePathWalker(float radius, Vector3 center, Color color, float baseSpeed)
    {
        this.radius = radius;
        this.center = center;
        this.color = color;
        this.baseSpeed = baseSpeed;
    }

    public void Initialize(Vector3 currentPosition)
    {
        center = currentPosition;
        center.y = 0f;
        initialized = true;
    }

    public void UpdatePath(Vector3 currentPosition)
    {
        if (!initialized) Initialize(currentPosition);

        // Adjust speed based on radius
        currentSpeed = baseSpeed * Mathf.Clamp(radius / 3f, 0.5f, 2f);

        // Calculate current angle
        float currentAngle = Time.time * currentSpeed / radius;

        // Calculate target position and direction
        Vector3 targetPosition = center + new Vector3(
            Mathf.Sin(currentAngle) * radius,
            0f,
            Mathf.Cos(currentAngle) * radius);

        targetDirection = new Vector3(
            Mathf.Cos(currentAngle),
            0f,
            -Mathf.Sin(currentAngle)).normalized;

        // Calculate path correction
        Vector3 toCenter = center - currentPosition;
        toCenter.y = 0f;
        float distanceFromPath = Mathf.Abs(toCenter.magnitude - radius);

        if (distanceFromPath > 0.01f)
        {
            Vector3 correction = toCenter.normalized * distanceFromPath * 0.75f;
            if (toCenter.magnitude <= radius) correction = -correction;
            targetDirection = (targetDirection + Vector3.ProjectOnPlane(correction, targetDirection)).normalized;
        }
    }

    public Vector3 GetTargetDirection()
    {
        return targetDirection;
    }

    public Vector3 GetTargetVelocity()
    {
        return targetDirection * currentSpeed;
    }

    public float GetTrajectoryCorrection()
    {
        return 1f;
    }

    public void DrawPath()
    {
        float heightOffset = 0.05f;
        int segments = 32;

        UltiDraw.Begin();

        Vector3 prevPoint = center + new Vector3(0f, heightOffset, radius);
        for (int i = 1; i <= segments; i++)
        {
            float angle = (float)i / segments * Mathf.PI * 2f;
            Vector3 nextPoint = center + new Vector3(
                Mathf.Sin(angle) * radius,
                heightOffset,
                Mathf.Cos(angle) * radius);

            UltiDraw.DrawLine(
                prevPoint,
                nextPoint,
                0.03f,
                0.03f,
                color.Opacity(1.0f));

            prevPoint = nextPoint;
        }

        UltiDraw.End();
    }
}