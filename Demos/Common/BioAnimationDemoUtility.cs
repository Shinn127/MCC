using System;
using System.Collections.Generic;
using AI4Animation;
using Unity.Sentis;
using UnityEngine;

public static class BioAnimationDemoUtility
{
    public const int Framerate = 60;
    public const int Points = 111;
    public const int PointSamples = 12;
    public const int FuturePoints = 50;
    public const int RootPointIndex = 60;
    public const int PointDensity = 10;

    public const int TrajectoryFeatureCount = 84;
    public const int FutureTrajectoryOutputOffset = 3;
    public const int PostureOutputOffset = 39;

    public const int LeftHandBoneIndex = 11;
    public const int RightHandBoneIndex = 15;
    public const int LeftFootBoneIndex = 19;
    public const int RightFootBoneIndex = 23;
    public const int MaxHistoryLength = 600;
    public const int ClipConditionDimension = 128;

    public static readonly Color LeftHandTrailColor = new Color(0.87f, 0.26f, 0.49f, 0.95f);
    public static readonly Color RightHandTrailColor = new Color(0.24f, 0.60f, 0.91f, 0.95f);

    public static int GetBaseInputFeatureCount(int boneCount)
    {
        return TrajectoryFeatureCount + boneCount * 12;
    }

    public static int GetPhaseOutputOffset(int boneCount)
    {
        return PostureOutputOffset + boneCount * 12;
    }

    public static void InitializeMotionState(Actor actor, out Vector3[] positions, out Vector3[] rights, out Vector3[] ups, out Vector3[] velocities)
    {
        if (actor == null)
        {
            throw new ArgumentNullException(nameof(actor));
        }

        positions = new Vector3[actor.Bones.Length];
        rights = new Vector3[actor.Bones.Length];
        ups = new Vector3[actor.Bones.Length];
        velocities = new Vector3[actor.Bones.Length];

        for (int i = 0; i < actor.Bones.Length; i++)
        {
            positions[i] = actor.Bones[i].GetTransform().position;
            rights[i] = actor.Bones[i].GetTransform().right;
            ups[i] = actor.Bones[i].GetTransform().up;
            velocities[i] = Vector3.zero;
        }
    }

    public static Trajectory CreateTrajectory(Controller controller, Vector3 seedPosition, Vector3 seedDirection)
    {
        if (controller == null)
        {
            throw new ArgumentNullException(nameof(controller));
        }

        Trajectory trajectory = new Trajectory(Points, controller.GetNames(), seedPosition, seedDirection);
        SeedTrajectoryStyles(trajectory);
        return trajectory;
    }

    public static void SeedTrajectoryStyles(Trajectory trajectory)
    {
        if (trajectory == null || trajectory.Points.Length == 0 || trajectory.Points[0].Styles.Length == 0)
        {
            return;
        }

        for (int i = 0; i < trajectory.Points.Length; i++)
        {
            trajectory.Points[i].Styles[0] = 1f;
        }
    }

    public static void InitializeHistory(List<Vector3>[] histories)
    {
        if (histories == null)
        {
            throw new ArgumentNullException(nameof(histories));
        }

        for (int i = 0; i < histories.Length; i++)
        {
            histories[i] = histories[i] ?? new List<Vector3>(MaxHistoryLength);
            histories[i].Clear();
        }
    }

    public static void UpdateHistory(List<Vector3>[] histories, int index, Vector3 position)
    {
        if (histories == null || index < 0 || index >= histories.Length || histories[index] == null)
        {
            return;
        }

        if (histories[index].Count >= MaxHistoryLength)
        {
            histories[index].RemoveAt(0);
        }

        histories[index].Add(position);
    }

    public static IPathWalker CreatePathWalker(int pathMode, float circleRadius, Vector3 circleCenter, Color circleColor, float squareSize, Vector3 squareCenter, Color squareColor, float baseWalkingSpeed)
    {
        switch (pathMode)
        {
            case 1:
                return new CirclePathWalker(circleRadius, circleCenter, circleColor, baseWalkingSpeed);
            case 2:
                return new SquarePathWalker(squareSize, squareCenter, squareColor, baseWalkingSpeed);
            default:
                return null;
        }
    }

    public static float PoolBias(Controller controller, float[] styleWeights)
    {
        if (controller == null || styleWeights == null)
        {
            return 0f;
        }

        float bias = 0f;
        for (int i = 0; i < styleWeights.Length && i < controller.Styles.Length; i++)
        {
            float localBias = controller.Styles[i].Bias;
            float max = 0f;

            for (int j = 0; j < controller.Styles[i].Multipliers.Length; j++)
            {
                if (controller.Styles[i].Query() && Input.GetKey(controller.Styles[i].Multipliers[j].Key))
                {
                    max = Mathf.Max(max, controller.Styles[i].Bias * controller.Styles[i].Multipliers[j].Value);
                }
            }

            for (int j = 0; j < controller.Styles[i].Multipliers.Length; j++)
            {
                if (controller.Styles[i].Query() && Input.GetKey(controller.Styles[i].Multipliers[j].Key))
                {
                    localBias = Mathf.Min(max, localBias * controller.Styles[i].Multipliers[j].Value);
                }
            }

            bias += styleWeights[i] * localBias;
        }

        return bias;
    }

    public static Trajectory.Point GetSample(Trajectory trajectory, int index)
    {
        return trajectory.Points[Mathf.Clamp(index * PointDensity, 0, trajectory.Points.Length - 1)];
    }

    public static Trajectory.Point GetPreviousSample(Trajectory trajectory, int index)
    {
        return GetSample(trajectory, index / PointDensity);
    }

    public static Trajectory.Point GetNextSample(Trajectory trajectory, int index)
    {
        return index % PointDensity == 0 ? GetSample(trajectory, index / PointDensity) : GetSample(trajectory, index / PointDensity + 1);
    }

    public static void PredictTrajectory(Controller controller, Trajectory trajectory, PIDController pid, ref Vector3 targetDirection, ref Vector3 targetVelocity, ref float trajectoryCorrection, float targetGain, float targetDecay)
    {
        float turn = controller.QueryTurn();
        Vector3 move = controller.QueryMove();
        float[] style = controller.GetStyle();
        bool hasLocomotionStyle = style.Length > 1;
        bool control = turn != 0f || move.magnitude != 0f || (hasLocomotionStyle && style[1] != 0f);

        if (Time.time < 1f && hasLocomotionStyle)
        {
            move = Vector3.forward;
            turn = 0f;
            style = new float[style.Length];
            style[1] = 1f;
            control = true;
        }

        float curvature = trajectory.GetCurvature(0, Points, PointDensity);
        float target = PoolBias(controller, trajectory.Points[RootPointIndex].Styles);
        float current = trajectory.Points[RootPointIndex].GetVelocity().magnitude;
        float bias = target;

        if (turn == 0f)
        {
            bias += pid.Update(Utility.Interpolate(target, current, curvature), current, 1f / Framerate);
        }
        else
        {
            pid.Reset();
        }

        move = bias * move.normalized;
        if (move.magnitude == 0f && turn != 0f)
        {
            move = 2f / 3f * Vector3.forward;
        }
        else
        {
            bool hasTurnMultiplier = controller.Styles.Length > 1 && controller.Styles[1].Multipliers.Length > 1;
            bool modifierPressed = hasTurnMultiplier && Input.GetKey(controller.Styles[1].Multipliers[1].Key);
            if (move.z == 0f && turn != 0f && !modifierPressed)
            {
                move = bias * new Vector3(move.x, 0f, 1f).normalized;
            }
            else
            {
                move = Vector3.Lerp(move, bias * Vector3.forward, trajectory.Points[RootPointIndex].GetVelocity().magnitude / 6f);
            }
        }

        targetDirection = Vector3.Lerp(targetDirection, Quaternion.AngleAxis(turn * 60f, Vector3.up) * trajectory.Points[RootPointIndex].GetDirection(), control ? targetGain : targetDecay);
        targetVelocity = Vector3.Lerp(targetVelocity, Quaternion.LookRotation(targetDirection, Vector3.up) * move, control ? targetGain : targetDecay);
        trajectoryCorrection = Utility.Interpolate(trajectoryCorrection, Mathf.Max(move.normalized.magnitude, Mathf.Abs(turn)), control ? targetGain : targetDecay);

        Vector3[] blendedPositions = new Vector3[trajectory.Points.Length];
        blendedPositions[RootPointIndex] = trajectory.Points[RootPointIndex].GetTransformation().GetPosition();

        for (int i = RootPointIndex + 1; i < trajectory.Points.Length; i++)
        {
            float weight = (float)(i - RootPointIndex) / FuturePoints;
            float scalePos = 1f - Mathf.Pow(1f - weight, 0.75f);
            float scaleDir = 1f - Mathf.Pow(1f - weight, 1.25f);
            float scaleVel = 1f - Mathf.Pow(1f - weight, 1.0f);
            float scale = 1f / (trajectory.Points.Length - (RootPointIndex + 1f));

            blendedPositions[i] = blendedPositions[i - 1] + Vector3.Lerp(trajectory.Points[i].GetPosition() - trajectory.Points[i - 1].GetPosition(), scale * targetVelocity, scalePos);
            trajectory.Points[i].SetDirection(Vector3.Lerp(trajectory.Points[i].GetDirection(), targetDirection, scaleDir));
            trajectory.Points[i].SetVelocity(Vector3.Lerp(trajectory.Points[i].GetVelocity(), targetVelocity, scaleVel));
        }

        for (int i = RootPointIndex + 1; i < trajectory.Points.Length; i++)
        {
            trajectory.Points[i].SetPosition(blendedPositions[i]);
        }

        for (int i = RootPointIndex; i < trajectory.Points.Length; i++)
        {
            float weight = (float)(i - RootPointIndex) / FuturePoints;
            for (int j = 0; j < trajectory.Points[i].Styles.Length && j < style.Length && j < controller.Styles.Length; j++)
            {
                trajectory.Points[i].Styles[j] = Utility.Interpolate(trajectory.Points[i].Styles[j], style[j], Utility.Normalise(weight, 0f, 1f, controller.Styles[j].Transition, 1f));
            }
            Utility.Normalise(ref trajectory.Points[i].Styles);
            trajectory.Points[i].SetSpeed(Utility.Interpolate(trajectory.Points[i].GetSpeed(), targetVelocity.magnitude, control ? targetGain : targetDecay));
        }
    }

    public static void UpdateFutureTrajectory(Trajectory trajectory, Vector3 targetDirection, Vector3 targetVelocity, float targetGain, bool forceWalkStyle)
    {
        Vector3[] blendedPositions = new Vector3[trajectory.Points.Length];
        blendedPositions[RootPointIndex] = trajectory.Points[RootPointIndex].GetTransformation().GetPosition();

        for (int i = RootPointIndex + 1; i < trajectory.Points.Length; i++)
        {
            float weight = (float)(i - RootPointIndex) / FuturePoints;
            float scalePos = 1f - Mathf.Pow(1f - weight, 0.75f);
            float scaleDir = 1f - Mathf.Pow(1f - weight, 1.25f);
            float scaleVel = 1f - Mathf.Pow(1f - weight, 1.0f);
            float scale = 1f / (trajectory.Points.Length - (RootPointIndex + 1f));

            blendedPositions[i] = blendedPositions[i - 1] + Vector3.Lerp(trajectory.Points[i].GetPosition() - trajectory.Points[i - 1].GetPosition(), scale * targetVelocity, scalePos);
            trajectory.Points[i].SetDirection(Vector3.Lerp(trajectory.Points[i].GetDirection(), targetDirection, scaleDir));
            trajectory.Points[i].SetVelocity(Vector3.Lerp(trajectory.Points[i].GetVelocity(), targetVelocity, scaleVel));
        }

        for (int i = RootPointIndex + 1; i < trajectory.Points.Length; i++)
        {
            trajectory.Points[i].SetPosition(blendedPositions[i]);
        }

        for (int i = RootPointIndex; i < trajectory.Points.Length; i++)
        {
            float weight = (float)(i - RootPointIndex) / FuturePoints;
            if (forceWalkStyle)
            {
                int walkStyleIndex = trajectory.Points[i].Styles.Length > 1 ? 1 : 0;
                for (int j = 0; j < trajectory.Points[i].Styles.Length; j++)
                {
                    trajectory.Points[i].Styles[j] = Utility.Interpolate(trajectory.Points[i].Styles[j], j == walkStyleIndex ? 1f : 0f, Utility.Normalise(weight, 0f, 1f, 0.5f, 1f));
                }
                Utility.Normalise(ref trajectory.Points[i].Styles);
            }

            trajectory.Points[i].SetSpeed(Utility.Interpolate(trajectory.Points[i].GetSpeed(), targetVelocity.magnitude, targetGain));
        }
    }

    public static int EncodeInputFeatures(Trajectory trajectory, Actor actor, Vector3[] positions, Vector3[] rights, Vector3[] ups, Vector3[] velocities, float[] buffer)
    {
        Matrix4x4 currentRoot = trajectory.Points[RootPointIndex].GetTransformation();
        int offset = 0;

        for (int i = 0; i < PointSamples; i++)
        {
            Trajectory.Point sample = GetSample(trajectory, i);
            Vector3 pos = sample.GetPosition().PositionTo(currentRoot);
            Vector3 dir = sample.GetDirection().DirectionTo(currentRoot);
            Vector3 vel = sample.GetVelocity().DirectionTo(currentRoot);

            buffer[offset + i + 0] = pos.x;
            buffer[offset + i + 12] = pos.z;
            buffer[offset + i + 24] = dir.x;
            buffer[offset + i + 36] = dir.z;
            buffer[offset + i + 48] = vel.x;
            buffer[offset + i + 60] = vel.z;
            buffer[offset + i + 72] = sample.GetSpeed();
        }

        offset = TrajectoryFeatureCount;
        Matrix4x4 previousRoot = trajectory.Points[RootPointIndex - 1].GetTransformation();
        for (int i = 0; i < actor.Bones.Length; i++)
        {
            Vector3 pos = positions[i].PositionTo(previousRoot);
            Vector3 right = rights[i].DirectionTo(previousRoot);
            Vector3 up = ups[i].DirectionTo(previousRoot);
            Vector3 vel = velocities[i].DirectionTo(previousRoot);

            buffer[offset + 0 + i * 3 + actor.Bones.Length * 3 * 0] = pos.x;
            buffer[offset + 1 + i * 3 + actor.Bones.Length * 3 * 0] = pos.y;
            buffer[offset + 2 + i * 3 + actor.Bones.Length * 3 * 0] = pos.z;
            buffer[offset + 0 + i * 3 + actor.Bones.Length * 3 * 1] = right.x;
            buffer[offset + 1 + i * 3 + actor.Bones.Length * 3 * 1] = right.y;
            buffer[offset + 2 + i * 3 + actor.Bones.Length * 3 * 1] = right.z;
            buffer[offset + 0 + i * 3 + actor.Bones.Length * 3 * 2] = up.x;
            buffer[offset + 1 + i * 3 + actor.Bones.Length * 3 * 2] = up.y;
            buffer[offset + 2 + i * 3 + actor.Bones.Length * 3 * 2] = up.z;
            buffer[offset + 0 + i * 3 + actor.Bones.Length * 3 * 3] = vel.x;
            buffer[offset + 1 + i * 3 + actor.Bones.Length * 3 * 3] = vel.y;
            buffer[offset + 2 + i * 3 + actor.Bones.Length * 3 * 3] = vel.z;
        }

        return GetBaseInputFeatureCount(actor.Bones.Length);
    }

    public static void ShiftPastTrajectory(Trajectory trajectory)
    {
        for (int i = 0; i < RootPointIndex; i++)
        {
            trajectory.Points[i].SetPosition(trajectory.Points[i + 1].GetPosition());
            trajectory.Points[i].SetDirection(trajectory.Points[i + 1].GetDirection());
            trajectory.Points[i].SetVelocity(trajectory.Points[i + 1].GetVelocity());
            trajectory.Points[i].SetSpeed(trajectory.Points[i + 1].GetSpeed());
            for (int j = 0; j < trajectory.Points[i].Styles.Length; j++)
            {
                trajectory.Points[i].Styles[j] = trajectory.Points[i + 1].Styles[j];
            }
        }
    }

    public static Matrix4x4 ApplyPredictionToTrajectory(Trajectory trajectory, TensorFloat prediction, Matrix4x4 currentRoot, float trajectoryCorrection)
    {
        float rootStyle = trajectory.Points[RootPointIndex].Styles.Length > 0 ? trajectory.Points[RootPointIndex].Styles[0] : 0f;
        float update = Mathf.Pow(1f - rootStyle, 0.25f);
        Vector3 rootMotion = update * new Vector3(prediction[0], prediction[1], prediction[2]);
        rootMotion /= Framerate;

        Vector3 translationalOffset = new Vector3(rootMotion.x, 0f, rootMotion.z);
        float rotationalOffset = rootMotion.y;

        trajectory.Points[RootPointIndex].SetPosition(translationalOffset.PositionFrom(currentRoot));
        trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * trajectory.Points[RootPointIndex].GetDirection());
        trajectory.Points[RootPointIndex].SetVelocity(translationalOffset.DirectionFrom(currentRoot) * Framerate);

        Matrix4x4 nextRoot = trajectory.Points[RootPointIndex].GetTransformation();
        Vector3 futureOffset = translationalOffset.DirectionFrom(nextRoot);

        for (int i = RootPointIndex + 1; i < trajectory.Points.Length; i++)
        {
            trajectory.Points[i].SetPosition(trajectory.Points[i].GetPosition() + futureOffset);
            trajectory.Points[i].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * trajectory.Points[i].GetDirection());
            trajectory.Points[i].SetVelocity(trajectory.Points[i].GetVelocity() + futureOffset * Framerate);
        }

        for (int i = RootPointIndex + 1; i < trajectory.Points.Length; i++)
        {
            int previousSampleIndex = GetPreviousSample(trajectory, i).GetIndex() / PointDensity;
            int nextSampleIndex = GetNextSample(trajectory, i).GetIndex() / PointDensity;
            float factor = (float)(i % PointDensity) / PointDensity;

            Vector3 previousPosition = new Vector3(prediction[FutureTrajectoryOutputOffset + (previousSampleIndex - 6)], 0f, prediction[FutureTrajectoryOutputOffset + (previousSampleIndex - 6) + 6]).PositionFrom(nextRoot);
            Vector3 previousDirection = new Vector3(prediction[FutureTrajectoryOutputOffset + (previousSampleIndex - 6) + 12], 0f, prediction[FutureTrajectoryOutputOffset + (previousSampleIndex - 6) + 18]).normalized.DirectionFrom(nextRoot);
            Vector3 previousVelocity = new Vector3(prediction[FutureTrajectoryOutputOffset + (previousSampleIndex - 6) + 24], 0f, prediction[FutureTrajectoryOutputOffset + (previousSampleIndex - 6) + 30]).DirectionFrom(nextRoot);

            Vector3 nextPosition = new Vector3(prediction[FutureTrajectoryOutputOffset + (nextSampleIndex - 6)], 0f, prediction[FutureTrajectoryOutputOffset + (nextSampleIndex - 6) + 6]).PositionFrom(nextRoot);
            Vector3 nextDirection = new Vector3(prediction[FutureTrajectoryOutputOffset + (nextSampleIndex - 6) + 12], 0f, prediction[FutureTrajectoryOutputOffset + (nextSampleIndex - 6) + 18]).normalized.DirectionFrom(nextRoot);
            Vector3 nextVelocity = new Vector3(prediction[FutureTrajectoryOutputOffset + (nextSampleIndex - 6) + 24], 0f, prediction[FutureTrajectoryOutputOffset + (nextSampleIndex - 6) + 30]).DirectionFrom(nextRoot);

            Vector3 position = Vector3.Lerp(previousPosition, nextPosition, factor);
            Vector3 direction = Vector3.Lerp(previousDirection, nextDirection, factor).normalized;
            Vector3 velocity = Vector3.Lerp(previousVelocity, nextVelocity, factor);
            position = Vector3.Lerp(trajectory.Points[i].GetPosition() + velocity / Framerate, position, 0.5f);

            trajectory.Points[i].SetPosition(Utility.Interpolate(trajectory.Points[i].GetPosition(), position, trajectoryCorrection));
            trajectory.Points[i].SetDirection(Utility.Interpolate(trajectory.Points[i].GetDirection(), direction, trajectoryCorrection));
            trajectory.Points[i].SetVelocity(Utility.Interpolate(trajectory.Points[i].GetVelocity(), velocity, trajectoryCorrection));
        }

        return nextRoot;
    }

    public static void UpdatePostureState(TensorFloat prediction, Matrix4x4 currentRoot, Vector3[] positions, Vector3[] rights, Vector3[] ups, Vector3[] velocities, int boneCount)
    {
        for (int i = 0; i < boneCount; i++)
        {
            Vector3 position = new Vector3(prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 0 + 0], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 0 + 1], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 0 + 2]).PositionFrom(currentRoot);
            Vector3 right = new Vector3(prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 1 + 0], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 1 + 1], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 1 + 2]).normalized.DirectionFrom(currentRoot);
            Vector3 up = new Vector3(prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 2 + 0], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 2 + 1], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 2 + 2]).normalized.DirectionFrom(currentRoot);
            Vector3 velocity = new Vector3(prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 3 + 0], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 3 + 1], prediction[PostureOutputOffset + i * 3 + boneCount * 3 * 3 + 2]).DirectionFrom(currentRoot);

            positions[i] = Vector3.Lerp(positions[i] + velocity / Framerate, position, 0.5f);
            rights[i] = right;
            ups[i] = up;
            velocities[i] = velocity;
        }
    }

    public static void ApplyPosture(Transform rootTransform, Actor actor, Matrix4x4 nextRoot, Vector3[] positions, Vector3[] rights, Vector3[] ups, Vector3[] velocities, bool stabilizeQuaternionSign)
    {
        rootTransform.position = nextRoot.GetPosition();
        rootTransform.rotation = nextRoot.GetRotation();

        for (int i = 0; i < actor.Bones.Length; i++)
        {
            actor.Bones[i].SetPosition(positions[i]);
            actor.Bones[i].SetVelocity(velocities[i]);

            Vector3 forward = Vector3.Cross(rights[i], ups[i]).normalized;
            Quaternion rotation = Quaternion.LookRotation(forward, ups[i]).normalized;
            if (stabilizeQuaternionSign && rotation.w < 0f)
            {
                rotation = new Quaternion(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
            }

            actor.Bones[i].SetRotation(rotation);
        }
    }

    public static void RecordExtremityHistory(Actor actor, bool recordHands, List<Vector3>[] handHistory, bool recordFeet, List<Vector3>[] footHistory)
    {
        if (recordHands)
        {
            if (actor.Bones.Length > LeftHandBoneIndex)
            {
                UpdateHistory(handHistory, 0, actor.Bones[LeftHandBoneIndex].GetTransform().position);
            }

            if (actor.Bones.Length > RightHandBoneIndex)
            {
                UpdateHistory(handHistory, 1, actor.Bones[RightHandBoneIndex].GetTransform().position);
            }
        }

        if (recordFeet)
        {
            if (actor.Bones.Length > LeftFootBoneIndex)
            {
                UpdateHistory(footHistory, 0, actor.Bones[LeftFootBoneIndex].GetTransform().position);
            }

            if (actor.Bones.Length > RightFootBoneIndex)
            {
                UpdateHistory(footHistory, 1, actor.Bones[RightFootBoneIndex].GetTransform().position);
            }
        }
    }

    public static void DrawDebug(Trajectory trajectory, Vector3 targetDirection, Vector3 targetVelocity, bool showTrajectory, bool showVelocities, Actor actor, Vector3[] velocities)
    {
        if (showTrajectory)
        {
            UltiDraw.Begin();
            UltiDraw.DrawLine(trajectory.Points[RootPointIndex].GetPosition(), trajectory.Points[RootPointIndex].GetPosition() + targetDirection, 0.05f, 0f, UltiDraw.Red.Opacity(0.75f));
            UltiDraw.DrawLine(trajectory.Points[RootPointIndex].GetPosition(), trajectory.Points[RootPointIndex].GetPosition() + targetVelocity, 0.05f, 0f, UltiDraw.Green.Opacity(0.75f));
            UltiDraw.End();
            trajectory.Draw(10);
        }

        if (showVelocities)
        {
            UltiDraw.Begin();
            for (int i = 0; i < actor.Bones.Length; i++)
            {
                UltiDraw.DrawArrow(actor.Bones[i].GetTransform().position, actor.Bones[i].GetTransform().position + velocities[i], 0.75f, 0.0075f, 0.05f, UltiDraw.Purple.Opacity(0.5f));
            }
            UltiDraw.End();
        }
    }

    public static void DrawExtremityTrails(Actor actor, bool showHandTrajectory, List<Vector3>[] handHistory, bool showFootTrajectory, List<Vector3>[] footHistory)
    {
        if (showHandTrajectory)
        {
            UltiDraw.Begin();
            DrawHandTrail(actor, handHistory, 0, LeftHandBoneIndex, LeftHandTrailColor, UltiDraw.Red);
            DrawHandTrail(actor, handHistory, 1, RightHandBoneIndex, RightHandTrailColor, UltiDraw.Blue);
            UltiDraw.End();
        }

        if (showFootTrajectory)
        {
            UltiDraw.Begin();
            DrawFootTrail(actor, footHistory, 0, LeftFootBoneIndex, UltiDraw.Green);
            DrawFootTrail(actor, footHistory, 1, RightFootBoneIndex, UltiDraw.Yellow);
            if (actor.Bones.Length > RightFootBoneIndex)
            {
                DrawFootProjection(actor.Bones[LeftFootBoneIndex].GetTransform().position, UltiDraw.Green);
                DrawFootProjection(actor.Bones[RightFootBoneIndex].GetTransform().position, UltiDraw.Yellow);
            }
            UltiDraw.End();
        }
    }

    private static void DrawHandTrail(Actor actor, List<Vector3>[] history, int historyIndex, int boneIndex, Color trailColor, Color markerColor)
    {
        if (actor.Bones.Length <= boneIndex || history == null || historyIndex >= history.Length || history[historyIndex] == null || history[historyIndex].Count <= 1)
        {
            return;
        }

        for (int i = 1; i < history[historyIndex].Count; i++)
        {
            UltiDraw.DrawLine(history[historyIndex][i - 1], history[historyIndex][i], 0.025f, trailColor);
        }

        UltiDraw.DrawSphere(actor.Bones[boneIndex].GetTransform().position, Quaternion.identity, 0.075f, markerColor);
    }

    private static void DrawFootTrail(Actor actor, List<Vector3>[] history, int historyIndex, int boneIndex, Color color)
    {
        if (actor.Bones.Length <= boneIndex || history == null || historyIndex >= history.Length || history[historyIndex] == null || history[historyIndex].Count <= 1)
        {
            return;
        }

        for (int i = 1; i < history[historyIndex].Count; i++)
        {
            float opacity = 0.2f + 0.8f * ((float)i / history[historyIndex].Count);
            UltiDraw.DrawLine(history[historyIndex][i - 1], history[historyIndex][i], 0.035f, color.Opacity(opacity * 0.95f));
        }

        UltiDraw.DrawSphere(actor.Bones[boneIndex].GetTransform().position, Quaternion.identity, 0.1f, color);
    }

    private static void DrawFootProjection(Vector3 position, Color color)
    {
        Vector3 groundPosition = new Vector3(position.x, 0f, position.z);
        UltiDraw.DrawLine(position, groundPosition, 0.015f, color.Opacity(0.5f));
        UltiDraw.DrawCircle(groundPosition, Quaternion.LookRotation(Vector3.up), 0.05f, color.Opacity(0.25f));
    }
}
