using System;
using System.Collections.Generic;
using AI4Animation;
using Unity.Sentis;
using UnityEngine;

[RequireComponent(typeof(Actor))]
public class BioAnimation_Kinematic : MonoBehaviour
{
    private static readonly string[] StyleNames = { "Aeroplane", "Zombie", "Akimbo", "March", "Superman", "TwoFootJump" };
    private static readonly int[] StyleIndexMap = { 0, 99, 1, 48, 81, 90 };

    public bool Inspect = false;

    public bool ShowTrajectory = true;
    public bool ShowVelocities = true;
    public bool ShowGUI = true;

    public float TargetGain = 0.25f;
    public float TargetDecay = 0.05f;
    public bool TrajectoryControl = true;
    public float TrajectoryCorrection = 1f;

    public ModelAsset modelPath;
    public BackendType device = BackendType.GPUCompute;
    public int input_dim;
    public TensorShape InputShape;
    public int output_dim;
    public TensorShape OutputShape;

    public Controller Controller;
    public int CurrentStyleIndex;

    public Texture Forward;
    public Texture Left;
    public Texture Right;
    public Texture Back;
    public Texture TurnLeft;
    public Texture TurnRight;
    public Texture Disc;

    public enum PathMode
    {
        ManualControl,
        CirclePath,
        SquarePath
    }

    [Header("Path Settings")]
    public PathMode pathMode = PathMode.ManualControl;
    public float baseWalkingSpeed = 1.5f;

    [Header("Circle Path Settings")]
    public float circleRadius = 3f;
    public Vector3 circleCenter = Vector3.zero;
    public Color circleColor = Color.cyan;

    [Header("Square Path Settings")]
    public float squareSize = 4f;
    public Vector3 squareCenter = Vector3.zero;
    public Color squareColor = Color.magenta;

    public bool ShowHandTrajectory = true;
    public bool ShowFootTrajectory = true;

    private Actor Actor;
    private MANN NN;
    private Trajectory Trajectory;
    private PIDController PID;
    private IPathWalker currentPathWalker;
    private TensorFloat pred;

    private Vector3 TargetDirection;
    private Vector3 TargetVelocity;
    private Vector3[] Positions = new Vector3[0];
    private Vector3[] Rights = new Vector3[0];
    private Vector3[] Ups = new Vector3[0];
    private Vector3[] Velocities = new Vector3[0];
    private List<Vector3>[] HandPositionsHistory = new List<Vector3>[2];
    private List<Vector3>[] FootPositionsHistory = new List<Vector3>[2];
    private float[] InputBuffer = new float[0];
    private float[] StyleBuffer = new float[0];
    private int listIdx;

    void Reset()
    {
        Controller = new Controller();
    }

    private void Awake()
    {
        Actor = GetComponent<Actor>();
        if (Controller == null)
        {
            Controller = new Controller();
        }

        NN = new MANN();
        NN.CreateSession(modelPath, device);
        PID = new PIDController(0.2f, 0.8f, 0f);

        TargetDirection = new Vector3(transform.forward.x, 0f, transform.forward.z);
        TargetVelocity = Vector3.zero;

        BioAnimationDemoUtility.InitializeMotionState(Actor, out Positions, out Rights, out Ups, out Velocities);
        Trajectory = BioAnimationDemoUtility.CreateTrajectory(Controller, transform.position, TargetDirection);
        BioAnimationDemoUtility.InitializeHistory(HandPositionsHistory);
        BioAnimationDemoUtility.InitializeHistory(FootPositionsHistory);

        InitializeBuffers();
        CurrentStyleIndex = StyleIndexMap[0];
    }

    void Start()
    {
        Utility.SetFPS(BioAnimationDemoUtility.Framerate);
        currentPathWalker = BioAnimationDemoUtility.CreatePathWalker((int)pathMode, circleRadius, circleCenter, circleColor, squareSize, squareCenter, squareColor, baseWalkingSpeed);
        currentPathWalker?.Initialize(transform.position);
    }

    void Update()
    {
        HandleStyleSelection();

        if (currentPathWalker != null)
        {
            UpdatePathFollowing();
        }
        else
        {
            BioAnimationDemoUtility.PredictTrajectory(Controller, Trajectory, PID, ref TargetDirection, ref TargetVelocity, ref TrajectoryCorrection, TargetGain, TargetDecay);
        }

        Animate();
    }

    private void Animate()
    {
        Matrix4x4 currentRoot = Trajectory.Points[BioAnimationDemoUtility.RootPointIndex].GetTransformation();

        Array.Clear(InputBuffer, 0, InputBuffer.Length);
        int featureOffset = BioAnimationDemoUtility.EncodeInputFeatures(Trajectory, Actor, Positions, Rights, Ups, Velocities, InputBuffer);

        Array.Clear(StyleBuffer, 0, StyleBuffer.Length);
        if (StyleBuffer.Length > 0)
        {
            int styleIndex = Mathf.Clamp(CurrentStyleIndex, 0, StyleBuffer.Length - 1);
            StyleBuffer[styleIndex] = 1f;
            Array.Copy(StyleBuffer, 0, InputBuffer, featureOffset, StyleBuffer.Length);
        }

        TensorFloat inputFeatures = new TensorFloat(InputShape, InputBuffer);
        pred?.Dispose();
        pred = NN.Inference(inputFeatures);
        inputFeatures.Dispose();
        pred.MakeReadable();

        BioAnimationDemoUtility.ShiftPastTrajectory(Trajectory);
        Matrix4x4 nextRoot = BioAnimationDemoUtility.ApplyPredictionToTrajectory(Trajectory, pred, currentRoot, TrajectoryCorrection);
        BioAnimationDemoUtility.UpdatePostureState(pred, currentRoot, Positions, Rights, Ups, Velocities, Actor.Bones.Length);
        BioAnimationDemoUtility.ApplyPosture(transform, Actor, nextRoot, Positions, Rights, Ups, Velocities, false);
        BioAnimationDemoUtility.RecordExtremityHistory(Actor, ShowHandTrajectory, HandPositionsHistory, ShowFootTrajectory, FootPositionsHistory);
    }

    void OnGUI()
    {
        if (NN == null || !ShowGUI)
        {
            return;
        }

        UltiDraw.Begin();
        UltiDraw.OnGUILabel(new Vector2(0.5f, 0.18f), new Vector2(0.1f, 0.05f), 0.0225f, "Style: " + StyleNames[listIdx], Color.black);

        float baseX = 0.5f;
        float baseY = 0.9f;
        float buttonSize = 0.03f;
        float horizontalSpacing = 0.035f;
        float verticalSpacing = 0.06f;

        UltiDraw.GUITexture(new Vector2(baseX, baseY), buttonSize, Disc, Input.GetKey(Controller.Forward) ? UltiDraw.Orange : UltiDraw.BlackGrey);
        UltiDraw.GUITexture(new Vector2(baseX, baseY), buttonSize, Forward, UltiDraw.White);
        UltiDraw.GUITexture(new Vector2(baseX - horizontalSpacing, baseY), buttonSize, Disc, Input.GetKey(Controller.TurnLeft) ? UltiDraw.Orange : UltiDraw.BlackGrey);
        UltiDraw.GUITexture(new Vector2(baseX - horizontalSpacing, baseY), buttonSize, TurnLeft, UltiDraw.White);
        UltiDraw.GUITexture(new Vector2(baseX + horizontalSpacing, baseY), buttonSize, Disc, Input.GetKey(Controller.TurnRight) ? UltiDraw.Orange : UltiDraw.BlackGrey);
        UltiDraw.GUITexture(new Vector2(baseX + horizontalSpacing, baseY), buttonSize, TurnRight, UltiDraw.White);
        UltiDraw.GUITexture(new Vector2(baseX, baseY + verticalSpacing), buttonSize, Disc, Input.GetKey(Controller.Back) ? UltiDraw.Orange : UltiDraw.BlackGrey);
        UltiDraw.GUITexture(new Vector2(baseX, baseY + verticalSpacing), buttonSize, Back, UltiDraw.White);
        UltiDraw.GUITexture(new Vector2(baseX - horizontalSpacing, baseY + verticalSpacing), buttonSize, Disc, Input.GetKey(Controller.Left) ? UltiDraw.Orange : UltiDraw.BlackGrey);
        UltiDraw.GUITexture(new Vector2(baseX - horizontalSpacing, baseY + verticalSpacing), buttonSize, Left, UltiDraw.White);
        UltiDraw.GUITexture(new Vector2(baseX + horizontalSpacing, baseY + verticalSpacing), buttonSize, Disc, Input.GetKey(Controller.Right) ? UltiDraw.Orange : UltiDraw.BlackGrey);
        UltiDraw.GUITexture(new Vector2(baseX + horizontalSpacing, baseY + verticalSpacing), buttonSize, Right, UltiDraw.White);
        UltiDraw.End();
    }

    private void OnRenderObject()
    {
        if (!Application.isPlaying || NN == null)
        {
            return;
        }

        currentPathWalker?.DrawPath();
        BioAnimationDemoUtility.DrawDebug(Trajectory, TargetDirection, TargetVelocity, ShowTrajectory, ShowVelocities, Actor, Velocities);
        BioAnimationDemoUtility.DrawExtremityTrails(Actor, ShowHandTrajectory, HandPositionsHistory, ShowFootTrajectory, FootPositionsHistory);
    }

    private void OnDisable()
    {
        pred?.Dispose();
        NN?.Dispose();
    }

    private void InitializeBuffers()
    {
        int baseFeatureCount = BioAnimationDemoUtility.GetBaseInputFeatureCount(Actor.Bones.Length);
        if (input_dim < baseFeatureCount)
        {
            throw new InvalidOperationException($"input_dim {input_dim} is smaller than the required base feature count {baseFeatureCount}.");
        }

        InputShape = new TensorShape(1, input_dim);
        OutputShape = new TensorShape(1, output_dim);
        InputBuffer = new float[input_dim];
        StyleBuffer = new float[input_dim - baseFeatureCount];
        pred = TensorFloat.Zeros(OutputShape);
    }

    private void HandleStyleSelection()
    {
        int styleCount = StyleNames.Length;
        if (Input.GetKeyDown(KeyCode.J))
        {
            listIdx = (listIdx - 1 + styleCount) % styleCount;
            CurrentStyleIndex = StyleIndexMap[listIdx];
        }

        if (Input.GetKeyDown(KeyCode.L))
        {
            listIdx = (listIdx + 1) % styleCount;
            CurrentStyleIndex = StyleIndexMap[listIdx];
        }
    }

    private void UpdatePathFollowing()
    {
        currentPathWalker.UpdatePath(Trajectory.Points[BioAnimationDemoUtility.RootPointIndex].GetPosition());
        TargetDirection = Vector3.Lerp(TargetDirection, currentPathWalker.GetTargetDirection(), TargetGain);
        TargetVelocity = Vector3.Lerp(TargetVelocity, currentPathWalker.GetTargetVelocity(), TargetGain);
        TrajectoryCorrection = currentPathWalker.GetTrajectoryCorrection();
        BioAnimationDemoUtility.UpdateFutureTrajectory(Trajectory, TargetDirection, TargetVelocity, TargetGain, true);
    }
}
