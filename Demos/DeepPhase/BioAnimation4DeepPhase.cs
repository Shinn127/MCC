using System;
using System.Collections.Generic;
using System.IO;
using AI4Animation;
using Unity.Sentis;
using UnityEngine;

[RequireComponent(typeof(Actor))]
public class BioAnimation4DeepPhase : MonoBehaviour
{
    public bool Inspect = false;

    public bool ShowTrajectory = true;
    public bool ShowVelocities = true;

    public float TargetGain = 0.25f;
    public float TargetDecay = 0.05f;
    public bool TrajectoryControl = true;
    public float TrajectoryCorrection = 1f;

    public int input_dim;
    public int output_dim;
    public int main_dim;
    public int phase_dim;
    public int style_dim;
    public int style = 0;

    public ModelAsset modelPath;
    public BackendType device = BackendType.GPUCompute;
    public TensorShape InputShape;
    public TensorShape OutputShape;

    public Controller Controller;

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
    private float[] Phase = new float[0];
    private float[] Style = new float[0];
    private float[] InputBuffer = new float[0];
    private List<Vector3>[] HandPositionsHistory = new List<Vector3>[2];
    private List<Vector3>[] FootPositionsHistory = new List<Vector3>[2];

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

        Phase = new float[phase_dim];
        Style = new float[style_dim];
        LoadInitialPhase();
        InitializeBuffers();
    }

    void Start()
    {
        Utility.SetFPS(BioAnimationDemoUtility.Framerate);
        currentPathWalker = BioAnimationDemoUtility.CreatePathWalker((int)pathMode, circleRadius, circleCenter, circleColor, squareSize, squareCenter, squareColor, baseWalkingSpeed);
        currentPathWalker?.Initialize(transform.position);
    }

    void Update()
    {
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

        Array.Copy(Phase, 0, InputBuffer, featureOffset, Phase.Length);
        Array.Clear(Style, 0, Style.Length);
        if (Style.Length > 0)
        {
            Style[Mathf.Clamp(style, 0, Style.Length - 1)] = 1f;
        }
        Array.Copy(Style, 0, InputBuffer, featureOffset + Phase.Length, Style.Length);

        TensorFloat inputFeatures = new TensorFloat(InputShape, InputBuffer);
        pred?.Dispose();
        pred = NN.Inference(inputFeatures);
        inputFeatures.Dispose();
        pred.MakeReadable();

        BioAnimationDemoUtility.ShiftPastTrajectory(Trajectory);
        Matrix4x4 nextRoot = BioAnimationDemoUtility.ApplyPredictionToTrajectory(Trajectory, pred, currentRoot, TrajectoryCorrection);
        BioAnimationDemoUtility.UpdatePostureState(pred, currentRoot, Positions, Rights, Ups, Velocities, Actor.Bones.Length);
        UpdatePhaseState();
        BioAnimationDemoUtility.ApplyPosture(transform, Actor, nextRoot, Positions, Rights, Ups, Velocities, false);
        BioAnimationDemoUtility.RecordExtremityHistory(Actor, ShowHandTrajectory, HandPositionsHistory, ShowFootTrajectory, FootPositionsHistory);
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
        int requiredInputCount = baseFeatureCount + phase_dim + style_dim;
        if (input_dim < requiredInputCount)
        {
            throw new InvalidOperationException($"input_dim {input_dim} is smaller than the required feature count {requiredInputCount}.");
        }

        main_dim = baseFeatureCount;
        InputShape = new TensorShape(1, input_dim);
        OutputShape = new TensorShape(1, output_dim);
        InputBuffer = new float[input_dim];
        pred = TensorFloat.Zeros(OutputShape);
    }

    private void LoadInitialPhase()
    {
        string filePath = "Assets/Demos/DeepPhase/init_phase.json";
        string jsonContent = File.ReadAllText(filePath);
        init_phase phaseData = JsonUtility.FromJson<init_phase>(jsonContent);
        if (phaseData?.phase == null)
        {
            return;
        }

        int length = Mathf.Min(Phase.Length, phaseData.phase.Length);
        Array.Copy(phaseData.phase, 0, Phase, 0, length);
    }

    private void UpdatePhaseState()
    {
        int phaseOffset = BioAnimationDemoUtility.GetPhaseOutputOffset(Actor.Bones.Length);
        for (int i = 0; i < Phase.Length && phaseOffset + i < output_dim; i++)
        {
            Phase[i] = pred[phaseOffset + i];
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

    [Serializable]
    public class init_phase
    {
        public float[] phase;
    }
}
