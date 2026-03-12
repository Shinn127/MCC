using System;
using System.Collections.Generic;
using AI4Animation;
using Unity.Sentis;
using UnityEngine;

[RequireComponent(typeof(Actor))]
public class BioAnimation_CAMDM : MonoBehaviour
{
    public bool ShowTrajectory = true;
    public bool ShowVelocities = true;

    public float TargetGain = 0.25f;
    public float TargetDecay = 0.05f;
    public bool TrajectoryControl = true;
    public float TrajectoryCorrection = 1f;

    public Controller Controller;

    public TextAsset clipsDataAsset;
    public bool UseClipConditioning = false;
    public int ClipConditionRowIndex;
    public float[,] clipsArray;

    public ModelAsset modelPath;
    public BackendType device = BackendType.GPUCompute;
    public TextAsset modelConfig;
    public DiffusionNetwork diffusionNet;
    public int past_frame_dim;
    public TensorShape PastFrameShape;
    public int output_dim;
    public TensorShape OutputShape;

    public bool ShowHandTrajectory = true;
    public bool ShowFootTrajectory = true;
    public int StyleIndex = 7;

    private Actor Actor;
    private Trajectory Trajectory;
    private PIDController PID;
    private TensorFloat pred;
    private TensorInt cond;
    private TensorFloat ClipConditionTensor;

    private Vector3 TargetDirection;
    private Vector3 TargetVelocity;
    private Vector3[] Positions = new Vector3[0];
    private Vector3[] Rights = new Vector3[0];
    private Vector3[] Ups = new Vector3[0];
    private Vector3[] Velocities = new Vector3[0];
    private List<Vector3>[] HandPositionsHistory = new List<Vector3>[2];
    private List<Vector3>[] FootPositionsHistory = new List<Vector3>[2];
    private float[] PastFrameBuffer = new float[0];
    private float[] ClipConditionBuffer = new float[0];
    private int LoadedClipRowIndex = -1;

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

        PID = new PIDController(0.2f, 0.8f, 0f);
        TargetDirection = new Vector3(transform.forward.x, 0f, transform.forward.z);
        TargetVelocity = Vector3.zero;

        BioAnimationDemoUtility.InitializeMotionState(Actor, out Positions, out Rights, out Ups, out Velocities);
        Trajectory = BioAnimationDemoUtility.CreateTrajectory(Controller, transform.position, TargetDirection);
        BioAnimationDemoUtility.InitializeHistory(HandPositionsHistory);
        BioAnimationDemoUtility.InitializeHistory(FootPositionsHistory);

        InitializeBuffers();

        diffusionNet = new DiffusionNetwork();
        diffusionNet.CreateSession(modelPath, device, modelConfig, OutputShape, false);

        if (UseClipConditioning)
        {
            LoadClipData();
        }
    }

    void Start()
    {
        Utility.SetFPS(BioAnimationDemoUtility.Framerate);
    }

    void Update()
    {
        BioAnimationDemoUtility.PredictTrajectory(Controller, Trajectory, PID, ref TargetDirection, ref TargetVelocity, ref TrajectoryCorrection, TargetGain, TargetDecay);
        Animate();
    }

    private void Animate()
    {
        Matrix4x4 currentRoot = Trajectory.Points[BioAnimationDemoUtility.RootPointIndex].GetTransformation();

        Array.Clear(PastFrameBuffer, 0, PastFrameBuffer.Length);
        BioAnimationDemoUtility.EncodeInputFeatures(Trajectory, Actor, Positions, Rights, Ups, Velocities, PastFrameBuffer);

        TensorFloat pastFrame = new TensorFloat(PastFrameShape, PastFrameBuffer);
        pred?.Dispose();
        if (TryGetClipCondition(out TensorFloat clipCondition))
        {
            pred = diffusionNet.InferenceCLIP(pastFrame, clipCondition);
        }
        else
        {
            cond[0] = StyleIndex;
            pred = diffusionNet.Inference(pastFrame, cond);
        }
        pastFrame.Dispose();
        pred.MakeReadable();

        BioAnimationDemoUtility.ShiftPastTrajectory(Trajectory);
        Matrix4x4 nextRoot = BioAnimationDemoUtility.ApplyPredictionToTrajectory(Trajectory, pred, currentRoot, TrajectoryCorrection);
        BioAnimationDemoUtility.UpdatePostureState(pred, currentRoot, Positions, Rights, Ups, Velocities, Actor.Bones.Length);
        BioAnimationDemoUtility.ApplyPosture(transform, Actor, nextRoot, Positions, Rights, Ups, Velocities, true);
        BioAnimationDemoUtility.RecordExtremityHistory(Actor, ShowHandTrajectory, HandPositionsHistory, ShowFootTrajectory, FootPositionsHistory);
    }

    private void OnRenderObject()
    {
        if (!Application.isPlaying || diffusionNet == null)
        {
            return;
        }

        BioAnimationDemoUtility.DrawDebug(Trajectory, TargetDirection, TargetVelocity, ShowTrajectory, ShowVelocities, Actor, Velocities);
        BioAnimationDemoUtility.DrawExtremityTrails(Actor, ShowHandTrajectory, HandPositionsHistory, ShowFootTrajectory, FootPositionsHistory);
    }

    private void OnDisable()
    {
        diffusionNet?.Dispose();
        pred?.Dispose();
        cond?.Dispose();
        ClipConditionTensor?.Dispose();
    }

    private void InitializeBuffers()
    {
        int baseFeatureCount = BioAnimationDemoUtility.GetBaseInputFeatureCount(Actor.Bones.Length);
        if (past_frame_dim < baseFeatureCount)
        {
            throw new InvalidOperationException($"past_frame_dim {past_frame_dim} is smaller than the required base feature count {baseFeatureCount}.");
        }

        PastFrameShape = new TensorShape(1, past_frame_dim);
        OutputShape = new TensorShape(1, output_dim);
        PastFrameBuffer = new float[past_frame_dim];
        ClipConditionBuffer = new float[BioAnimationDemoUtility.ClipConditionDimension];
        pred = TensorFloat.Zeros(OutputShape);
        cond = TensorInt.Zeros(new TensorShape(1));
        cond[0] = StyleIndex;
    }

    private void LoadClipData()
    {
        if (clipsDataAsset == null)
        {
            clipsArray = null;
            return;
        }

        byte[] byteData = clipsDataAsset.bytes;
        float[] flatArray = new float[byteData.Length / sizeof(float)];
        Buffer.BlockCopy(byteData, 0, flatArray, 0, byteData.Length);

        int rowCount = flatArray.Length / BioAnimationDemoUtility.ClipConditionDimension;
        if (rowCount == 0)
        {
            clipsArray = null;
            return;
        }

        clipsArray = new float[rowCount, BioAnimationDemoUtility.ClipConditionDimension];
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < BioAnimationDemoUtility.ClipConditionDimension; j++)
            {
                clipsArray[i, j] = flatArray[i * BioAnimationDemoUtility.ClipConditionDimension + j];
            }
        }
    }

    private bool TryGetClipCondition(out TensorFloat clipCondition)
    {
        clipCondition = null;
        if (!UseClipConditioning)
        {
            return false;
        }

        if (clipsArray == null)
        {
            LoadClipData();
            if (clipsArray == null)
            {
                return false;
            }
        }

        int rowCount = clipsArray.GetLength(0);
        if (rowCount == 0)
        {
            return false;
        }

        int rowIndex = Mathf.Clamp(ClipConditionRowIndex, 0, rowCount - 1);
        if (ClipConditionTensor == null || LoadedClipRowIndex != rowIndex)
        {
            ClipConditionTensor?.Dispose();
            for (int i = 0; i < BioAnimationDemoUtility.ClipConditionDimension; i++)
            {
                ClipConditionBuffer[i] = clipsArray[rowIndex, i];
            }

            ClipConditionTensor = new TensorFloat(new TensorShape(1, BioAnimationDemoUtility.ClipConditionDimension), ClipConditionBuffer);
            LoadedClipRowIndex = rowIndex;
        }

        clipCondition = ClipConditionTensor;
        return true;
    }
}
