using System;
using Unity.Sentis;
using UnityEngine;

[Serializable]
public class DiffusionParams
{
    public int diffusion_steps;
    public float[] coef1 = new float[0];
    public float[] coef2 = new float[0];
    public float[] var = new float[0];
    public float[] simple_var = new float[0];

    public void Validate()
    {
        if (diffusion_steps <= 0)
        {
            throw new InvalidOperationException("diffusion_steps must be greater than zero.");
        }

        if (coef1 == null || coef1.Length != diffusion_steps)
        {
            throw new InvalidOperationException("coef1 length must match diffusion_steps.");
        }

        if (coef2 == null || coef2.Length != diffusion_steps)
        {
            throw new InvalidOperationException("coef2 length must match diffusion_steps.");
        }

        if (var == null || var.Length != diffusion_steps)
        {
            throw new InvalidOperationException("var length must match diffusion_steps.");
        }

        if (simple_var == null || simple_var.Length != diffusion_steps)
        {
            throw new InvalidOperationException("simple_var length must match diffusion_steps.");
        }
    }
}

[Serializable]
public class DiffusionNet : SentisModelSession
{
    private const string LatentInputName = "x";
    private const string FeaturesInputName = "xf";
    private const string ConditionInputName = "cond";
    private const string TimestepInputName = "t";

    [SerializeField] private DiffusionParams diffusionParams = new DiffusionParams();

    [NonSerialized] private ITensorAllocator allocator;
    [NonSerialized] private Ops ops;
    [NonSerialized] private TensorInt[] timeSteps;
    [NonSerialized] private TensorShape outputShape;
    [NonSerialized] private float[] noiseScales;

    private bool useSimpleVariance;

    public DiffusionParams DiffusionParameters => diffusionParams;

    public void CreateSession(ModelAsset modelPath, BackendType device, TextAsset configPath, TensorShape outputShape, bool useSimpleVariance)
    {
        if (configPath == null)
        {
            throw new ArgumentNullException(nameof(configPath));
        }

        CreateWorkerSession(modelPath, device);

        try
        {
            this.outputShape = outputShape;
            this.useSimpleVariance = useSimpleVariance;
            diffusionParams = JsonUtility.FromJson<DiffusionParams>(configPath.text) ?? new DiffusionParams();
            diffusionParams.Validate();

            timeSteps = new TensorInt[diffusionParams.diffusion_steps];
            noiseScales = new float[diffusionParams.diffusion_steps];

            for (int i = 0; i < diffusionParams.diffusion_steps; i++)
            {
                timeSteps[i] = new TensorInt(new TensorShape(1), new[] { i });
                float variance = this.useSimpleVariance ? diffusionParams.simple_var[i] : diffusionParams.var[i];
                noiseScales[i] = Mathf.Exp(0.5f * variance);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public void createSession(ModelAsset modelPath, BackendType device, TextAsset configPath, TensorShape outputShape, bool useSimpleVariance)
    {
        CreateSession(modelPath, device, configPath, outputShape, useSimpleVariance);
    }

    public TensorFloat Inference(TensorFloat pastFrame, TensorInt condition)
    {
        if (condition == null)
        {
            throw new ArgumentNullException(nameof(condition));
        }

        return RunDiffusion(pastFrame, () => Worker.SetInput(ConditionInputName, condition));
    }

    public TensorFloat InferenceCLIP(TensorFloat pastFrame, TensorFloat condition)
    {
        if (condition == null)
        {
            throw new ArgumentNullException(nameof(condition));
        }

        return RunDiffusion(pastFrame, () => Worker.SetInput(ConditionInputName, condition));
    }

    protected override void OnSessionCreated(BackendType device)
    {
        allocator = new TensorCachingAllocator();
        ops = WorkerFactory.CreateOps(device, allocator);
    }

    protected override void DisposeManagedState()
    {
        if (timeSteps != null)
        {
            for (int i = 0; i < timeSteps.Length; i++)
            {
                timeSteps[i]?.Dispose();
            }
        }

        timeSteps = null;
        noiseScales = null;
        outputShape = default;
        useSimpleVariance = false;

        ops?.Dispose();
        ops = null;

        allocator?.Dispose();
        allocator = null;
    }

    private TensorFloat RunDiffusion(TensorFloat pastFrame, Action setCondition)
    {
        if (pastFrame == null)
        {
            throw new ArgumentNullException(nameof(pastFrame));
        }

        if (setCondition == null)
        {
            throw new ArgumentNullException(nameof(setCondition));
        }

        EnsureSession();

        if (ops == null || timeSteps == null || noiseScales == null)
        {
            throw new InvalidOperationException("Diffusion session is not fully initialized.");
        }

        TensorFloat latent = null;

        try
        {
            latent = ops.RandomNormal(outputShape, 0f, 1f, null);

            for (int step = diffusionParams.diffusion_steps - 1; step >= 0; step--)
            {
                Worker.SetInput(LatentInputName, latent);
                Worker.SetInput(FeaturesInputName, pastFrame);
                setCondition();
                Worker.SetInput(TimestepInputName, timeSteps[step]);
                Worker.Execute();

                TensorFloat denoised = Worker.PeekOutput() as TensorFloat;
                if (denoised == null)
                {
                    throw new InvalidOperationException("Failed to read diffusion output as TensorFloat.");
                }

                TensorFloat weightedLatent = ops.Mul(diffusionParams.coef1[step], latent);
                TensorFloat weightedDenoised = ops.Mul(diffusionParams.coef2[step], denoised);
                TensorFloat mean = ops.Add(weightedLatent, weightedDenoised);

                weightedLatent.Dispose();
                weightedDenoised.Dispose();

                TensorFloat nextLatent;
                if (step > 0)
                {
                    TensorFloat noise = ops.RandomNormal(outputShape, 0f, 1f, null);
                    TensorFloat scaledNoise = ops.Mul(noiseScales[step], noise);
                    nextLatent = ops.Add(mean, scaledNoise);

                    noise.Dispose();
                    scaledNoise.Dispose();
                    mean.Dispose();
                }
                else
                {
                    nextLatent = mean;
                }

                latent.Dispose();
                latent = nextLatent;
            }

            latent.TakeOwnership();
            return latent;
        }
        catch
        {
            latent?.Dispose();
            throw;
        }
    }
}
