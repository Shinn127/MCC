using System;
using Unity.Sentis;

[Serializable]
public class MANN : SentisModelSession
{
    public void CreateSession(ModelAsset modelPath, BackendType device)
    {
        CreateWorkerSession(modelPath, device);
    }

    public TensorFloat Inference(TensorFloat input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        EnsureSession();

        Worker.Execute(input);
        return TakeOwnership(Worker.PeekOutput() as TensorFloat);
    }
}
