using System;
using Unity.Sentis;

[Serializable]
public abstract class SentisModelSession : IDisposable
{
    [NonSerialized] protected Model RuntimeModel;
    [NonSerialized] protected IWorker Worker;

    public bool IsCreated => Worker != null;

    protected void CreateWorkerSession(ModelAsset modelPath, BackendType device)
    {
        if (modelPath == null)
        {
            throw new ArgumentNullException(nameof(modelPath));
        }

        Dispose();

        try
        {
            RuntimeModel = ModelLoader.Load(modelPath);
            Worker = WorkerFactory.CreateWorker(device, RuntimeModel);
            OnSessionCreated(device);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    protected void EnsureSession()
    {
        if (Worker == null)
        {
            throw new InvalidOperationException($"{GetType().Name} session has not been created.");
        }
    }

    protected static T TakeOwnership<T>(T tensor, string outputName = "default output") where T : Tensor
    {
        if (tensor == null)
        {
            throw new InvalidOperationException($"Failed to read {outputName} as {typeof(T).Name}.");
        }

        tensor.TakeOwnership();
        return tensor;
    }

    protected virtual void OnSessionCreated(BackendType device)
    {
    }

    protected virtual void DisposeManagedState()
    {
    }

    public virtual void Dispose()
    {
        DisposeManagedState();

        Worker?.Dispose();
        Worker = null;
        RuntimeModel = null;
    }
}
