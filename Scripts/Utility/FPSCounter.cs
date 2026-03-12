using UnityEngine;
using System.IO;
using System.Text;
using UnityEditor; // Needed for SceneManager in Editor
using UnityEngine.SceneManagement; // Needed for SceneManager at runtime

public class FPSCounter : MonoBehaviour
{
    public bool logToFile = true;
    public bool logToConsole = true; // New boolean to control console logging
    public string fileName = "fps_log.csv";
    public float updateInterval = 0.25f; // 更新频率（秒）
    public float startDelay = 5f; // 延迟5秒后开始记录

    private float accum = 0;
    private int frames = 0;
    private float timeLeft;
    private float currentFPS;
    private StringBuilder csvContent = new StringBuilder();
    private float startTime;
    private bool isLogging = false;

    void Start()
    {
        timeLeft = updateInterval;
        csvContent.AppendLine("Time, FPS"); // CSV表头
        startTime = Time.time;
    }

    void Update()
    {
        // Check if 5 seconds have passed and we're not logging yet
        if (!isLogging && Time.time - startTime >= startDelay)
        {
            isLogging = true;
            if (logToConsole) Debug.Log("Starting FPS logging after 5 second delay");
        }

        // Only proceed with FPS calculation if we're logging
        if (!isLogging) return;

        timeLeft -= Time.deltaTime;
        accum += Time.timeScale / Time.deltaTime;
        frames++;

        if (timeLeft <= 0)
        {
            currentFPS = accum / frames;
            string logLine = $"{Time.time:F2}, {currentFPS:F2}";

            if (logToConsole) Debug.Log(logLine); // Only log to console if enabled

            if (logToFile)
            {
                csvContent.AppendLine(logLine);
            }

            timeLeft = updateInterval;
            accum = 0;
            frames = 0;
        }
    }

    void OnApplicationQuit()
    {
        if (logToFile && csvContent.Length > 0 && isLogging)
        {
            string scenePath = GetSceneDirectory();
            if (!string.IsNullOrEmpty(scenePath))
            {
                string path = Path.Combine(scenePath, fileName);
                File.WriteAllText(path, csvContent.ToString());
                if (logToConsole) Debug.Log($"FPS数据已保存到: {path}");
            }
            else
            {
                if (logToConsole) Debug.LogWarning("无法获取场景路径，FPS数据将不会被保存");
            }
        }
    }

    private string GetSceneDirectory()
    {
        string scenePath = "";

#if UNITY_EDITOR
        // In Editor, use AssetDatabase to get scene path
        if (UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene().isLoaded)
        {
            scenePath = UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene().path;
            scenePath = Path.GetDirectoryName(scenePath);
            scenePath = Path.Combine(Application.dataPath, "..", scenePath);
            scenePath = Path.GetFullPath(scenePath);
        }
#else
        // At runtime, we can't get the original scene path, so fall back to persistentDataPath
        scenePath = Application.persistentDataPath;
#endif

        return scenePath;
    }
}