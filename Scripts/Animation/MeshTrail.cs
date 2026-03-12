using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshTrail : MonoBehaviour
{
    public bool trailEnabled = true; // Added boolean to enable/disable trail
    public bool enableMeshTransparency = true; // New variable to control mesh transparency

    public float activeTime = 5f;
    public float meshRefreshRate = 0.1f;
    public float meshDestroyDelay = 7f;
    public Transform positionToSpawn;
    public Material mat;

    private bool isTrailActive;
    private SkinnedMeshRenderer[] skinnedMeshRenderers;
    private List<GameObject> activeMeshObjects = new List<GameObject>();

    void Update()
    {
        if (!trailEnabled) // Check if trail is disabled
        {
            // If we disable the trail while it's active, clean up any existing trail objects
            if (isTrailActive)
            {
                StopAllCoroutines();
                CleanUpTrail();
                isTrailActive = false;
            }
            return;
        }

        if (!isTrailActive)
        {
            isTrailActive = true;
            StartCoroutine(ActivateTrail(activeTime));
        }
    }

    IEnumerator ActivateTrail(float timeActive)
    {
        while (timeActive > 0 && trailEnabled) // Added trailEnabled check
        {
            timeActive -= meshRefreshRate;

            if (skinnedMeshRenderers == null)
            {
                skinnedMeshRenderers = GetComponentsInChildren<SkinnedMeshRenderer>();
            }

            for (int i = 0; i < skinnedMeshRenderers.Length; i++)
            {
                GameObject gObj = new GameObject();
                gObj.transform.SetPositionAndRotation(positionToSpawn.position, positionToSpawn.rotation);

                MeshRenderer mr = gObj.AddComponent<MeshRenderer>();
                MeshFilter mf = gObj.AddComponent<MeshFilter>();

                Mesh mesh = new Mesh();
                skinnedMeshRenderers[i].BakeMesh(mesh);

                mf.mesh = mesh;

                // Create new material instance for independent alpha control
                Material instanceMat = new Material(mat);
                mr.material = instanceMat;

                activeMeshObjects.Add(gObj);

                // Only start fade coroutine if transparency is enabled
                if (enableMeshTransparency)
                {
                    StartCoroutine(FadeMeshObject(gObj, instanceMat));
                }

                Destroy(gObj, meshDestroyDelay);
            }

            yield return new WaitForSeconds(meshRefreshRate);
        }

        isTrailActive = false;
    }

    IEnumerator FadeMeshObject(GameObject meshObject, Material material)
    {
        float fadeDuration = meshDestroyDelay;
        float elapsedTime = 0f;
        Color originalColor = material.color;

        while (elapsedTime < fadeDuration && meshObject != null)
        {
            elapsedTime += Time.deltaTime;

            // Use cosine function for interpolation
            float t = Mathf.Clamp01(elapsedTime / fadeDuration);
            float alpha = Mathf.Cos(t * Mathf.PI * 0.5f); // Smooth transition from 1 to 0

            material.color = new Color(originalColor.r, originalColor.g, originalColor.b, alpha);
            yield return null;
        }

        // Remove from list when object is destroyed
        if (meshObject == null && activeMeshObjects.Contains(meshObject))
        {
            activeMeshObjects.Remove(meshObject);
        }
    }

    // Clean up all active trail objects
    private void CleanUpTrail()
    {
        foreach (GameObject obj in activeMeshObjects)
        {
            if (obj != null)
            {
                Destroy(obj);
            }
        }
        activeMeshObjects.Clear();
    }

    // Public method to toggle trail at runtime
    public void SetTrailEnabled(bool enabled)
    {
        trailEnabled = enabled;
    }

    // Public method to toggle mesh transparency at runtime
    public void SetMeshTransparencyEnabled(bool enabled)
    {
        enableMeshTransparency = enabled;
    }
}