using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LLMposter.App
{
    /// <summary>
    /// The application's composition root. It starts before scene-local behaviour and
    /// survives scene changes so future cross-scene services have one clear owner.
    /// </summary>
    [DefaultExecutionOrder(-1000)]
    public sealed class AppBootstrap : MonoBehaviour
    {
        [SerializeField] private string firstSceneName = "Main";

        public static AppBootstrap Instance { get; private set; }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }

            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        private IEnumerator Start()
        {
            if (SceneManager.GetActiveScene().name != "Bootstrap")
            {
                yield break;
            }

            AsyncOperation loadOperation = SceneManager.LoadSceneAsync(firstSceneName);
            while (loadOperation != null && !loadOperation.isDone)
            {
                yield return null;
            }
        }
    }
}
