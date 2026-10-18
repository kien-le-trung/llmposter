using LLMposter.App;
using LLMposter.Presentation;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace LLMposter.Editor
{
    /// <summary>
    /// Creates the small baseline scene setup from source-controlled assets. Keeping
    /// this as a menu command also makes it easy to study how editor automation works.
    /// </summary>
    public static class BaselineProjectBuilder
    {
        private const string ProjectRoot = "Assets/_Project";
        private const string BootstrapScenePath = ProjectRoot + "/Scenes/Bootstrap.unity";
        private const string MainScenePath = ProjectRoot + "/Scenes/Main.unity";
        private const string PanelSettingsPath = ProjectRoot + "/UI/MainPanelSettings.asset";
        private const string VisualTreePath = ProjectRoot + "/UI/MainScreen.uxml";

        [MenuItem("Tools/LLMposter/Rebuild Baseline Scenes")]
        public static void Build()
        {
            PanelSettings panelSettings = GetOrCreatePanelSettings();
            CreateBootstrapScene();
            CreateMainScene(panelSettings);
            ConfigureBuildSettings();
            ConfigurePlayerSettings();

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            EditorSceneManager.OpenScene(BootstrapScenePath);

            Debug.Log("LLMposter baseline scenes created successfully.");
        }

        private static PanelSettings GetOrCreatePanelSettings()
        {
            PanelSettings settings = AssetDatabase.LoadAssetAtPath<PanelSettings>(PanelSettingsPath);
            if (settings != null)
            {
                return settings;
            }

            settings = ScriptableObject.CreateInstance<PanelSettings>();
            settings.name = "Main Panel Settings";
            settings.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            settings.referenceResolution = new Vector2Int(1920, 1080);
            settings.match = 0.5f;
            AssetDatabase.CreateAsset(settings, PanelSettingsPath);
            return settings;
        }

        private static void CreateBootstrapScene()
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject appRoot = new GameObject("[App]");
            appRoot.AddComponent<AppBootstrap>();
            EditorSceneManager.SaveScene(scene, BootstrapScenePath);
        }

        private static void CreateMainScene(PanelSettings panelSettings)
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            GameObject cameraObject = new GameObject("Main Camera");
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.055f, 0.071f, 0.114f);
            cameraObject.tag = "MainCamera";

            GameObject uiObject = new GameObject("Main Screen");
            UIDocument document = uiObject.AddComponent<UIDocument>();
            document.panelSettings = panelSettings;
            document.visualTreeAsset = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(VisualTreePath);
            uiObject.AddComponent<MainScreenController>();

            EditorSceneManager.SaveScene(scene, MainScenePath);
        }

        private static void ConfigureBuildSettings()
        {
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(BootstrapScenePath, true),
                new EditorBuildSettingsScene(MainScenePath, true),
            };
        }

        private static void ConfigurePlayerSettings()
        {
            PlayerSettings.productName = "LLMposter";
            PlayerSettings.companyName = "LLMposter";
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
        }
    }
}
