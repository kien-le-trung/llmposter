using UnityEngine;
using UnityEngine.UIElements;

namespace LLMposter.Presentation
{
    /// <summary>
    /// Connects the Main scene to its UI document. This component owns presentation
    /// behaviour only; application and game state do not belong here.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class MainScreenController : MonoBehaviour
    {
        private Button verifyButton;
        private Label statusLabel;

        private void OnEnable()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            verifyButton = root.Q<Button>("verify-button");
            statusLabel = root.Q<Label>("status-label");

            verifyButton.clicked += HandleVerifyClicked;
        }

        private void OnDisable()
        {
            if (verifyButton != null)
            {
                verifyButton.clicked -= HandleVerifyClicked;
            }
        }

        private void HandleVerifyClicked()
        {
            statusLabel.text = "Event received. The baseline is working.";
            Debug.Log("[MainScreen] UI event received.");
        }
    }
}
