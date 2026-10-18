# LLMposter Unity Client

This is the baseline Unity client for learning game-application architecture. It
contains no LLMposter rules, backend calls, player models, or round state yet.

## Open and run

1. Open Unity Hub and choose **Add > Add project from disk**.
2. Select this `unity` directory.
3. Use Unity `6000.2.10f1` (the version recorded in `ProjectSettings`).
4. Open `Assets/_Project/Scenes/Bootstrap.unity` if it is not already open.
5. Press Play.
6. Click **Verify UI event** and watch both the status label and Console.

Always enter Play mode through `Bootstrap`. It is scene 0 in the build settings,
so standalone builds will do this automatically.

## What the baseline teaches

### Assets and project settings

- `Assets` contains code, scenes, UI, art, audio, and other authored content.
- `Packages/manifest.json` declares Unity packages.
- `ProjectSettings` contains settings shared by the team.
- `Library`, `Temp`, `Logs`, and `UserSettings` are generated locally and ignored.
- Unity `.meta` files identify assets and must be committed with their assets.

### Scenes and object lifetimes

- `Bootstrap` is the entry scene. Its `[App]` GameObject owns `AppBootstrap`.
- `AppBootstrap` uses `DontDestroyOnLoad`, so it survives scene transitions.
- `Main` contains scene-local objects: a camera and the main UI document.
- Persistent application services and scene-specific presentation therefore have
  visibly different owners.

### GameObjects and components

A GameObject is a container. Components provide its data and behaviour:

- `[App]` + `AppBootstrap` starts the application.
- `Main Camera` + `Camera` defines the rendered view.
- `Main Screen` + `UIDocument` renders the UI tree.
- `Main Screen` + `MainScreenController` binds UI events to C# behaviour.

### Unity lifecycle

- `Awake` establishes `AppBootstrap` before other runtime behaviour.
- `Start` begins loading the first presentation scene.
- `OnEnable` subscribes to the button event.
- `OnDisable` unsubscribes, preventing stale or duplicate callbacks.

### UI Toolkit

- `MainScreen.uxml` describes the visual hierarchy, similar to HTML.
- `MainScreen.uss` styles that hierarchy, similar to CSS.
- `MainScreenController.cs` owns behaviour, keeping structure, style, and logic
  separate.

## Folder layout

```text
Assets/_Project/
  Editor/     editor-only setup tools
  Runtime/    code included in the game build
  Scenes/     loadable application scenes
  UI/         UI Toolkit layouts, styles, and panel settings
```

The leading underscore keeps first-party project content easy to find above
third-party assets in Unity's Project window.

## Rebuilding the generated setup

The scenes and panel settings are checked in and ready to use. If they are ever
deleted or you want to study their construction, run:

`Tools > LLMposter > Rebuild Baseline Scenes`

The next planned layer is a backend-independent application state model. Network
communication should be added only after that boundary is clear.
