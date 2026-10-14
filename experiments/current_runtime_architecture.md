# Current Runtime Architecture
*Visualizing the execution flow of the active Multi-Modal Character Consistency Pipeline (MMCCP).*

## System Architecture Diagram

```mermaid
graph TD
    %% Input Layer
    User[User Input / Text] --> SA[Story Analyzer]
    SA --> |Generates| SS[Story Spec]
    SS --> |Registers Characters| CR[Character Registry]

    %% Planning Layer
    SS --> SP[Scene Planner]
    CR --> SP
    SP --> |Outputs| SEP[Scene Execution Plan]
    
    %% Setup & Conditioning Layer
    SEP --> PB[Prompt Builder]
    SEP --> CC[ControlNet Conditioner]
    SEP --> IA[Identity Adapter]
    
    PB -.-> |BREAK tokens| RC[Regional Conditioner]
    IA -.-> |IP-Adapter/FaceID| CC
    
    %% Orchestration & Generation
    CC --> GO[Generation Orchestrator]
    PB --> GO
    
    subgraph "Forge Backend (GPU)"
        GO --> |Pass 1: txt2img| F1[Forge Native U-Net]
        F1 --> |Draft Image| F2[Forge img2img]
        GO --> |Pass 2: adaptive denoise| F2
        Ref[Refinement Controller] --> |ReActor + CodeFormer| F2
    end
    
    F2 --> |Final Render| PA[Programmatic Auditor]
    
    %% QA & Healing Loop
    PA --> |Tests: ArcFace, CLIP, RetinaFace| Eval{QA Passed?}
    Eval -- Yes --> Out[Final Saved Image]
    Eval -- No --> HC[Healing Controller]
    
    HC --> |Adjust Parameters: Enable OpenPose, Boost CFG| GO
```

## Runtime Behavior Summary

1. **Ingestion:** Narrative text is parsed by the `StoryAnalyzer` (using LLMs or heuristics) to construct a `StorySpec` containing defined characters and scenes.
2. **Registration:** `CharacterRegistry` loads and secures canonical character anchors in memory.
3. **Planning:** The `ScenePlanner` reads a `SceneSpec` and determines the camera angle, needed denoising strength (based on shot type), and retrieves correct character anchors. The `PromptBuilder` manages spatial tagging.
4. **Conditioning:** Adapters (`IPAdapterSDXLConditioner`) compile Forge ControlNet payloads.
5. **Generation Pass 1:** The `GenerationOrchestrator` calls Forge's `txt2img` to lay down a structural draft using native modifications (FreeU, PAG, CFG-Fix).
6. **Generation Pass 2:** The orchestrator passes the draft to Forge's `img2img` with the computed adaptive denoising schedule. The `RefinementController` activates ReActor+CodeFormer to sharpen and fix facial consistency.
7. **Quality Audit:** The `ProgrammaticAuditor` intercepts the final image, evaluating face count, identity likeness (ArcFace), and prop existence (CLIP).
8. **Self-Healing:** If the auditor flags a failure, the `HealingController` steps in, dynamically adjusts settings (e.g., turning on OpenPose for spatial overlap failures), and re-triggers generation up to a max retry limit.
