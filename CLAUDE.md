# Rules for Claude sessions

- Read `PROJECT_UNDERSTANDING.md`, then `startup.md`.
- **Gemini image generation always uses the Batch API** (`scripts/generate_images_gemini.py --batch submit|status|fetch`), even for a few images.
- API keys (HF, Roboflow, Gemini) are attached by the egress proxy, not stored in the container. If a script demands one, pass a dummy (`GEMINI_API_KEY=placeholder`); the proxy replaces it.
