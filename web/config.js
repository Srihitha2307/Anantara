// Where the Anantara backend lives.
//   ""                            -> same origin (used when the FastAPI server serves the page: local run, HF Space, Render)
//   "https://anantara-ai.onrender.com" -> live Render backend (also used by GitHub Pages)
// If the backend is not reachable, the page falls back to the cached sample planets.
window.ANANTARA_API = "https://anantara-ai.onrender.com";
