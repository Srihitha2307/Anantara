// Where the Anantara backend lives.
//   ""                         -> same origin (used when FastAPI serves this page: local run, Hugging Face Space, Render)
//   "https://YOUR-SPACE.hf.space" -> use this when the page is on GitHub Pages and the API is hosted elsewhere
// If the backend cannot be reached, the page falls back to the cached sample planets.
window.ANANTARA_API = "";
