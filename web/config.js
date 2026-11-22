// Where the Anantara backend lives.
//   ""                            -> same origin (used when the FastAPI server serves the page: local run, HF Space, Render)
//   "https://<slug>.onrender.com" -> when the page is on GitHub Pages and the API is hosted on Render (set it once you have a URL)
// If the backend is not reachable, the page falls back to the cached sample planets.
window.ANANTARA_API = "";
