# LOF-GOF Predictor UI (Vercel)

Static HTML frontend that calls the Render API. No build step required.

## Local preview
Open `index.html` in a browser and set the API base URL to `http://localhost:8000` (or your Render URL).

## Vercel deploy
1) Create a new Vercel project and point it to `frontend` (Framework: None/Static).  
2) Root directory: `frontend`. Output directory: `frontend`.  
3) Deploy. Update the API base URL input on the page to your Render service (e.g., `https://lof-gof-predictor-api.onrender.com`).
