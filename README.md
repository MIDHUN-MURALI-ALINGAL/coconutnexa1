# CocoNexa 🥥

Coconut maturity detection with a YOLO model and a Streamlit front end.
Classes: Young, Semi-mature, Mature, Over mature.

## Project files
- app.py – the Streamlit app (camera/upload, detection, sidebar dashboard, voice output)
- best.pt – your trained YOLO model (copy it here, next to app.py)
- requirements.txt – Python packages
- render.yaml – Render deployment settings
- .streamlit/config.toml – theme and upload size

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Render
1. Push this folder to a GitHub repository, including best.pt.
   Large model files need Git LFS (git lfs track "*.pt").
2. On render.com click New > Web Service and connect the repository.
3. Render reads render.yaml. If you set the fields by hand:
   - Build command: pip install -r requirements.txt
   - Start command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
4. Click Deploy. Camera and voice need HTTPS, which Render provides.

## Notes
- Class names inside best.pt are matched to the four display names automatically
  (for example "semi_mature", "Semi-mature" and "semimature" all work).
- Voice output uses the browser's built-in speech. Speech input is not included.
- Render's free plan has limited memory. PyTorch with ultralytics may need a paid instance.

## Demo mode
If best.pt is missing, the app still runs. Camera, upload, dashboard and voice all work,
but results are placeholders and are marked as DEMO. Add best.pt to turn on real detection.
