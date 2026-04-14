# Waste-AI

Waste-AI is an AI-powered waste classification web app that helps users identify waste type from images, track prediction history, view area-based pickup schedules, and report illegal dumping.

## Features

- Upload waste images and get instant AI classification (`general`, `hazardous`, `organic`, `recyclable`)
- User authentication (register/login)
- Prediction history per user
- User settings with area selection
- Area-based garbage pickup schedule
- Dump reporting with image and geo-location
- Browser frontend served by Flask

## Project Structure

```text
waste-ai/
├── frontend/                # HTML/CSS/JS UI pages
│   ├── index.html
│   ├── home.html
│   ├── upload.html
│   ├── history.html
│   ├── report.html
│   ├── settings.html
│   └── ...
├── scripts/
│   ├── app.py               # Main Flask API + frontend serving
│   ├── train.py             # Model training script
│   ├── preprocess_dataset.py
│   ├── clean_dataset.py
│   └── demo_predict.py
├── models/
│   └── waste_classifier.pth # Trained model weights
└── data/
    ├── raw/
    └── reports/             # Uploaded report images
```

## Tech Stack

- Backend: Flask, Flask-CORS
- ML: PyTorch, torchvision, Pillow
- Database: MySQL
- Frontend: HTML, CSS, JavaScript

## Prerequisites

- Python 3.9+
- MySQL server running locally
- Trained model file at `models/waste_classifier.pth`

## Setup

1. Clone the repo and enter the project directory:
   ```bash
   git clone <your-repo-url>
   cd waste-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install flask flask-cors pillow torch torchvision mysql-connector-python requests werkzeug
   ```

4. Create MySQL database and required tables:
   - `users`
   - `predictions`
   - `area_schedules`
   - `reports`

5. Place trained model at:
   ```text
   models/waste_classifier.pth
   ```

6. Update DB credentials in `scripts/app.py` (`get_db()`).

## Run the App

```bash
python scripts/app.py
```

Server starts at `http://127.0.0.1:5001`.

## Main API Endpoints

- `POST /register` - Register user
- `POST /login` - Login user
- `POST /predict` - Classify waste from file/URL/base64
- `GET /history/<user_id>` - Fetch user prediction history
- `GET /areas` - List available areas
- `GET|POST /settings/<user_id>` - Get/update user settings
- `GET /schedule/<user_id>` - Get pickup schedule for user area
- `POST /report-dump` - Submit dump report (image + location)

## Notes

- Frontend pages are served by Flask from the `frontend/` folder.
- CORS is enabled for local development.
- For production, move DB credentials to environment variables.

## Future Improvements

- Add `requirements.txt` for one-command dependency setup
- Add `.env` support and move DB credentials out of source code
- Add SQL schema/migration files for easy database setup
- Improve frontend responsiveness and UI consistency across all pages (especially `about.html`)
- Add model confidence score in prediction response
- Add drag-and-drop upload and camera capture support
- Add admin dashboard for reports and analytics
- Add unit and integration tests for backend routes
- Dockerize app for easier deployment
- Deploy full stack (frontend + backend + DB) on cloud

## License

For academic/demo use. Add a license file if you plan public distribution.
