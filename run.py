from app import app
from flask_cors import CORS
CORS(app)  # Enable CORS for all routes

if __name__ == '__main__':
    app.run(debug=True)
