import logging

from flask import Flask, jsonify

from app.detect import detect_blueprint
from app.propagate import propagate_blueprint

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.register_blueprint(detect_blueprint)
app.register_blueprint(propagate_blueprint)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800, debug=False)
