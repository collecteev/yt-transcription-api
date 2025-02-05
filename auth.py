from flask import request, jsonify
import os

def require_custom_authentication(func):
    """Middleware for API authentication."""
    def wrapper(*args, **kwargs):
        secret_code = os.getenv("SECRET_CODE")
        auth_header = request.headers.get("Authorization")

        if not auth_header or auth_header.split("Bearer ")[-1] != secret_code:
            return jsonify({"error": "Unauthorized"}), 401

        return func(*args, **kwargs)

    return wrapper
