import os


class Settings:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev_secret_change_me")
    JWT_ALGORITHM: str = "HS256"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/slot_model.onnx")
    STATIC_DIR: str = os.getenv("STATIC_DIR", "static")
    TEMPLATES_DIR: str = os.getenv("TEMPLATES_DIR", "templates")


settings = Settings()


