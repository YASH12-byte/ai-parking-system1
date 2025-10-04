import sys
from pathlib import Path
p = Path(__file__).resolve().parent
sys.path.insert(0, str(p))

from src.server import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info', reload=True)
