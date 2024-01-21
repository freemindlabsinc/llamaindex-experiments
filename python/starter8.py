import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


static = StaticFiles(directory="static",html=True)
app = FastAPI()

@app.get("/api")
async def root():
 return {"message": "Hello World"}

@app.get("/api/memory/{prompt}")
async def memory_endpoint(prompt: str):
 """When ChatGPT doesn't know the answer to a question, this endpoint will return the most similar answer from the memory bank."""
 logger.info(f"Prompt: {prompt}")
 return f"{prompt} size is {len(prompt)}"


app.add_middleware(
 CORSMiddleware,
 allow_origins=[ "http://localhost:8001",
  "https://chat.openai.com"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

app.mount("/", static, name="static")