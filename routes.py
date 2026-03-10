from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from tokens import generate_tokens
from frontend import INDEX_HTML

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list[dict]


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        generate_tokens(req.messages),
        media_type="text/event-stream",
    )


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)
