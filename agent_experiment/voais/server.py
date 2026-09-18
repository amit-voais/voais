"""WebRTC signalling and a browser client for local development.

SmallWebRTC needs no media server - the browser peers directly with this
process - which is what makes it usable on a laptop with no Docker and no GPU.
Swapping this for LiveKit later changes only the transport; the pipeline and the
model services stay as they are.
"""
import asyncio
import os

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pydantic import BaseModel

STATIC = os.path.join(os.path.dirname(__file__), "static")
ICE = ["stun:stun.l.google.com:19302"]


class Offer(BaseModel):
    sdp: str
    type: str
    pc_id: str | None = None
    restart_pc: bool = False


def serve(run_bot, params_factory, host: str = "0.0.0.0", port: int = 7860):
    app = FastAPI(title="VoAIs agent")
    connections: dict[str, SmallWebRTCConnection] = {}

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.post("/api/offer")
    async def offer(body: Offer, background: BackgroundTasks):
        if body.pc_id and body.pc_id in connections:
            conn = connections[body.pc_id]
            await conn.renegotiate(sdp=body.sdp, type=body.type, restart_pc=body.restart_pc)
            return conn.get_answer()

        conn = SmallWebRTCConnection(ICE)
        await conn.initialize(sdp=body.sdp, type=body.type)

        @conn.event_handler("closed")
        async def _closed(c):
            connections.pop(c.pc_id, None)
            logger.info(f"peer {c.pc_id} closed")

        transport = SmallWebRTCTransport(webrtc_connection=conn, params=params_factory())
        background.add_task(run_bot, transport)

        answer = conn.get_answer()
        connections[answer["pc_id"]] = conn
        return answer

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    logger.info(f"open http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
