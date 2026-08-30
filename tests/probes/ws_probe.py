"""Connect like the browser does and print every message the server sends."""
import asyncio, json, os, time
import numpy as np, soundfile as sf, aiohttp

WAV = os.path.expanduser("~/hindi_sample.wav")

async def main():
    d, sr = sf.read(WAV, dtype="float32")
    if d.ndim > 1: d = d.mean(axis=1)
    if sr != 16000:
        i = np.linspace(0, len(d)-1, int(len(d)*16000/sr))
        d = np.interp(i, np.arange(len(d)), d).astype("float32")
    d = d[:16000*4]                      # 4 s of speech
    tail = np.zeros(16000*2, np.float32) # then 2 s of silence -> turn should end

    async with aiohttp.ClientSession() as s:
        async with s.ws_connect("ws://localhost:7880/ws") as ws:
            t0 = time.perf_counter()
            async def rx():
                async for m in ws:
                    el = (time.perf_counter()-t0)*1000
                    if m.type == aiohttp.WSMsgType.BINARY:
                        print(f"  [{el:7.0f} ms] AUDIO {len(m.data)} bytes")
                    else:
                        print(f"  [{el:7.0f} ms] {m.data[:140]}")
            task = asyncio.create_task(rx())
            for chunk in np.array_split(np.concatenate([d, tail]), 300):
                await ws.send_bytes((chunk*32767).astype(np.int16).tobytes())
                await asyncio.sleep(0.02)
            await asyncio.sleep(12)
            task.cancel()

asyncio.run(main())
