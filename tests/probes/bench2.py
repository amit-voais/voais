"""Human-1 full-duplex benchmark: real-time factor + first-response latency."""
import asyncio, sys, time
import numpy as np, aiohttp, sphn

URI, SR, FRAME = "ws://localhost:8003/api/chat", 24000, 1920  # 80ms


async def main(wav):
    pcm, _ = sphn.read(wav, sample_rate=SR)
    pcm = np.asarray(pcm, np.float32)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=0) if pcm.shape[0] < pcm.shape[1] else pcm.mean(axis=1)

    w, r = sphn.OpusStreamWriter(SR), sphn.OpusStreamReader(SR)
    rx_samples = 0
    t_first = None
    text = []
    lags = []

    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(URI) as ws:
            t0 = time.perf_counter()

            async def recv():
                nonlocal rx_samples, t_first
                async for m in ws:
                    if m.type != aiohttp.WSMsgType.BINARY or not m.data:
                        continue
                    k, p = m.data[0], m.data[1:]
                    if k == 1:
                        got = r.append_bytes(p)
                        if got is not None and len(got):
                            if t_first is None:
                                t_first = time.perf_counter()
                            rx_samples += len(got)
                    elif k == 2:
                        text.append(p.decode(errors="replace"))

            task = asyncio.create_task(recv())

            n_frames = (len(pcm) - FRAME) // FRAME
            for i in range(n_frames):
                target = t0 + i * FRAME / SR       # when this frame SHOULD go
                now = time.perf_counter()
                if target > now:
                    await asyncio.sleep(target - now)
                else:
                    lags.append((now - target) * 1000)   # we fell behind
                b = w.append_pcm(pcm[i * FRAME:(i + 1) * FRAME])
                if b is not None and len(b):
                    await ws.send_bytes(b"\x01" + b)

            t_sent_end = time.perf_counter()
            await asyncio.sleep(3)                  # drain tail
            task.cancel()

    sent_s = n_frames * FRAME / SR
    elapsed = t_sent_end - t0
    rx_s = rx_samples / SR
    print("\n===== HUMAN-1 on L4 =====")
    print(f"  audio sent            : {sent_s:.2f} s over {elapsed:.2f} s wall")
    print(f"  send real-time factor : {sent_s/elapsed:.3f}  (1.000 = perfect)")
    print(f"  1st audio out         : {(t_first-t0)*1000:.0f} ms after connect" if t_first else "  1st audio out: NONE")
    print(f"  audio received        : {rx_s:.2f} s")
    print(f"  rx / sent ratio       : {rx_s/sent_s:.3f}  (~1.0 = keeping up)")
    if lags:
        print(f"  frames behind schedule: {len(lags)}/{n_frames} ({100*len(lags)/n_frames:.0f}%)")
        print(f"  worst lag             : {max(lags):.0f} ms   avg {np.mean(lags):.0f} ms")
    else:
        print(f"  frames behind schedule: 0/{n_frames}  <-- real-time maintained")
    if text:
        print(f"  model text            : {''.join(text)[:300]}")


asyncio.run(main(sys.argv[1]))
