import asyncio, edge_tts, os
async def synth(text: str, voice: str, rate: str, out_wav: str):
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    c = edge_tts.Communicate(text, voice=voice, rate=rate)
    wav = b''
    async for ch in c.stream():
        if ch['type']=='audio':
            wav += ch['data']
    with open(out_wav,'wb') as f: f.write(wav)
    return out_wav
def say(text: str, voice: str, rate: str, out_wav: str):
    asyncio.run(synth(text, voice, rate, out_wav))
