import numpy as np, sounddevice as sd, webrtcvad
def list_input_devices():
    devs = sd.query_devices(); out=[]
    for idx,d in enumerate(devs):
        if int(d.get('max_input_channels',0))>0: out.append((idx,d['name']))
    return out
class VADListener:
    def __init__(self, sample_rate=16000, frame_ms=30, vad_level=2, min_utt_ms=500, max_utt_ms=8000, hang_ms=600, device=None, threshold=0.01):
        self.sr=sample_rate; self.frame_len=int(sample_rate*frame_ms/1000)
        self.min_frames=int(min_utt_ms/frame_ms); self.max_frames=int(max_utt_ms/frame_ms); self.hang_frames=int(hang_ms/frame_ms)
        self.vad=webrtcvad.Vad(vad_level); self.device=device; self.threshold=threshold
    def listen(self, echo_guard=None):
        with sd.RawInputStream(samplerate=self.sr, blocksize=self.frame_len, dtype='int16', channels=1, device=self.device) as stream:
            voiced=0; hang=0; utt=[]
            while True:
                buf,_ = stream.read(self.frame_len)
                frame = bytes(buf)
                pcm16 = np.frombuffer(frame, dtype=np.int16)
                pcmf = pcm16.astype(np.float32)/32768.0
                if echo_guard and echo_guard(): continue
                rms = np.sqrt(np.mean(pcmf*pcmf)+1e-12)
                if rms < self.threshold:
                    if voiced>0:
                        hang += 1
                        if hang>=self.hang_frames and len(utt)>0:
                            audio=np.frombuffer(b''.join(utt),dtype=np.int16).astype(np.float32)/32768.0
                            yield audio; voiced=0; hang=0; utt=[]
                    continue
                sp = self.vad.is_speech(frame, self.sr)
                if sp:
                    voiced+=1; utt.append(frame)
                    if voiced>=self.max_frames:
                        audio=np.frombuffer(b''.join(utt),dtype=np.int16).astype(np.float32)/32768.0
                        yield audio; voiced=0; hang=0; utt=[]
                else:
                    if voiced>=self.min_frames:
                        hang+=1
                        if hang>=self.hang_frames:
                            audio=np.frombuffer(b''.join(utt),dtype=np.int16).astype(np.float32)/32768.0
                            yield audio; voiced=0; hang=0; utt=[]
                    else:
                        voiced=0; hang=0; utt=[]
