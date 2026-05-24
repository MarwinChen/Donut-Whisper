import whisper
from transformers import WhisperTokenizer


class AudioProcessor(object):
    def __init__(self, n_mels):
        self.n_mels = n_mels

    def __call__(self, audio):
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio.astype(np.float32), n_mels=self.n_mels)
        return mel

whisper_model = whisper.load_model("base")
wav_processor = AudioProcessor(whisper_model.dims.n_mels)
tokenizer_1 = whisper.tokenizer.get_tokenizer(multilingual=False, language="en", task='transcribe')
tokenizer_2 = WhisperTokenizer.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base", multilingual=False, language="en", task='transcribe')
    
text = "Hello, good morning."

label_1 = tokenizer_1.encode(text)
text_1_2 = tokenizer_1.decode(label_1)

label_2 = tokenizer_2.encode(text)
text_2_1 = tokenizer_2.decode(label_2)

print(label_1)
print(text_1_2)

print(label_2)
print(text_2_1)