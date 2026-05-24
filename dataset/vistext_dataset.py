from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
from decord import VideoReader, cpu
import cv2
from transformers import DefaultDataCollator
import random
import whisper

class VistextDataset(Dataset):
    def __init__(self, data_path, image_processor, wav_processor, wav_processor_large):
        self.data_path = data_path
        self.image_processor = image_processor
        self.wav_processor = wav_processor
        self.wav_processor_large = wav_processor_large
        with open(self.data_path, 'r', encoding="utf-8") as fp:
            self.data = json.load(fp)

    def __len__(self):
        return len(self.data)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            length_list.append(len(sample["text"]))
        return length_list

    def __getitem__(self, i):
        try:
            source = self.data[i]

            if "audio" in source:
                audio_file = source["audio"]            
                audio, sr = sf.read(audio_file)
            else:
                raise NotImplementedError(
                    "Remote audio loading is disabled in the public dataset loader. "
                    "Please provide a local 'audio' path in each JSON item."
                )

            if len(audio.shape) >= 2:
                audio = audio[:, 0]

            text = source["text"]
            # text_zh = source["text_zh"]

            if "video" in source or "tos_video" in source:
                if "video" in source:
                    video_file = source["video"]
                    vr = VideoReader(video_file, ctx=cpu(i % 8), num_threads=1)
                else:
                    raise NotImplementedError(
                        "Remote video loading is disabled in the public dataset loader. "
                        "Please provide a local 'video' path in each JSON item."
                    )

                if "image_cnt" in source:
                    num_images = int(source["image_cnt"])
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    width = vr[0].shape[1]

                    flag = False
                    for _ in range(2):
                        words = text.split()
                        lines = []
                        current_line = ""
                        for word in words:
                            test_text = current_line + " " + word if current_line else word
                            text_size, _ = cv2.getTextSize(test_text, font, font_scale, thickness)
                            text_width = text_size[0]
                            if text_width <= width:
                                current_line = test_text
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word
                        if current_line:
                            lines.append(current_line)
                        num_images = len(lines)
                        if num_images > 20:
                            font_scale = 0.5
                            thickness = 1
                        else:
                            flag = True
                            break   
                    
                    try:
                        assert flag                     
                    except Exception as e:
                        print(f"GGG: Num Images: {num_images}, Text: {text}")
                        raise e

                if "timestamps" in source:
                    start, end = source["timestamps"]
                    # audio = audio[int(start * sr): int(end * sr)]

                    total_frame_num = len(vr)
                    ori_fps = vr.get_avg_fps()
                    start = round(start * ori_fps)
                    end = round(end * ori_fps)
                    end = min(end, total_frame_num - 1)

                    frame_idx = np.linspace(start, end, num_images + 2, dtype=int).tolist()[1:-1]
                    images = vr.get_batch(frame_idx).asnumpy()
                    data_id = "['{}', '{}', '{}']".format(audio_file, video_file, source["timestamps"])

                else:
                    total_frame_num = len(vr)
                    frame_idx = np.linspace(0, total_frame_num - 1, num_images + 2, dtype=int).tolist()[1:-1]
                    images = vr.get_batch(frame_idx).asnumpy()
                    data_id = "['{}', '{}', '{}']".format(audio_file, video_file, None)

                images_with_text = []
                if "image_cnt" not in source:
                    for i in range(num_images):
                        new_image = images[i]
                        new_image = new_image / 255 * 2 - 1

                        current_text = lines[i]

                        text_size, baseline = cv2.getTextSize(current_text, font, font_scale, thickness)
                        text_width = text_size[0]
                        text_height = text_size[1]
                        text_x = int((width - text_width) / 2)
                        text_y = new_image.shape[0] - 30

                        padding = 5
                        x1 = max(text_x - padding, 0)
                        y1 = text_y + baseline - text_height - 2 * padding
                        x2 = min(text_x + text_width + padding, width)
                        y2 = text_y + baseline

                        new_image[y1:y2, x1:x2] = -1.0

                        cv2.putText(new_image, current_text, (text_x, text_y), font, font_scale, (1, 1, 1), thickness)

                        new_image = (new_image + 1) / 2 * 255
                        new_image = new_image.astype(np.uint8)
                        images_with_text.append(new_image)

                else:
                    for i in range(num_images):
                        new_image = images[i]
                        images_with_text.append(new_image)

                images = []
                for img in images_with_text:
                    images.append(self.image_processor(img)["pixel_values"][0])
                images = np.stack(images)

            else:
                data_id = "['{}', '{}', '{}']".format(audio_file, None, None)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                width = 1280

                words = text.split()
                lines = []
                current_line = ""
                for word in words:
                    test_text = current_line + " " + word if current_line else word
                    text_size, _ = cv2.getTextSize(test_text, font, font_scale, thickness)
                    text_width = text_size[0]
                    if text_width <= width:
                        current_line = test_text
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                num_images = len(lines)

                images_with_text = []
                for i in range(num_images):
                    new_image = np.zeros((960, 1280, 3), dtype=np.uint8)
                    new_image = new_image / 255 * 2 - 1

                    current_text = lines[i]

                    text_size, baseline = cv2.getTextSize(current_text, font, font_scale, thickness)
                    text_width = text_size[0]
                    text_height = text_size[1]
                    text_x = int((width - text_width) / 2)
                    text_y = new_image.shape[0] - 30

                    padding = 5
                    x1 = max(text_x - padding, 0)
                    y1 = text_y + baseline - text_height - 2 * padding
                    x2 = min(text_x + text_width + padding, width)
                    y2 = text_y + baseline

                    new_image[y1:y2, x1:x2] = -1.0

                    cv2.putText(new_image, current_text, (text_x, text_y), font, font_scale, (1, 1, 1), thickness)

                    new_image = (new_image + 1) / 2 * 255
                    new_image = new_image.astype(np.uint8)
                    images_with_text.append(new_image)

                images = []
                for img in images_with_text:
                    images.append(self.image_processor(img)["pixel_values"][0])
                images = np.stack(images)

            spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
            spectrogram_large = self.wav_processor_large(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()

            batch = dict(
                audios=audio,
                spectrograms=spectrogram,
                spectrograms_large=spectrogram_large,
                images=images,
                texts=text,
                # texts_zh=text_zh,
                data_ids=data_id,
            )
            return batch

        except Exception as e:
            print(f'GGGG {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
            source = self.data[i]
            print(f'Audio path: {source.get("audio")}')
            print(f'Video path: {source.get("video")}')
            if isinstance(e, NotImplementedError):
                raise
            return self.__getitem__(random.choice(range(len(self))))

class VistextDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, tokenizer_zh):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_zh = tokenizer_zh

    def __call__(self, samples):
        audios = [torch.from_numpy(s["audios"]).to(torch.float16) for s in samples]
        spectrograms = [s["spectrograms"] for s in samples]
        spectrograms_large = [s["spectrograms_large"] for s in samples]
        spectrograms = torch.stack(spectrograms).to(torch.float16)
        spectrograms_large = torch.stack(spectrograms_large).to(torch.float16)

        images = [torch.from_numpy(s["images"]) for s in samples]
        images_len = [it.size(0) for it in images]
        images = torch.cat(images, dim=0).to(torch.float16)
        
        texts = [s['texts'] for s in samples]
        # texts_zh = [s['texts_zh'] for s in samples]
        # labels = [torch.tensor(self.tokenizer.encode(t), dtype=torch.int64) for t in texts]
        # labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.eot)
        
        # labels = self.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        labels = [self.tokenizer(s['texts'], return_tensors="pt").input_ids.squeeze() for s in samples]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        # labels = [self.tokenizer_zh(s['texts_zh'], return_tensors="pt").input_ids.squeeze() for s in samples]
        # labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        data_ids = [s["data_ids"] for s in samples]

        batch = {"audios": audios, 
        "spectrograms": spectrograms, 
        "spectrograms_large": spectrograms_large, 
        "images": images, 
        "labels": labels, 
        "texts": texts, 
        "data_ids": data_ids, 
        "images_len": images_len
        }
        return batch

class AudioProcessor(object):
    def __init__(self, n_mels):
        self.n_mels = n_mels

    def __call__(self, audio):
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio.astype(np.float32), n_mels=self.n_mels)
        return mel
