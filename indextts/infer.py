import os
import re
import sys

import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.common import tokenize_by_CJK_char
from indextts.vqvae.xtts_dvae import DiscreteVAE

from indextts.utils.front import TextNormalizer
class IndexTTS:
    def __init__(self, cfg_path='checkpoints/config.yaml', model_dir='checkpoints', is_fp16=True):
        self.cfg = OmegaConf.load(cfg_path)
        self.device = 'cuda:0'
        self.model_dir = model_dir
        self.is_fp16 = is_fp16
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        if self.is_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = None
        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        load_checkpoint(self.dvae, self.dvae_path)
        self.dvae = self.dvae.to(self.device)
        if self.is_fp16:
            self.dvae.eval().half()
        else:
            self.dvae.eval()
        print(">> vqvae weights restored from:", self.dvae_path)

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            self.gpt.post_init_gpt2_config(use_deepspeed=True, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

        self.bigvgan = Generator(self.cfg.bigvgan)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location='cpu')
        self.bigvgan.load_state_dict(vocoder_dict['generator'])
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset['bpe_model'])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")

    def preprocess_text(self, text):
        # chinese_punctuation = "，。！？；：“”‘’（）【】《》"
        # english_punctuation = ",.!?;:\"\"''()[]<>"
        #
        # # 创建一个映射字典
        # punctuation_map = str.maketrans(chinese_punctuation, english_punctuation)

        # 使用translate方法替换标点符号
        # return text.translate(punctuation_map)
        return self.normalizer.infer(text)

    def remove_long_silence(self, codes, silent_token=52, max_consecutive=30):
        code_lens = []
        codes_list = []
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if self.cfg.gpt.stop_mel_token not in code:
                code_lens.append(len(code))
                len_ = len(code)
            else:
                # len_ = code.cpu().tolist().index(8193)+1
                len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                len_ = len_ - 2

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                code = code.cpu().tolist()
                ncode = []
                n = 0
                for k in range(0, len_):
                    if code[k] != silent_token:
                        ncode.append(code[k])
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode.append(code[k])
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                len_ = len(ncode)
                ncode = torch.LongTensor(ncode)
                codes_list.append(ncode.cuda())
                isfix = True
                #codes[i] = self.stop_mel_token
                #codes[i, 0:len_] = ncode
            else:
                codes_list.append(codes[i])
            code_lens.append(len_)
        code_lens = torch.LongTensor(code_lens).cuda()
        if isfix:
            codes = pad_sequence(codes_list, batch_first=True)
        return codes, code_lens

    def infer_return_wav(self, audio_prompt, text):
        """
        Generate audio from text and return the waveform directly
        
        Args:
            audio_prompt: Path to the voice sample file
            text: Text to convert to speech
        
        Returns:
            torch.Tensor: Generated audio waveform
        """
        print(f"origin text:{text}")
        text = self.preprocess_text(text)
        print(f"normalized text:{text}")

        audio, sr = torchaudio.load(audio_prompt)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        cond_mel = MelSpectrogramFeatures()(audio).to(self.device)

        auto_conditioning = cond_mel

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(self.bpe_path)

        punctuation = ["!", "?", ".", ";", "！", "？", "。", "；"]
        pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
        sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
        print(sentences)

        wavs = []

        for sent in sentences:
            print(sent)
            cleand_text = tokenize_by_CJK_char(sent)
            print(cleand_text)
            text_tokens = torch.IntTensor(tokenizer.encode(cleand_text)).unsqueeze(0).to(self.device)
            text_tokens = text_tokens.to(self.device)

            with torch.no_grad():
                if self.is_fp16:
                    with torch.cuda.amp.autocast(enabled=self.dtype is not None, dtype=self.dtype):
                        codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                          cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                        device=text_tokens.device),
                                                          do_sample=True,
                                                          top_p=0.8,
                                                          top_k=30,
                                                          temperature=1.0,
                                                          num_return_sequences=1,
                                                          length_penalty=0.0,
                                                          num_beams=3,
                                                          repetition_penalty=10.0,
                                                          max_generate_length=600)

                        codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)

                        latent = self.gpt(auto_conditioning, text_tokens,
                                         torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                         code_lens*self.gpt.mel_length_compression,
                                         cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                         return_latent=True, clip_inputs=False)
                        latent = latent.transpose(1, 2)
                        wav, _ = self.bigvgan(latent.transpose(1, 2), auto_conditioning.transpose(1, 2))
                        wav = wav.squeeze(1).cpu()
                else:
                    # Similar code block for non-FP16 case...
                    pass

                wav = torch.clip(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav)

        wav = torch.cat(wavs, dim=1)
        return wav.type(torch.int16)


if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    #text="晕 XUAN4 是 一 种 GAN3 觉"
    #text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    text="There is a vehicle arriving in dock number 7?"

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True)
    generated_wav = tts.infer_return_wav(audio_prompt=prompt_wav, text=text)
    torchaudio.save("gen.wav", generated_wav, 24000)
