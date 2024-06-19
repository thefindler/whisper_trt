import json
import torch
import tensorrt_llm
from typing import Dict, Iterable, Optional

from pathlib import Path
from collections import OrderedDict
import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F

from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)

    def get_session(self, engine_dir):
        config_path = engine_dir / 'encoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        dtype = config['builder_config']['precision']
        n_mels = config['builder_config']['n_mels']
        num_languages = config['builder_config']['num_languages']

        self.dtype = dtype
        self.n_mels = n_mels
        self.num_languages = num_languages

        serialize_path = engine_dir / f'encoder.engine'

        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())

        return session

    def get_audio_features(self, mel):

        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)

        inputs = OrderedDict()
        inputs['x'] = mel
        inputs['input_lengths'] = input_lengths

        output_list = [
            TensorInfo('x', str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       input_lengths.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        audio_features = outputs['output']
        return audio_features
        

class WhisperDecoding:

    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_config(self, engine_dir):
        config_path = engine_dir / 'decoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config['plugin_config'])
        decoder_config.update(config['builder_config'])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        dtype = self.decoder_config['precision']
        serialize_path = engine_dir / f'decoder.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            num_heads=self.decoder_config['num_heads'],
            num_kv_heads=self.decoder_config['num_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            num_layers=self.decoder_config['num_layers'],
            gpt_attention_plugin=self.decoder_config['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['remove_input_padding'],
            cross_attention=self.decoder_config['cross_attention'],
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            has_token_type_embedding=self.
            decoder_config['has_token_type_embedding'],
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 sampling_config):
        
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device='cuda')

        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            sampling_config.max_new_tokens,
            beam_width=sampling_config.num_beams,
            encoder_max_input_length=encoder_outputs.shape[1])

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids



class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()



class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

class WhisperTRT:
    def __init__(self, engine_dir, tokenizer, compute_type='float16'):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        with open(engine_dir / 'model_metadata.json', 'r') as f:
            self.model_metadata = json.load(f)
        self.tokenizer = tokenizer
        self.text_decoder = TextDecoder(
            self.model_metadata["n_vocab"],
            self.model_metadata["n_text_ctx"],
            self.model_metadata["n_text_state"],
            self.model_metadata["n_text_head"],
            self.model_metadata["n_text_layer"],
        )
        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir, runtime_mapping)
        self.n_mels = self.encoder.n_mels
        self.is_multilingual = True
        self.compute_type = compute_type
        
    def encode(self, mel):
        return self.encoder.get_audio_features(mel.type(str_dtype_to_torch(self.compute_type)))
    
    @torch.no_grad()
    def detect_language(self, audio_features):
        """
        Detect the spoken language in the audio, and return them as list of strings, along with the ids
        of the most probable language tokens and the probability distribution over all language tokens.
        This is performed outside the main decode loop in order to not interfere with kv-caching.

        Returns
        -------
        language_tokens : Tensor, shape = (n_audio,)
            ids of the most probable language tokens, which appears after the startoftranscript token.
        language_probs : List[Dict[str, float]], length = n_audio
            list of dictionaries containing the probability distribution over all languages.
        """

        single = audio_features.ndim == 2
        if single:
            audio_features = audio_features.unsqueeze(0)

        # forward pass using a single token, startoftranscript
        n_audio = audio_features.shape[0]
        x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(audio_features.device)  # [n_audio, 1]
        logits = self.text_decoder(x, audio_features)[:, 0]

        # collect detected languages; suppress all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(self.tokenizer.all_language_tokens)] = False
        logits[:, mask] = -np.inf
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()
        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
            }
            for i in range(n_audio)
        ]

        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        return language_tokens, language_probs

    # input features is mels.to(device) and encoded features is with audio features
    def generate(self, features, prompts, **generate_kwargs):
        if features.shape[1] == self.n_mels:
            features = self.encode(features)
        print(features)
        print(self.detect_language(features))
        decoder_input_ids = torch.tensor(prompts)
            
        sampling_config = SamplingConfig(**generate_kwargs)
        
        output_ids = self.decoder.generate(decoder_input_ids,
                                           features,
                                           sampling_config)

        return output_ids