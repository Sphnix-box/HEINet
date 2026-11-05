import os
import numpy as np
import pandas as pd
import json
import warnings
import logging
import gc
import random
import math
import re
import ast
from tqdm import tqdm
from typing import Optional
from datetime import datetime
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
try:
    from bert_score import score as bert_metric
except ImportError:
    print("Warning: bert_score not installed. Install via 'pip install bert-score'")
    bert_metric = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('Warning: matplotlib is not installed. Please install it with: pip install matplotlib')
    plt = None
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader, TensorDataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU (CUDA not available)')
import argparse
parser = argparse.ArgumentParser(description='Hyperparameter Sensitivity Analysis')
parser.add_argument('--curvature_c', type=float, default=1.0, help='Curvature of the Poincaré ball')
parser.add_argument('--alpha_loss', type=float, default=0.1, help='Weight of Pragmatic Shift Loss')
parser.add_argument('--fusion_layer', type=int, default=4, help='Layer index to insert the HEI fusion module')
parser.add_argument('--cross_attn_heads', type=int, default=16, help='Number of heads in bipartite cross-attention')
parser.add_argument('--visual_num_heads', type=int, default=8, help='Number of heads in visual transformer')
parser.add_argument('--visual_num_layers', type=int, default=4, help='Number of layers in visual transformer')
parser.add_argument('--acoustic_num_heads', type=int, default=2, help='Number of heads in acoustic transformer')
parser.add_argument('--acoustic_num_layers', type=int, default=4, help='Number of layers in acoustic transformer')
parser.add_argument('--curvature_mode', type=str, default='fixed', choices=['zero', 'fixed', 'learnable'], help='Curvature mode: zero (Euclidean limit), fixed (c=curvature_c), learnable (bounded)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--curvature_min', type=float, default=1e-3, help='Lower bound for learnable curvature')
parser.add_argument('--curvature_max', type=float, default=5.0, help='Upper bound for learnable curvature')
parser.add_argument('--curvature_init', type=float, default=1.0, help='Initial curvature value for learnable mode')
parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
parser.add_argument('--output_root', type=str, default=None, help='Root dir for model/result outputs (default: PROJECT_ROOT)')
args, unknown = parser.parse_known_args()
SEED = args.seed
CURVATURE_C = args.curvature_c
CURVATURE_MODE = args.curvature_mode
CURVATURE_MIN = args.curvature_min
CURVATURE_MAX = args.curvature_max
CURVATURE_INIT = args.curvature_init
ALPHA_LOSS = args.alpha_loss
FUSION_LAYER = args.fusion_layer
CROSS_ATTN_HEADS = args.cross_attn_heads
VISUAL_NUM_HEADS = args.visual_num_heads
VISUAL_NUM_LAYERS = args.visual_num_layers
ACOUSTIC_NUM_HEADS = args.acoustic_num_heads
ACOUSTIC_NUM_LAYERS = args.acoustic_num_layers
TARGET_COLUMN = 'code_mixed_explanation'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = args.output_root if args.output_root else PROJECT_ROOT
DATA_ROOT = os.path.join(PROJECT_ROOT, 'Data')
TEXT_INPUT_PATH = os.path.join(DATA_ROOT, 'Text')
ACOUSTIC_INPUT_PATH = os.path.join(DATA_ROOT, 'Audio')
VISUAL_INPUT_PATH = os.path.join(DATA_ROOT, 'Video')
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
_RUN_TAG = f'seed{SEED}_alpha{ALPHA_LOSS}_c{CURVATURE_C}_layer{FUSION_LAYER}_vh{VISUAL_NUM_HEADS}_vl{VISUAL_NUM_LAYERS}_ah{ACOUSTIC_NUM_HEADS}_al{ACOUSTIC_NUM_LAYERS}_xh{CROSS_ATTN_HEADS}'
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f'models_{_RUN_TAG}_{RUN_TIMESTAMP}')
RESULT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f'results_{_RUN_TAG}_{RUN_TIMESTAMP}')
RESULT_VAL_DIR = os.path.join(RESULT_OUTPUT_DIR, 'val')
RESULT_TEST_DIR = os.path.join(RESULT_OUTPUT_DIR, 'test')
RESULT_PIC_DIR = os.path.join(RESULT_OUTPUT_DIR, 'pic')
RESULT_VAL_PIC_DIR = os.path.join(RESULT_OUTPUT_DIR, 'pic_val')
DEFAULT_MBART_LOCAL_PATH = os.path.join(PROJECT_ROOT, 'mbart_mtm')
DEFAULT_MBART_REMOTE_ID = 'facebook/mbart-large-50-many-to-many-mmt'
if os.path.isdir(DEFAULT_MBART_LOCAL_PATH):
    DEFAULT_MBART_CHECKPOINT = DEFAULT_MBART_LOCAL_PATH
else:
    DEFAULT_MBART_CHECKPOINT = DEFAULT_MBART_REMOTE_ID
BART_CHECKPOINT = os.environ.get('CCGNET_MBART_MODEL', DEFAULT_MBART_CHECKPOINT)
# mBART-50 language codes. BART (HEINet.py) used the English-only facebook/bart-base
# with empty prefixes, tokenizing the code-mixed text as-is; en_XX is the mBART-50 analog.
MBART_SRC_LANG = os.environ.get('CCGNET_MBART_SRC_LANG', 'en_XX')
MBART_TGT_LANG = os.environ.get('CCGNET_MBART_TGT_LANG', 'en_XX')
for required_dir in (TEXT_INPUT_PATH, ACOUSTIC_INPUT_PATH, VISUAL_INPUT_PATH):
    if not os.path.isdir(required_dir):
        raise FileNotFoundError(f'无法找到数据目录: {required_dir}，请确认 Data 文件夹结构是否正确。')
for output_dir in (MODEL_OUTPUT_DIR, RESULT_OUTPUT_DIR, RESULT_VAL_DIR, RESULT_TEST_DIR, RESULT_PIC_DIR, RESULT_VAL_PIC_DIR):
    os.makedirs(output_dir, exist_ok=True)
LOWERCASE_UTTERANCES = False
UNFOLDED_DIALOGUE = True
if UNFOLDED_DIALOGUE:
    SOURCE_COLUMN = 'dialogue'
else:
    SOURCE_COLUMN_1 = 'target'
    SOURCE_COLUMN_2 = 'context'
SOURCE_MAX_LEN = 480
TARGET_MAX_LEN = 50
MAX_UTTERANCES = 25
ACOUSTIC_DIM = 154
ACOUSTIC_MAX_LEN = 600
VISUAL_DIM = 2048
VISUAL_MAX_LEN = 96
BATCH_SIZE = 8
MAX_EPOCHS = args.epochs
if VISUAL_DIM % VISUAL_NUM_HEADS != 0:
    raise ValueError(f'VISUAL_NUM_HEADS={VISUAL_NUM_HEADS} 必须整除 VISUAL_DIM={VISUAL_DIM}')
if ACOUSTIC_DIM % ACOUSTIC_NUM_HEADS != 0:
    raise ValueError(f'ACOUSTIC_NUM_HEADS={ACOUSTIC_NUM_HEADS} 必须整除 ACOUSTIC_DIM={ACOUSTIC_DIM}')
BASE_LEARNING_RATE = 5e-06
NEW_LEARNING_RATE = 5e-05
WEIGHT_DECAY = 0.0001
NUM_BEAMS = 5
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3
EARLY_STOPPING_THRESHOLD = float('inf')

def set_random_seed(seed: int):
    print('Seed: {}'.format(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(SEED)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers import MBart50TokenizerFast, AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.models.mbart.modeling_mbart import MBartPreTrainedModel, MBartDecoder, MBartLearnedPositionalEmbedding, MBartEncoderLayer, shift_tokens_right, _make_causal_mask, _expand_mask
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformer_encoder import TransformerEncoder

class BipartiteIncongruityGraph(nn.Module):

    def __init__(self, dim_model: int, dim_context: int, dropout_rate: float=0.0, num_heads: int=1):
        super(BipartiteIncongruityGraph, self).__init__()
        self.dim_model = dim_model
        self.dim_context = dim_context
        if dim_model % num_heads != 0:
            raise ValueError(f'cross_attn_heads={num_heads} 必须整除 dim_model={dim_model}')
        self.attention_layer = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, dropout=dropout_rate, bias=True, add_zero_attn=False, batch_first=True, device=DEVICE)
        self.u_k = nn.Linear(dim_context, dim_model, bias=False)
        self.w1_k = nn.Linear(dim_model, 1, bias=False)
        self.w2_k = nn.Linear(dim_model, 1, bias=False)
        self.u_v = nn.Linear(dim_context, dim_model, bias=False)
        self.w1_v = nn.Linear(dim_model, 1, bias=False)
        self.w2_v = nn.Linear(dim_model, 1, bias=False)

    def forward(self, text_nodes: torch.Tensor, context_nodes: torch.Tensor):
        key_context = self.u_k(context_nodes)
        value_context = self.u_v(context_nodes)
        lambda_k = F.sigmoid(self.w1_k(text_nodes) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(text_nodes) + self.w2_v(value_context))
        k_cap = (1 - lambda_k) * text_nodes + lambda_k * key_context
        v_cap = (1 - lambda_v) * text_nodes + lambda_v * value_context
        incongruity_features, _ = self.attention_layer(query=text_nodes, key=k_cap, value=v_cap)
        return incongruity_features

class HyperbolicPragmaticBottleneck(nn.Module):

    def __init__(self, dim_model: int, dropout_rate: float, bottleneck_dim: int=None, c: float=1.0, curvature_mode: str='fixed', c_min: float=1e-3, c_max: float=5.0, c_init: float=1.0):
        super(HyperbolicPragmaticBottleneck, self).__init__()
        # Curvature configuration. Three modes are parameter-matched except that
        # 'learnable' adds exactly one scalar parameter (rho).
        #   zero      : Euclidean limit, geometry ops are skipped entirely.
        #   fixed     : constant curvature self.c (default 1.0), original behaviour.
        #   learnable : c = c_min + (c_max - c_min) * sigmoid(rho), rho trainable.
        self.curvature_mode = curvature_mode
        self.c = c
        self.c_min = c_min
        self.c_max = c_max
        if curvature_mode == 'learnable':
            p = (c_init - c_min) / (c_max - c_min)
            p = min(max(p, 1e-6), 1 - 1e-6)
            rho0 = math.log(p / (1 - p))
            self.rho = nn.Parameter(torch.tensor(float(rho0)))
        self.dropout_rate = dropout_rate
        if bottleneck_dim is None:
            bottleneck_dim = dim_model // 2
        self.audio_adapter = nn.Sequential(nn.LayerNorm(dim_model), nn.Linear(dim_model, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, dim_model))
        self.video_adapter = nn.Sequential(nn.LayerNorm(dim_model), nn.Linear(dim_model, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, dim_model))
        self.gate_audio = nn.Linear(dim_model * 2, 1)
        self.gate_video = nn.Linear(dim_model * 2, 1)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        for m in [self.audio_adapter, self.video_adapter]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0)
        nn.init.zeros_(self.gate_audio.weight)
        nn.init.constant_(self.gate_audio.bias, -2.0)
        nn.init.zeros_(self.gate_video.weight)
        nn.init.constant_(self.gate_video.bias, -2.0)

    def get_c(self):
        """Current curvature. Tensor (graph-preserving) for learnable, float otherwise."""
        if self.curvature_mode == 'learnable':
            return self.c_min + (self.c_max - self.c_min) * torch.sigmoid(self.rho)
        return self.c

    def expmap0(self, u, c=None):
        c = self.get_c() if c is None else c
        sqrt_c = c ** 0.5
        u_norm = torch.norm(u, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        gamma_1 = torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, y, c=None):
        c = self.get_c() if c is None else c
        sqrt_c = c ** 0.5
        max_norm = 1.0 / sqrt_c - 1e-05
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        if isinstance(max_norm, float):
            y_norm = torch.clamp(y_norm, max=max_norm)
        else:
            y_norm = torch.minimum(y_norm, max_norm)
        return torch.atanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)

    def mobius_scalar_mul(self, r, x, c=None):
        c = self.get_c() if c is None else c
        sqrt_c = c ** 0.5
        max_norm = 1.0 / sqrt_c - 1e-05
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        if isinstance(max_norm, float):
            x_norm = torch.clamp(x_norm, max=max_norm)
        else:
            x_norm = torch.minimum(x_norm, max_norm)
        artanh_term = torch.atanh(sqrt_c * x_norm)
        tanh_term = torch.tanh(r * artanh_term)
        return 1.0 / sqrt_c * tanh_term * (x / x_norm)

    def mobius_add(self, x, y, c=None):
        c = self.get_c() if c is None else c
        cx2 = c * torch.sum(x * x, dim=-1, keepdim=True)
        cy2 = c * torch.sum(y * y, dim=-1, keepdim=True)
        cxy = c * torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * cxy + cy2) * x + (1 - cx2) * y
        denom = 1 + 2 * cxy + cx2 * cy2
        res = num / denom.clamp_min(1e-15)
        sqrt_c = c ** 0.5
        res_norm = torch.norm(res, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 / sqrt_c - 1e-05
        cond = res_norm > max_norm
        projected = res / res_norm.clamp_min(1e-15) * max_norm
        return torch.where(cond, projected, res)

    def forward(self, text_input, audio_raw, video_raw):
        audio_feat = self.audio_adapter(audio_raw)
        video_feat = self.video_adapter(video_raw)
        g_a = torch.sigmoid(self.gate_audio(torch.cat([text_input, audio_feat], dim=-1)))
        g_v = torch.sigmoid(self.gate_video(torch.cat([text_input, video_feat], dim=-1)))
        if self.curvature_mode == 'zero':
            # Explicit Euclidean limit: exp_0(x)=x, mobius scalar mul -> g*x,
            # mobius add -> x+y, log_0(x)=x. Gates and residual are preserved.
            multimodal_delta = g_a * audio_feat + g_v * video_feat
        else:
            c = self.get_c()
            audio_hyp = self.expmap0(audio_feat, c)
            video_hyp = self.expmap0(video_feat, c)
            audio_gated_hyp = self.mobius_scalar_mul(g_a, audio_hyp, c)
            video_gated_hyp = self.mobius_scalar_mul(g_v, video_hyp, c)
            fused_hyp = self.mobius_add(audio_gated_hyp, video_gated_hyp, c)
            multimodal_delta = self.logmap0(fused_hyp, c)
        output = self.final_layer_norm(text_input + self.dropout_layer(multimodal_delta))
        return output

class HEI_BART_Layer(nn.Module):

    def __init__(self, dim_model: int, dim_audio: int, dim_video: int, dropout_rate: float, c: float=1.0, cross_attn_heads: int=1, curvature_mode: str='fixed', c_min: float=1e-3, c_max: float=5.0, c_init: float=1.0):
        super(HEI_BART_Layer, self).__init__()
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.graph_audio = BipartiteIncongruityGraph(dim_model, dim_audio, dropout_rate, num_heads=cross_attn_heads)
        self.graph_video = BipartiteIncongruityGraph(dim_model, dim_video, dropout_rate, num_heads=cross_attn_heads)
        self.hyperbolic_bottleneck = HyperbolicPragmaticBottleneck(dim_model, dropout_rate, c=c, curvature_mode=curvature_mode, c_min=c_min, c_max=c_max, c_init=c_init)

    def forward(self, text_input, acoustic_context, visual_context):
        acoustic_context = acoustic_context.permute(0, 2, 1)
        acoustic_context = self.acoustic_context_transform(acoustic_context).permute(0, 2, 1)
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context).permute(0, 2, 1)
        audio_incongruity = self.graph_audio(text_nodes=text_input, context_nodes=acoustic_context)
        video_incongruity = self.graph_video(text_nodes=text_input, context_nodes=visual_context)
        output = self.hyperbolic_bottleneck(text_input, audio_incongruity, video_incongruity)
        return output

class MultimodalBartEncoder(MBartPreTrainedModel):

    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        self.embed_positions = MBartLearnedPositionalEmbedding(config.max_position_embeddings, embed_dim)
        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.init_weights()
        self.gradient_checkpointing = False
        self.fusion_at_layer = [FUSION_LAYER]
        self.visual_transformer = TransformerEncoder(d_model=VISUAL_DIM, num_layers=VISUAL_NUM_LAYERS, num_heads=VISUAL_NUM_HEADS, dim_feedforward=VISUAL_DIM)
        self.acoustic_transformer = TransformerEncoder(d_model=ACOUSTIC_DIM, num_layers=ACOUSTIC_NUM_LAYERS, num_heads=ACOUSTIC_NUM_HEADS, dim_feedforward=ACOUSTIC_DIM)
        self.HEI_layer = HEI_BART_Layer(dim_model=embed_dim, dim_audio=ACOUSTIC_DIM, dim_video=VISUAL_DIM, dropout_rate=0.2, c=CURVATURE_C, cross_attn_heads=CROSS_ATTN_HEADS, curvature_mode=CURVATURE_MODE, c_min=CURVATURE_MIN, c_max=CURVATURE_MAX, c_init=CURVATURE_INIT)

    def forward(self, input_ids=None, attention_mask=None, acoustic_input=None, visual_input=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        literal_hidden_states = None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for idx, encoder_layer in enumerate(self.layers):
            if idx in self.fusion_at_layer:
                literal_hidden_states = hidden_states.clone()
                acoustic_context = self.acoustic_transformer(acoustic_input)[-1]
                visual_context = self.visual_transformer(visual_input)[-1]
                hidden_states = self.HEI_layer(text_input=hidden_states, acoustic_context=acoustic_context, visual_context=visual_context)
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):

                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)
                        return custom_forward
                    layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(encoder_layer), hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions, literal_hidden_states] if v is not None))
        output = BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)
        output.literal_hidden_states = literal_hidden_states
        return output

class MultimodalBartModel(MBartPreTrainedModel):

    def __init__(self, config: MBartConfig):
        super().__init__(config)
        padding_idx, vocab_size = (config.pad_token_id, config.vocab_size)
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = MultimodalBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)
        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self, input_ids=None, attention_mask=None, acoustic_input=None, visual_input=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, acoustic_input=acoustic_input, visual_input=visual_input, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        out = Seq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)
        if hasattr(encoder_outputs, 'literal_hidden_states'):
            out.literal_hidden_states = encoder_outputs.literal_hidden_states
        return out

class MultimodalBartForConditionalGeneration(MBartPreTrainedModel):
    base_model_prefix = 'model'
    _keys_to_ignore_on_load_missing = ['final_logits_bias', 'lm_head\\.weight']

    def __init__(self, config: MBartConfig):
        super().__init__(config)
        self.model = MultimodalBartModel(config)
        self.register_buffer('final_logits_bias', torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer('final_logits_bias', new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, acoustic_input=None, visual_input=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(input_ids, attention_mask=attention_mask, acoustic_input=acoustic_input, visual_input=visual_input, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            if hasattr(outputs, 'literal_hidden_states') and outputs.literal_hidden_states is not None:
                shift_loss = F.mse_loss(outputs.encoder_last_hidden_state, outputs.literal_hidden_states)
                alpha = ALPHA_LOSS
                masked_lm_loss = masked_lm_loss + alpha * shift_loss
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data

def prepare_dataset(text_path: str, acosutic_path: str, visual_path: str, lowercase_utterances: bool=False, unfolded_dialogue: bool=True):
    data = read_json_data(text_path)
    code_mixed_explanation = []
    if unfolded_dialogue:
        dialogue = []
        for i in range(1, len(data) + 1):
            data_point = data[str(i)]
            example_target_speaker = str(data_point['target_speaker']).upper()
            example_target_utterance = str(data_point['target_utterance'])
            example_dialogue = '[CONTEXT] '
            for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
                example_dialogue = example_dialogue + str(speaker).upper() + ' : ' + str(utterance) + ' | '
            example_dialogue = example_dialogue + ' [TARGET] ' + example_target_speaker + ' : ' + example_target_utterance + ' | '
            example_dialogue = re.sub(' +', ' ', example_dialogue)
            dialogue.append(example_dialogue)
            code_mixed_explanation.append(str(data_point['code_mixed_explanation']))
        df = pd.DataFrame(list(zip(dialogue, code_mixed_explanation)), columns=['dialogue', 'code_mixed_explanation'])
        TOKENIZER.add_tokens(['[CONTEXT]', '[TARGET]'], special_tokens=True)
        MODEL.resize_token_embeddings(len(TOKENIZER))
        del dialogue
        del example_dialogue
    else:
        target = []
        context = []
        for i in range(1, len(data) + 1):
            data_point = data[str(i)]
            example_target_speaker = str(data_point['target_speaker']).upper()
            example_target_utterance = str(data_point['target_utterance'])
            example_target_utterance = example_target_speaker + ' : ' + example_target_utterance
            example_target_utterance = re.sub(' +', ' ', example_target_utterance)
            target.append(example_target_utterance)
            example_context_utterance = ''
            for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
                example_context_utterance = example_context_utterance + str(speaker).upper() + ' : ' + str(utterance) + ' | '
            example_context_utterance = re.sub(' +', ' ', example_context_utterance)
            context.append(example_context_utterance)
            code_mixed_explanation.append(str(data_point['code_mixed_explanation']))
        df = pd.DataFrame(list(zip(context, target, code_mixed_explanation)), columns=['context', 'target', 'code_mixed_explanation'])
        del target
        del context
        del example_context_utterance
    acosutic_data = pd.read_pickle(acosutic_path)
    df['audio_features'] = acosutic_data['audio_feats']
    visaul_data = pd.read_pickle(visual_path)
    df['visual_features'] = visaul_data['video_feats']
    df = df[df['code_mixed_explanation'] != '?']
    df = df.dropna()
    if lowercase_utterances:
        df = df.apply(lambda x: x.astype(str).str.lower())
    del data
    del text_path
    del acosutic_path
    del visual_path
    del acosutic_data
    del visaul_data
    del code_mixed_explanation
    del example_target_speaker
    del example_target_utterance
    gc.collect()
    return df

def pad_seq(tensor: torch.tensor, dim: int, max_len: int):
    if max_len > tensor.shape[0]:
        return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
    else:
        return tensor[:max_len]

def preprocess_dataset(dataset, unfolded_dialogue: bool=True):
    if unfolded_dialogue:
        source = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN].values.tolist()]
        model_inputs = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
        del source
    else:
        source_1 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_1].values.tolist()]
        source_2 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_2].values.tolist()]
        model_inputs = TOKENIZER(source_1, source_2, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
        del source_1
        del source_2
    target = [TARGET_PREFIX + t for t in dataset[TARGET_COLUMN].values.tolist()]
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(target, max_length=TARGET_MAX_LEN, padding='max_length', truncation=True)
        labels['input_ids'] = [[l if l != TOKENIZER.pad_token_id else -100 for l in label] for label in labels['input_ids']]
    model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
    model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
    model_inputs['acoustic_input'] = torch.stack([pad_seq(torch.tensor(af, dtype=torch.float), dim=ACOUSTIC_DIM, max_len=ACOUSTIC_MAX_LEN) for af in dataset['audio_features'].values.tolist()], 0).to(DEVICE)
    model_inputs['visual_input'] = torch.stack([pad_seq(torch.tensor(vf[0], dtype=torch.float), dim=VISUAL_DIM, max_len=VISUAL_MAX_LEN) for vf in dataset['visual_features'].values.tolist()], 0).to(DEVICE)
    model_inputs['labels'] = torch.tensor([l for l in labels['input_ids']], dtype=torch.long, device=DEVICE)
    del target
    del labels
    gc.collect()
    return model_inputs

def set_up_data_loader(text_path: str, acosutic_path: str, visual_path: str, lowercase_utterances: bool=False, unfolded_dialogue: bool=True):
    dataset = preprocess_dataset(prepare_dataset(text_path=text_path, acosutic_path=acosutic_path, visual_path=visual_path, lowercase_utterances=lowercase_utterances, unfolded_dialogue=unfolded_dialogue), unfolded_dialogue=unfolded_dialogue)
    dataset = TensorDataset(dataset['input_ids'], dataset['attention_mask'], dataset['acoustic_input'], dataset['visual_input'], dataset['labels'])
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def get_scores(reference_list: list, hypothesis_list: list):
    count = 0
    met = 0
    bleu_1 = 0
    bleu_2 = 0
    bleu_3 = 0
    bleu_4 = 0
    rouge1 = 0
    rouge2 = 0
    rougel = 0
    weights_1 = (1.0 / 1.0,)
    weights_2 = (1.0 / 2.0, 1.0 / 2.0)
    weights_3 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    weights_4 = (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure
        met += meteor_score([reference], hypothesis)
        reference = reference.split()
        hypothesis = hypothesis.split()
        bleu_1 += sentence_bleu([reference], hypothesis, weights_1)
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1
    bert_score_val = 0.0
    if bert_metric is not None:
        try:
            local_model_path = os.path.join(PROJECT_ROOT, 'roberta-large')
            if os.path.isdir(local_model_path):
                model_type = local_model_path
                required_files = ['config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(local_model_path, f))]
                if missing_files:
                    print(f'[BERTScore Error] Local model directory found but missing files: {missing_files}')
            else:
                print(f'[BERTScore Warning] Local model not found at {local_model_path}. Attempting to download (will fail if offline).')
                model_type = 'roberta-large'
            _, _, F1_bert = bert_metric(hypothesis_list, reference_list, model_type=model_type, num_layers=10, verbose=False, device=DEVICE, batch_size=16)
            bert_score_val = F1_bert.mean().item() * 100
        except Exception as e:
            print(f"\n{'=' * 20} BERTScore Calculation FAILED {'=' * 20}")
            print(f'Error: {e}')
            print(f'Model Type Path: {model_type}')
            print(f'Current Working Directory: {os.getcwd()}')
            print('=' * 60 + '\n')
    return {'rouge_1': rouge1 * 100 / count, 'rouge_2': rouge2 * 100 / count, 'rouge_L': rougel * 100 / count, 'bleu_1': bleu_1 * 100 / count, 'bleu_2': bleu_2 * 100 / count, 'bleu_3': bleu_3 * 100 / count, 'bleu_4': bleu_4 * 100 / count, 'meteor': met * 100 / count, 'bert_score': bert_score_val}

def _save(model, output_dir: str, tokenizer=None, state_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    if not isinstance(model, PreTrainedModel):
        if isinstance(unwrap_model(model), PreTrainedModel):
            if state_dict is None:
                state_dict = model.state_dict()
            unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
        else:
            if state_dict is None:
                state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        model.save_pretrained(output_dir, state_dict=state_dict)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

def save_model(model, output_dir: str, tokenizer=None, state_dict=None):
    _save(model, output_dir, tokenizer=tokenizer, state_dict=state_dict)

def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc='Training Iteration')):
        batch = tuple((t.to(DEVICE) for t in batch))
        input_ids, attention_mask, acoustic_input, visual_input, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, acoustic_input=acoustic_input, visual_input=visual_input, labels=labels)
        loss = outputs.loss
        epoch_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_train_loss / step

def val_epoch(model, data_loader, optimizer):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc='Validation Loss Iteration')):
            batch = tuple((t.to(DEVICE) for t in batch))
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, acoustic_input=acoustic_input, visual_input=visual_input, labels=labels)
            loss = outputs['loss']
            epoch_val_loss += loss.item()
    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_val_loss / step

def test_epoch(model, tokenizer, data_loader, desc, **gen_kwargs):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple((t.to(DEVICE) for t in batch))
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, acoustic_input=acoustic_input, visual_input=visual_input, **gen_kwargs)
            generated_ids = generated_ids.detach().cpu().numpy()
            generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = labels.detach().cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(decoded_preds)
            gold.extend(decoded_labels)
    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del generated_ids
    del decoded_preds
    del decoded_labels
    gc.collect()
    torch.cuda.empty_cache()
    return (predictions, gold)

def get_val_scores(model, tokenizer, data_loader, desc, epoch, cumulative_csv_path=None, **gen_kwargs):
    predictions, gold = test_epoch(model, tokenizer, data_loader, desc=desc, **gen_kwargs)
    result = get_scores(predictions, gold)
    df_results = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
    if cumulative_csv_path:
        df_results['epoch'] = epoch + 1
        if not os.path.exists(cumulative_csv_path):
            df_results.to_csv(cumulative_csv_path, index=False)
        else:
            df_results.to_csv(cumulative_csv_path, mode='a', header=False, index=False)
        df_results = df_results.drop(columns=['epoch'])
    if 'Validation' in desc:
        file_name = os.path.join(RESULT_VAL_DIR, f'HEINet_TAV_BART_alpha{ALPHA_LOSS}_c{CURVATURE_C}_layer{FUSION_LAYER}_vh{VISUAL_NUM_HEADS}_vl{VISUAL_NUM_LAYERS}_ah{ACOUSTIC_NUM_HEADS}_al{ACOUSTIC_NUM_LAYERS}_xh{CROSS_ATTN_HEADS}_epoch_{epoch + 1}_val_results.csv')
        df_results.to_csv(file_name, index=False)
        print('Validation File saved')
    elif 'Test' in desc:
        file_name = os.path.join(RESULT_TEST_DIR, f'HEINet_TAV_BART_alpha{ALPHA_LOSS}_c{CURVATURE_C}_layer{FUSION_LAYER}_vh{VISUAL_NUM_HEADS}_vl{VISUAL_NUM_LAYERS}_ah{ACOUSTIC_NUM_HEADS}_al{ACOUSTIC_NUM_LAYERS}_xh{CROSS_ATTN_HEADS}_epoch_{epoch + 1}_test_results.csv')
        df_results.to_csv(file_name, index=False)
        print('Test File saved')
    del predictions
    del gold
    gc.collect()
    torch.cuda.empty_cache()
    return result

def prepare_for_training(model, base_learning_rate: float, new_learning_rate: float, weight_decay: float, num_training_steps: int):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        new_params_list.append(param)
    optimizer = AdamW([{'params': base_params_list, 'lr': base_learning_rate, 'weight_decay': weight_decay}, {'params': new_params_list, 'lr': new_learning_rate, 'weight_decay': weight_decay}], lr=base_learning_rate, weight_decay=weight_decay)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache()
    return (optimizer, scheduler)

def plot_metrics(epochs, metrics_dict, save_dir):
    if plt is None:
        print('Warning: matplotlib is not available. Skipping plot generation.')
        return
    metrics_config = [('rouge_1', 'ROUGE-1', 'b', 'o'), ('rouge_2', 'ROUGE-2', 'g', 's'), ('rouge_L', 'ROUGE-L', 'r', '^'), ('bleu_1', 'BLEU-1', 'c', 'D'), ('bleu_2', 'BLEU-2', 'm', 'v'), ('bleu_3', 'BLEU-3', 'y', '<'), ('bleu_4', 'BLEU-4', 'orange', '>'), ('meteor', 'METEOR', 'purple', 'p'), ('bert_score', 'BERTScore', 'brown', '*')]
    saved_files = []
    for metric_key, metric_name, color, marker in metrics_config:
        if metric_key not in metrics_dict or len(metrics_dict[metric_key]) == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, metrics_dict[metric_key], color=color, marker=marker, linewidth=2.5, markersize=8, label=metric_name)
        values = metrics_dict[metric_key]
        max_value = max(values)
        max_index = values.index(max_value)
        max_epoch = epochs[max_index]
        ax.annotate(f'Max: {max_value:.2f}', xy=(max_epoch, max_value), xytext=(0, 10), textcoords='offset points', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3), fontsize=9, fontweight='bold')
        target_epoch = 67
        if target_epoch in epochs:
            idx_target = epochs.index(target_epoch)
            val_target = values[idx_target]
            if target_epoch != max_epoch:
                ax.annotate(f'Ep{target_epoch}: {val_target:.2f}', xy=(target_epoch, val_target), xytext=(0, -20), textcoords='offset points', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8), fontsize=9, fontweight='bold')
        ax.set_title(f'{metric_name} Over Epochs (alpha={ALPHA_LOSS}, c={CURVATURE_C}, layer={FUSION_LAYER}, vh={VISUAL_NUM_HEADS}, vl={VISUAL_NUM_LAYERS}, ah={ACOUSTIC_NUM_HEADS}, al={ACOUSTIC_NUM_LAYERS}, xh={CROSS_ATTN_HEADS})', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        ax.legend(loc='best', fontsize=11)
        if len(epochs) > 0:
            ax.set_xticks(epochs)
            ax.set_xticklabels([int(e) for e in epochs])
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{metric_key}_alpha{ALPHA_LOSS}_c{CURVATURE_C}_layer{FUSION_LAYER}_vh{VISUAL_NUM_HEADS}_vl{VISUAL_NUM_LAYERS}_ah{ACOUSTIC_NUM_HEADS}_al{ACOUSTIC_NUM_LAYERS}_xh{CROSS_ATTN_HEADS}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(save_path)
    print(f'Updated {len(saved_files)} test metric plots in {save_dir}')

def plot_val_metrics(epochs, val_metrics_dict, save_dir):
    if plt is None:
        print('Warning: matplotlib is not available. Skipping plot generation.')
        return
    metrics_config = [('rouge_1', 'Val ROUGE-1', 'b', 'o'), ('rouge_2', 'Val ROUGE-2', 'g', 's'), ('rouge_L', 'Val ROUGE-L', 'r', '^'), ('bleu_1', 'Val BLEU-1', 'c', 'D'), ('bleu_2', 'Val BLEU-2', 'm', 'v'), ('bleu_3', 'Val BLEU-3', 'y', '<'), ('bleu_4', 'Val BLEU-4', 'orange', '>'), ('meteor', 'Val METEOR', 'purple', 'p'), ('bert_score', 'Val BERTScore', 'brown', '*')]
    saved_files = []
    for metric_key, metric_name, color, marker in metrics_config:
        if metric_key not in val_metrics_dict or len(val_metrics_dict[metric_key]) == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, val_metrics_dict[metric_key], color=color, marker=marker, linewidth=2.5, markersize=8, label=metric_name)
        values = val_metrics_dict[metric_key]
        max_value = max(values)
        max_index = values.index(max_value)
        max_epoch = epochs[max_index]
        ax.annotate(f'Max: {max_value:.2f}', xy=(max_epoch, max_value), xytext=(0, 10), textcoords='offset points', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3), fontsize=9, fontweight='bold')
        target_epoch = 67
        if target_epoch in epochs:
            idx_target = epochs.index(target_epoch)
            val_target = values[idx_target]
            if target_epoch != max_epoch:
                ax.annotate(f'Ep{target_epoch}: {val_target:.2f}', xy=(target_epoch, val_target), xytext=(0, -20), textcoords='offset points', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8), fontsize=9, fontweight='bold')
        ax.set_title(f'{metric_name} Over Epochs (Validation, alpha={ALPHA_LOSS}, c={CURVATURE_C}, layer={FUSION_LAYER}, vh={VISUAL_NUM_HEADS}, vl={VISUAL_NUM_LAYERS}, ah={ACOUSTIC_NUM_HEADS}, al={ACOUSTIC_NUM_LAYERS}, xh={CROSS_ATTN_HEADS})', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        ax.legend(loc='best', fontsize=11)
        if len(epochs) > 0:
            ax.set_xticks(epochs)
            ax.set_xticklabels([int(e) for e in epochs])
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'val_{metric_key}_alpha{ALPHA_LOSS}_c{CURVATURE_C}_layer{FUSION_LAYER}_vh{VISUAL_NUM_HEADS}_vl{VISUAL_NUM_LAYERS}_ah{ACOUSTIC_NUM_HEADS}_al{ACOUSTIC_NUM_LAYERS}_xh{CROSS_ATTN_HEADS}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(save_path)
    print(f'Updated {len(saved_files)} validation metric plots in {save_dir}')

def train(model, tokenizer, train_data_loader, val_data_loader, test_data_loader, base_learning_rate, new_learning_rate, weight_decay, **gen_kwargs):
    total_steps = len(train_data_loader) * MAX_EPOCHS
    optimizer, scheduler = prepare_for_training(model=model, base_learning_rate=base_learning_rate, new_learning_rate=new_learning_rate, weight_decay=weight_decay, num_training_steps=total_steps)
    train_losses = []
    val_losses = []
    val_rouge_2 = []
    patience = 1
    val_metrics = {'rouge_1': [], 'rouge_2': [], 'rouge_L': [], 'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [], 'meteor': [], 'bert_score': []}
    test_metrics = {'rouge_1': [], 'rouge_2': [], 'rouge_L': [], 'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [], 'meteor': [], 'bert_score': []}
    epochs_list = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file_path = os.path.join(RESULT_OUTPUT_DIR, f'training_metrics_alpha{ALPHA_LOSS}_c{CURVATURE_C}_layer{FUSION_LAYER}_vh{VISUAL_NUM_HEADS}_vl{VISUAL_NUM_LAYERS}_ah{ACOUSTIC_NUM_HEADS}_al{ACOUSTIC_NUM_LAYERS}_xh{CROSS_ATTN_HEADS}_{timestamp}.csv')
    print(f'Training metrics will be saved to: {metrics_file_path}')
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_data_loader, optimizer, scheduler)
        train_losses.append(train_loss)
        val_loss = val_epoch(model, val_data_loader, optimizer)
        val_losses.append(val_loss)
        val_results = get_val_scores(model, tokenizer, val_data_loader, desc='Validation Generation Iteration', epoch=epoch, **gen_kwargs)
        val_rouge_2.append(val_results['rouge_2'])
        val_metrics['rouge_1'].append(val_results['rouge_1'])
        val_metrics['rouge_2'].append(val_results['rouge_2'])
        val_metrics['rouge_L'].append(val_results['rouge_L'])
        val_metrics['bleu_1'].append(val_results['bleu_1'])
        val_metrics['bleu_2'].append(val_results['bleu_2'])
        val_metrics['bleu_3'].append(val_results['bleu_3'])
        val_metrics['bleu_4'].append(val_results['bleu_4'])
        val_metrics['meteor'].append(val_results['meteor'])
        val_metrics['bert_score'].append(val_results['bert_score'])
        test_results = get_val_scores(model, tokenizer, test_data_loader, desc='Test Generation Iteration', epoch=epoch, **gen_kwargs)
        epochs_list.append(epoch + 1)
        test_metrics['rouge_1'].append(test_results['rouge_1'])
        test_metrics['rouge_2'].append(test_results['rouge_2'])
        test_metrics['rouge_L'].append(test_results['rouge_L'])
        test_metrics['bleu_1'].append(test_results['bleu_1'])
        test_metrics['bleu_2'].append(test_results['bleu_2'])
        test_metrics['bleu_3'].append(test_results['bleu_3'])
        test_metrics['bleu_4'].append(test_results['bleu_4'])
        test_metrics['meteor'].append(test_results['meteor'])
        test_metrics['bert_score'].append(test_results['bert_score'])
        _hpb = model.model.encoder.HEI_layer.hyperbolic_bottleneck
        if _hpb.curvature_mode == 'zero':
            current_c = 0.0
        else:
            cc = _hpb.get_c()
            current_c = float(cc.item()) if torch.is_tensor(cc) else float(cc)
        current_metrics = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'curvature_c': current_c}
        print('Epoch {} curvature_c = {}'.format(epoch + 1, current_c))
        for k, v in val_results.items():
            current_metrics[f'val_{k}'] = v
        for k, v in test_results.items():
            current_metrics[f'test_{k}'] = v
        df_metrics = pd.DataFrame([current_metrics])
        if not os.path.exists(metrics_file_path):
            df_metrics.to_csv(metrics_file_path, index=False)
        else:
            df_metrics.to_csv(metrics_file_path, mode='a', header=False, index=False)
        print(f'Saved epoch {epoch + 1} metrics to csv.')
        plot_metrics(epochs_list, test_metrics, RESULT_PIC_DIR)
        plot_val_metrics(epochs_list, val_metrics, RESULT_VAL_PIC_DIR)
        print('Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}'.format(epoch + 1, train_loss, val_loss, min(val_losses)))
        print('\nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}\tval_bert_score: {}'.format(val_results['rouge_1'], val_results['rouge_2'], val_results['rouge_L'], val_results['bleu_1'], val_results['bleu_2'], val_results['bleu_3'], val_results['bleu_4'], val_results['meteor'], val_results['bert_score']))
        print('\ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}\ttest_bert_score: {}'.format(test_results['rouge_1'], test_results['rouge_2'], test_results['rouge_L'], test_results['bleu_1'], test_results['bleu_2'], test_results['bleu_3'], test_results['bleu_4'], test_results['meteor'], test_results['bert_score']))
        path = os.path.join(MODEL_OUTPUT_DIR, f"HEIN e t_TAV_MBART_alpha{ALPHA_LOSS}_c{CURVATURE_C}_layer{FUSION_LAYER}_vh{VISUAL_NUM_HEADS}_vl{VISUAL_NUM_LAYERS}_ah{ACOUSTIC_NUM_HEADS}_al{ACOUSTIC_NUM_LAYERS}_xh{CROSS_ATTN_HEADS}_{TARGET_COLUMN}_epoch_{epoch + 1}_{datetime.now().strftime('%d-%m-%Y-%H:%M')}")
        # Keep only the single best-val-ROUGE-2 checkpoint (prune previous) to bound
        # disk use: mBART-large checkpoints are ~2.8GB each, 120 epochs would overflow.
        if val_results['rouge_2'] >= max(val_rouge_2):
            import glob, shutil
            for old in glob.glob(os.path.join(MODEL_OUTPUT_DIR, '*epoch_*')):
                if os.path.isdir(old):
                    shutil.rmtree(old, ignore_errors=True)
            save_model(model, path, tokenizer)
            print('Model saved at path: ', path)
        if val_results['rouge_2'] < max(val_rouge_2):
            patience = patience + 1
            if patience == EARLY_STOPPING_THRESHOLD:
                break
        else:
            patience = 1
        del train_loss
        del val_loss
        del path
        gc.collect()
        torch.cuda.empty_cache()
if __name__ == '__main__':
    print(f'加载 mBART 检查点: {BART_CHECKPOINT}')
    MODEL = MultimodalBartForConditionalGeneration.from_pretrained(BART_CHECKPOINT)
    MODEL.to(DEVICE)
    TOKENIZER = MBart50TokenizerFast.from_pretrained(BART_CHECKPOINT, src_lang=MBART_SRC_LANG, tgt_lang=MBART_TGT_LANG)
    FORCED_BOS_TOKEN_ID = TOKENIZER.convert_tokens_to_ids(MBART_TGT_LANG)
    MODEL.config.forced_bos_token_id = FORCED_BOS_TOKEN_ID
    print(f'mBART langs -> src={MBART_SRC_LANG}, tgt={MBART_TGT_LANG}, forced_bos_token_id={FORCED_BOS_TOKEN_ID}')
    SOURCE_PREFIX = ''
    TARGET_PREFIX = ''
    print(TARGET_COLUMN)
    print(MODEL_OUTPUT_DIR)
    print(RESULT_OUTPUT_DIR)
    print(SOURCE_PREFIX)
    print(TARGET_PREFIX)
    print(f'Sensitivity config -> layer={FUSION_LAYER}, cross_attn_heads={CROSS_ATTN_HEADS}, visual_heads={VISUAL_NUM_HEADS}, visual_layers={VISUAL_NUM_LAYERS}, acoustic_heads={ACOUSTIC_NUM_HEADS}, acoustic_layers={ACOUSTIC_NUM_LAYERS}')
    gc.collect()
    pytorch_total_params = sum((p.numel() for p in MODEL.parameters()))
    print('Total parameters: ', pytorch_total_params)
    pytorch_total_train_params = sum((p.numel() for p in MODEL.parameters() if p.requires_grad))
    print('Total trainable parameters: ', pytorch_total_train_params)
    for name, param in MODEL.named_parameters():
        if 'acoustic_transformer' in name or 'visual_transformer' in name or 'HEI_layer' in name:
            print(name)
    train_dataset = set_up_data_loader(text_path=os.path.join(TEXT_INPUT_PATH, 'train_data_final_plain.json'), acosutic_path=os.path.join(ACOUSTIC_INPUT_PATH, 'train_acoustic_features.pkl'), visual_path=os.path.join(VISUAL_INPUT_PATH, 'train_visual_features.pkl'), lowercase_utterances=LOWERCASE_UTTERANCES, unfolded_dialogue=UNFOLDED_DIALOGUE)
    val_dataset = set_up_data_loader(text_path=os.path.join(TEXT_INPUT_PATH, 'val_data_final_plain.json'), acosutic_path=os.path.join(ACOUSTIC_INPUT_PATH, 'val_acoustic_features.pkl'), visual_path=os.path.join(VISUAL_INPUT_PATH, 'val_visual_features.pkl'), lowercase_utterances=LOWERCASE_UTTERANCES, unfolded_dialogue=UNFOLDED_DIALOGUE)
    test_dataset = set_up_data_loader(text_path=os.path.join(TEXT_INPUT_PATH, 'test_data_final_plain.json'), acosutic_path=os.path.join(ACOUSTIC_INPUT_PATH, 'test_acoustic_features.pkl'), visual_path=os.path.join(VISUAL_INPUT_PATH, 'test_visual_features.pkl'), lowercase_utterances=LOWERCASE_UTTERANCES, unfolded_dialogue=UNFOLDED_DIALOGUE)
    gc.collect()
    gen_kwargs = {'num_beams': NUM_BEAMS, 'max_length': TARGET_MAX_LEN, 'early_stopping': EARLY_STOPPING, 'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE, 'length_penalty': 1.0, 'min_length': 5, 'forced_bos_token_id': FORCED_BOS_TOKEN_ID}
    train(model=MODEL, tokenizer=TOKENIZER, train_data_loader=train_dataset, val_data_loader=val_dataset, test_data_loader=test_dataset, base_learning_rate=BASE_LEARNING_RATE, new_learning_rate=NEW_LEARNING_RATE, weight_decay=WEIGHT_DECAY, **gen_kwargs)
    print('Model Trained!')