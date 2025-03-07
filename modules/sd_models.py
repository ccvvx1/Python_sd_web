import collections
import importlib
import os
import sys
import threading
import enum

import torch
import re
import safetensors.torch
from omegaconf import OmegaConf, ListConfig
from urllib import request
import ldm.modules.midas as midas

from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config, sd_unet, sd_models_xl, cache, extra_networks, processing, lowvram, sd_hijack, patches
from modules.timer import Timer
from modules.shared import opts
import tomesd
import numpy as np

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_aliases = {}
checkpoint_alisases = checkpoint_aliases  # for compatibility with old name
checkpoints_loaded = collections.OrderedDict()


class ModelType(enum.Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SSD = 4
    SD3 = 5


def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)
        abs_ckpt_dir = os.path.abspath(shared.cmd_opts.ckpt_dir) if shared.cmd_opts.ckpt_dir is not None else None

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)

            return metadata

        self.metadata = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "checkpoint/" + name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.short_title = self.name_for_extra if self.shorthash is None else f'{self.name_for_extra} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging, CLIPModel  # noqa: F401

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    """called once at startup to do various one-time tasks related to SD models"""

    os.makedirs(model_path, exist_ok=True)

    enable_midas_autodownload()
    patch_given_betas()


def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.title for x in checkpoints_list.values()]


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
        expected_sha256 = None
    else:
        model_url = f"{shared.hf_endpoint}/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        expected_sha256 = '6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa'

    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"], hash_prefix=expected_sha256)

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def get_closet_checkpoint_match(search_string):
    if not search_string:
        return None

    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    search_string_without_checksum = re.sub(re_strip_checksum, '', search_string)
    found = sorted([info for info in checkpoints_list.values() if search_string_without_checksum in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint():
    """Raises `FileNotFoundError` if no checkpoints are found."""
    model_checkpoint = shared.opts.sd_model_checkpoint

    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        error_message = "No checkpoints found. When searching for checkpoints, looked at:"
        if shared.cmd_opts.ckpt is not None:
            error_message += f"\n - file {os.path.abspath(shared.cmd_opts.ckpt)}"
        error_message += f"\n - directory {model_path}"
        if shared.cmd_opts.ckpt_dir is not None:
            error_message += f"\n - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}"
        error_message += "Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations."
        raise FileNotFoundError(error_message)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


checkpoint_dict_replacements_sd1 = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_replacements_sd2_turbo = { # Converts SD 2.1 Turbo from SGM to LDM format.
    'conditioner.embedders.0.': 'cond_stage_model.',
}


def transform_checkpoint_dict_key(k, replacements):
    for text, replacement in replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
# def ok32432():
    print("\n🔧 开始状态字典处理流程")
    
    # 初始状态检查
    print("[1/6] 📊 初始状态分析")
    original_keys = set(pl_sd.keys())
    print(f"   原始键数量: {len(pl_sd)}")
    print(f"   前5个键示例: {list(pl_sd.keys())[:5]}")

    # 处理state_dict键
    print("\n[2/6] 🗑️ 清理state_dict条目")
    state_dict_popped = pl_sd.pop("state_dict", pl_sd)
    is_dangerous_pop = state_dict_popped is pl_sd
    print(f"   首次pop操作结果: {'⚠️ 危险操作（返回整个字典）' if is_dangerous_pop else '✅ 安全移除'}")
    
    pl_sd.pop("state_dict", None)
    print(f"   二次清理后存在state_dict键: {'state_dict' in pl_sd}")

    # SD2 Turbo检测
    print("\n[3/6] 🔍 模型架构检测")
    test_key = 'conditioner.embedders.0.model.ln_final.weight'
    key_exists = test_key in pl_sd
    shape_info = "N/A"
    if key_exists:
        shape = pl_sd[test_key].shape
        shape_info = f"{shape} | 维度0长度: {shape[0]}" if len(shape) >0 else "scalar"
    is_sd2_turbo = key_exists and (shape_info.startswith("torch.Size([1024") if key_exists else False)
    
    print(f"   关键检测键 '{test_key}':")
    print(f"   - 存在: {key_exists}")
    print(f"   - 形状: {shape_info}")
    print(f"   SD2 Turbo判定结果: {is_sd2_turbo}")

    # 键名转换
    print("\n[4/6] 🔄 键名转换处理")
    replacement_rules = checkpoint_dict_replacements_sd2_turbo if is_sd2_turbo else checkpoint_dict_replacements_sd1
    print(f"   使用转换规则集: {'SD2 Turbo' if is_sd2_turbo else 'SD1.x'}")
    print(f"   规则数量: {len(replacement_rules)}")
    
    sd = {}
    conversion_stats = {'success': 0, 'skipped': 0, 'duplicate': 0}
    for i, (k, v) in enumerate(pl_sd.items()):
        new_key = transform_checkpoint_dict_key(k, replacement_rules)
        
        # 转换结果跟踪
        if new_key is None:
            conversion_stats['skipped'] +=1
            if i < 5:  # 显示前5个跳过的键示例
                print(f"   🚫 跳过键: {k}")
            continue
                
        if new_key in sd:
            conversion_stats['duplicate'] +=1
            print(f"   ⚠️ 键名冲突: {k} → {new_key} (已存在)")
        else:
            conversion_stats['success'] +=1
            if i < 5:  # 显示前5个成功转换示例
                print(f"   ✅ 转换: {k} → {new_key}")

        sd[new_key] = v

    # 转换统计
    print("\n转换统计:")
    print(f"   成功转换: {conversion_stats['success']}")
    print(f"   跳过条目: {conversion_stats['skipped']}")
    print(f"   重复键名: {conversion_stats['duplicate']}")

    # 字典更新
    print("\n[5/6] ♻️ 更新原始字典")
    before_size = len(pl_sd)
    pl_sd.clear()
    pl_sd.update(sd)
    print(f"   字典大小变化: {before_size} → {len(pl_sd)}")
    print(f"   内存变化: {sys.getsizeof(pl_sd)//1024}KB → {sys.getsizeof(sd)//1024}KB")

    # 最终检查
    print("\n[6/6] ✅ 最终验证")
    new_keys = set(pl_sd.keys())
    added = new_keys - original_keys
    removed = original_keys - new_keys
    print(f"   新增键数量: {len(added)}")
    print(f"   移除键数量: {len(removed)}")
    if added:
        print(f"   示例新增键: {list(added)[:3]}")
    if removed:
        print(f"   示例移除键: {list(removed)[:3]}")

    # return pl_sd


    return pl_sd


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}

        try:
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
             errors.report(f"Error reading metadata from file: {filename}", exc_info=True)

        return res


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):

    print("\n🔍 开始加载检查点文件流程")
    
    # 文件扩展名检测
    print("[1/6] 📂 解析文件信息")
    _, extension = os.path.splitext(checkpoint_file)
    print(f"   检测到文件扩展名: {extension} | 文件路径: {checkpoint_file}")
    
    if extension.lower() == ".safetensors":
        print("\n[2/6] 🔒 安全张量格式处理")
        # 设备选择逻辑
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()
        print(f"   最终设备选择: {device} (map_location={map_location}, weight_load_location={shared.weight_load_location})")
        
        if not shared.opts.disable_mmap_load_safetensors:
            print("   🚀 使用内存映射加载 (mmap enabled)")
            start = time.time()
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
            load_time = time.time() - start
            print(f"   ✅ 加载完成 | 张量数量: {len(pl_sd)} | 耗时: {load_time:.2f}s")
        else:
            print("   ⚠️ 禁用内存映射 (mmap disabled)")
            print("   🐢 完整文件加载到内存...")
            with open(checkpoint_file, 'rb') as f:
                file_size = os.fstat(f.fileno()).st_size
                print(f"   文件大小: {file_size//1024//1024}MB")
                start = time.time()
                pl_sd = safetensors.torch.load(f.read())
                
            print(f"   🔄 迁移张量到设备 {device}")
            tensor_count = 0
            converted_pl_sd = {}
            for k, v in tqdm(pl_sd.items(), desc="转换张量"):
                converted_pl_sd[k] = v.to(device)
                tensor_count +=1
            pl_sd = converted_pl_sd
            print(f"   已处理 {tensor_count} 个张量")
    else:
        print("\n[2/6] ⚠️ 传统格式处理")
        device = map_location or shared.weight_load_location
        print(f"   使用设备映射: {device}")
        print(f"   加载方法: torch.load()")
        # start = time.time()
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)
        # load_time = time.time() - start
        # print(f"   ✅ 加载完成 | 耗时: {load_time:.2f}s")

    # 全局状态打印
    if print_global_state and "global_step" in pl_sd:
        print("\n[3/6] 🌐 全局训练状态")
        print(f"   当前全局训练步数: {pl_sd['global_step']}")
    else:
        print("\n[3/6] ⚠️ 未找到全局训练步数信息")

    # 状态字典提取
    print("\n[4/6] 📖 提取状态字典")
    sd = get_state_dict_from_checkpoint(pl_sd)
    print(f"   获取到 {len(sd)} 个关键参数")
    
    # 内存统计
    # print("\n[5/6] 💾 内存使用情况")
    # print(f"   CPU内存占用: {psutil.Process().memory_info().rss//1024//1024}MB")
    # if devices.cuda_available:
    #     print(f"   GPU内存占用: {torch.cuda.memory_allocated()//1024//1024}MB")
    
    print("\n[6/6] 🎉 检查点加载流程完成")

    return sd


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):

    print("\n🔍 开始模型权重加载流程")
    
    # 计算模型哈希
    print("[1/5] 🔢 计算模型哈希值...")
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")
    print(f"   ✅ 哈希计算完成 | 哈希值: {sd_model_hash}")
    # print(f"   ⏱️ 哈希计算耗时: {timer.get_last_time('calculate hash'):.2f}s")

    # 检查缓存是否存在
    print("\n[2/5] 📦 检查模型缓存")
    print(f"   当前缓存条目数: {len(checkpoints_loaded)}")
    
    if checkpoint_info in checkpoints_loaded:
        # 缓存命中处理
        print(f"   🎯 缓存命中 [{sd_model_hash}]")
        print("   🔄 更新缓存位置为最近使用")
        checkpoints_loaded.move_to_end(checkpoint_info)
        print(f"   最新缓存顺序: {list(checkpoints_loaded.keys())[-1].shorthash}")
        return checkpoints_loaded[checkpoint_info]

    # 缓存未命中处理
    print(f"   ❌ 缓存未命中 [{sd_model_hash}]")
    print(f"\n[3/5] ⬇️ 从磁盘加载权重文件")
    print(f"   文件路径: {checkpoint_info.filename}")
    print(f"   文件大小: {os.path.getsize(checkpoint_info.filename)//1024//1024}MB")
    
    # 加载权重文件
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load weights from disk")
    
    print("\n[4/5] ✅ 权重加载完成")
    print(f"   加载张量数量: {len(res)}")
    # print(f"   ⏱️ 磁盘加载耗时: {timer.get_last_time('load weights from disk'):.2f}s")

    # 更新缓存（假设后续有添加操作）
    print("\n[5/5] 💾 更新模型缓存")
    # 此处假设有 checkpoints_loaded[checkpoint_info] = res 操作
    print(f"   新缓存条目数: {len(checkpoints_loaded)+1}")
    print(f"   🏷️ 新增缓存标识: {sd_model_hash}")



    return res


class SkipWritingToConfig:
    """This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight."""

    skip = False
    previous = None

    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous


def check_fp8(model):
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        enable_fp8 = False
    elif shared.opts.fp8_storage == "Enable":
        enable_fp8 = True
    elif getattr(model, "is_sdxl", False) and shared.opts.fp8_storage == "Enable for SDXL":
        enable_fp8 = True
    else:
        enable_fp8 = False
    return enable_fp8


def set_model_type(model, state_dict):
    model.is_sd1 = False
    model.is_sd2 = False
    model.is_sdxl = False
    model.is_ssd = False
    model.is_sd3 = False

    if "model.diffusion_model.x_embedder.proj.weight" in state_dict:
        model.is_sd3 = True
        model.model_type = ModelType.SD3
    elif hasattr(model, 'conditioner'):
        model.is_sdxl = True

        if 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in state_dict.keys():
            model.is_ssd = True
            model.model_type = ModelType.SSD
        else:
            model.model_type = ModelType.SDXL
    elif hasattr(model.cond_stage_model, 'model'):
        model.is_sd2 = True
        model.model_type = ModelType.SD2
    else:
        model.is_sd1 = True
        model.model_type = ModelType.SD1


def set_model_fields(model):
    if not hasattr(model, 'latent_channels'):
        model.latent_channels = 4


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if devices.fp8:
        # prevent model to load state dict in fp8
        model.half()

    if not SkipWritingToConfig.skip:
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title

    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    set_model_type(model, state_dict)
    set_model_fields(model)

    if model.is_sdxl:
        sd_models_xl.extend_sdxl(model)

    if model.is_ssd:
        sd_hijack.model_hijack.convert_sdxl_to_ssd(model)

    if shared.opts.sd_checkpoint_cache > 0:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = state_dict.copy()

    if hasattr(model, "before_load_weights"):
        model.before_load_weights(state_dict)

    model.load_state_dict(state_dict, strict=False)
    timer.record("apply weights to model")

    if hasattr(model, "after_load_weights"):
        model.after_load_weights(state_dict)

    del state_dict

    # Set is_sdxl_inpaint flag.
    # Checks Unet structure to detect inpaint model. The inpaint model's
    # checkpoint state_dict does not contain the key
    # 'diffusion_model.input_blocks.0.0.weight'.
    diffusion_model_input = model.model.state_dict().get(
        'diffusion_model.input_blocks.0.0.weight'
    )
    model.is_sdxl_inpaint = (
        model.is_sdxl and
        diffusion_model_input is not None and
        diffusion_model_input.shape[1] == 9
    )

    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("apply channels_last")

    if shared.cmd_opts.no_half:
        model.float()
        model.alphas_cumprod_original = model.alphas_cumprod
        devices.dtype_unet = torch.float32
        assert shared.cmd_opts.precision != "half", "Cannot use --precision half with --no-half"
        timer.record("apply float()")
    else:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.cmd_opts.no_half_vae:
            model.first_stage_model = None
        # with --upcast-sampling, don't convert the depth model weights to float16
        if shared.cmd_opts.upcast_sampling and depth_model:
            model.depth_model = None

        alphas_cumprod = model.alphas_cumprod
        model.alphas_cumprod = None
        model.half()
        model.alphas_cumprod = alphas_cumprod
        model.alphas_cumprod_original = alphas_cumprod
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

        devices.dtype_unet = torch.float16
        timer.record("apply half()")

    apply_alpha_schedule_override(model)

    for module in model.modules():
        if hasattr(module, 'fp16_weight'):
            del module.fp16_weight
        if hasattr(module, 'fp16_bias'):
            del module.fp16_bias

    if check_fp8(model):
        devices.fp8 = True
        first_stage = model.first_stage_model
        model.first_stage_model = None
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if shared.opts.cache_fp16_weight:
                    module.fp16_weight = module.weight.data.clone().cpu().half()
                    if module.bias is not None:
                        module.fp16_bias = module.bias.data.clone().cpu().half()
                module.to(torch.float8_e4m3fn)
        model.first_stage_model = first_stage
        timer.record("apply fp8")
    else:
        devices.fp8 = False

    devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    model.first_stage_model.to(devices.dtype_vae)
    timer.record("apply dtype to VAE")

    # clean up cache if limit is reached
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)

    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device)  # fix for training

    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("load VAE")


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """

    midas_path = os.path.join(paths.models_path, 'midas')

    # stable-diffusion-stability-ai hard-codes the midas model path to
    # a location that differs from where other scripts using this model look.
    # HACK: Overriding the path here.
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                os.mkdir(midas_path)

            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")

        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def patch_given_betas():
    import ldm.models.diffusion.ddpm

    def patched_register_schedule(*args, **kwargs):
        """a modified version of register_schedule function that converts plain list from Omegaconf into numpy"""

        if isinstance(args[1], ListConfig):
            args = (args[0], np.array(args[1]), *args[2:])

        original_register_schedule(*args, **kwargs)

    original_register_schedule = patches.patch(__name__, ldm.models.diffusion.ddpm.DDPM, 'register_schedule', patched_register_schedule)


def repair_config(sd_config, state_dict=None):
    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    if hasattr(sd_config.model.params, 'unet_config'):
        if shared.cmd_opts.no_half:
            sd_config.model.params.unet_config.params.use_fp16 = False
        elif shared.cmd_opts.upcast_sampling or shared.cmd_opts.precision == "half":
            sd_config.model.params.unet_config.params.use_fp16 = True

    if hasattr(sd_config.model.params, 'first_stage_config'):
        if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
            sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"

    # For UnCLIP-L, override the hardcoded karlo directory
    if hasattr(sd_config.model.params, "noise_aug_config") and hasattr(sd_config.model.params.noise_aug_config.params, "clip_stats_path"):
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)

    # Do not use checkpoint for inference.
    # This helps prevent extra performance overhead on checking parameters.
    # The perf overhead is about 100ms/it on 4090 for SDXL.
    if hasattr(sd_config.model.params, "network_config"):
        sd_config.model.params.network_config.params.use_checkpoint = False
    if hasattr(sd_config.model.params, "unet_config"):
        sd_config.model.params.unet_config.params.use_checkpoint = False



def rescale_zero_terminal_snr_abar(alphas_cumprod):
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return alphas_bar


def apply_alpha_schedule_override(sd_model, p=None):
    """
    Applies an override to the alpha schedule of the model according to settings.
    - downcasts the alpha schedule to half precision
    - rescales the alpha schedule to have zero terminal SNR
    """

    if not hasattr(sd_model, 'alphas_cumprod') or not hasattr(sd_model, 'alphas_cumprod_original'):
        return

    sd_model.alphas_cumprod = sd_model.alphas_cumprod_original.to(shared.device)

    if opts.use_downcasted_alpha_bar:
        if p is not None:
            p.extra_generation_params['Downcast alphas_cumprod'] = opts.use_downcasted_alpha_bar
        sd_model.alphas_cumprod = sd_model.alphas_cumprod.half().to(shared.device)

    if opts.sd_noise_schedule == "Zero Terminal SNR":
        if p is not None:
            p.extra_generation_params['Noise Schedule'] = opts.sd_noise_schedule
        sd_model.alphas_cumprod = rescale_zero_terminal_snr_abar(sd_model.alphas_cumprod).to(shared.device)


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight = 'conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight = 'conditioner.embedders.0.model.ln_final.weight'


class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.loaded_sd_models = []
        self.was_loaded_at_least_once = False
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.was_loaded_at_least_once:
            return self.sd_model

        if self.sd_model is None:
            with self.lock:
                if self.sd_model is not None or self.was_loaded_at_least_once:
                    return self.sd_model

                try:
                    load_model()

                except Exception as e:
                    errors.display(e, "loading stable diffusion model", full_traceback=True)
                    print("", file=sys.stderr)
                    print("Stable diffusion model failed to load", file=sys.stderr)
                    self.sd_model = None

        return self.sd_model

    def set_sd_model(self, v, already_loaded=False):
        self.sd_model = v
        if already_loaded:
            sd_vae.base_vae = getattr(v, "base_vae", None)
            sd_vae.loaded_vae_file = getattr(v, "loaded_vae_file", None)
            sd_vae.checkpoint_info = v.sd_checkpoint_info

        try:
            self.loaded_sd_models.remove(v)
        except ValueError:
            pass

        if v is not None:
            self.loaded_sd_models.insert(0, v)


model_data = SdModelData()


def get_empty_cond(sd_model):

    p = processing.StableDiffusionProcessingTxt2Img()
    extra_networks.activate(p, {})

    if hasattr(sd_model, 'get_learned_conditioning'):
        d = sd_model.get_learned_conditioning([""])
    else:
        d = sd_model.cond_stage_model([""])

    if isinstance(d, dict):
        d = d['crossattn']

    return d


def send_model_to_cpu(m):
    if m is not None:
        if m.lowvram:
            lowvram.send_everything_to_cpu()
        else:
            m.to(devices.cpu)

    devices.torch_gc()


def model_target_device(m):
    if lowvram.is_needed(m):
        return devices.cpu
    else:
        return devices.device


def send_model_to_device(m):
    lowvram.apply(m)

    if not m.lowvram:
        m.to(shared.device)


def send_model_to_trash(m):
    m.to(device="meta")
    devices.torch_gc()


def instantiate_from_config(config, state_dict=None):
    constructor = get_obj_from_str(config["target"])

    params = {**config.get("params", {})}

    if state_dict and "state_dict" in params and params["state_dict"] is None:
        params["state_dict"] = state_dict

    return constructor(**params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model(checkpoint_info=None, already_loaded_state_dict=None):

    # 模块导入跟踪
    print("[1/15] ⚙️ 正在导入sd_hijack模块...")
    from modules import sd_hijack
    print(f"   ✅ 模块导入完成 | 可用方法: {dir(sd_hijack)[:3]}...")

    # 检查点选择逻辑
    print("\n[2/15] 🔍 检查点选择流程")
    print(f"   输入checkpoint_info状态: {'已提供' if checkpoint_info else '未提供'}")
    checkpoint_info = checkpoint_info or select_checkpoint()
    print(f"   🎯 最终使用检查点: {getattr(checkpoint_info, 'filename', '未知')}")

    # 计时器初始化
    print("\n[3/15] ⏱️ 初始化性能计时器")
    timer = Timer()
    # print(f"   计时器精度: {timer.precision} | 当前记录数: {len(timer.records)}")

    # 模型卸载流程
    print("\n[4/15] 🗑️ 清理现有模型")
    if model_data.sd_model:
        model_size = sum(p.numel() for p in model_data.sd_model.parameters()) 
        print(f"   检测到现有模型 | 参数量: {model_size//1e6}M | 开始卸载...")
        send_model_to_trash(model_data.sd_model)
        print("   🚮 模型已移至回收站")
        model_data.sd_model = None
        print("   内存引用已清除")
        devices.torch_gc()
        freed_mem = devices.get_freed_memory()
        print(f"   🔄 显存回收完成 | 释放: {freed_mem//1e6}MB")
    else:
        print("   ⏭️ 无加载模型，跳过卸载步骤")
    
    timer.record("unload existing model")
    # print(f"   ⏱️ 阶段耗时: {timer.get_last_time():.2f}s")

    # 状态字典加载
    print("\n[5/15] 📦 加载模型权重")
    if already_loaded_state_dict is not None:
        print("   🔄 使用预加载状态字典")
        print(f"   字典键数量: {len(already_loaded_state_dict.keys())}")
        state_dict = already_loaded_state_dict
    else:
        print("   ⬇️ 从检查点加载新状态字典")
        # print(f"   检查点路径: {checkpoint_info.path}")
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
        print(f"   ✅ 加载完成 | 大小: {len(state_dict)//1e6}MB")
    
    # 配置检查流程
    print("\n[6/15] 🔧 配置验证与修复")
    print("   开始查找匹配的模型配置...")
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    print(f"   🛠️ 匹配到的配置文件: {checkpoint_config}")
    
    # CLIP权重检测
    print("\n[7/15] 🔎 CLIP权重检测")
    clip_candidates = [sd1_clip_weight, sd2_clip_weight, sdxl_clip_weight, sdxl_refiner_clip_weight]
    found_clips = [x for x in clip_candidates if x in state_dict]
    print(f"   检测CLIP类型: {found_clips[0] if found_clips else '未找到'}") 
    clip_is_included_into_sd = bool(found_clips)
    print(f"   CLIP包含状态: {'✅ 包含' if clip_is_included_into_sd else '❌ 未包含'}")

    timer.record("find config")
    # print(f"   ⏱️ 阶段耗时: {timer.get_last_time():.2f}s")

    # 配置加载与修复
    print("\n[8/15] 📄 加载OmegaConf配置")
    print(f"   配置文件路径: {checkpoint_config}")
    try:
        sd_config = OmegaConf.load(checkpoint_config)
        print(f"   ✅ 配置加载成功 | 结构: {len(sd_config)}个节点")
    except Exception as e:
        print(f"   ❌ 配置加载失败: {str(e)}")
        raise
    
    print("   开始修复配置兼容性问题...")
    repair_config(sd_config, state_dict)
    print("   🔧 配置修复完成")

    timer.record("load config")
    # print(f"   ⏱️ 阶段耗时: {timer.get_last_time():.2f}s")

    # 最终输出
    print("\n[9/15] 🚀 准备创建模型实例")
    print(f"   最终使用的配置文件: {checkpoint_config}")
    print(f"   总耗时统计: {timer.summary()}")

# def okewq23432():
    print("\n🔥 开始模型初始化流程")
    sd_model = None
    print("[1/8] 🛠️ 初始化sd_model为None")

    try:
        print("\n[2/8] ⚡ 尝试快速初始化模型")
        print(f"   配置参数 - clip_is_included: {clip_is_included_into_sd}")
        print(f"   命令行选项 - no_download_clip: {shared.cmd_opts.do_not_download_clip}")
        
        # 上下文管理器状态跟踪
        with sd_disable_initialization.DisableInitialization(
            disable_clip=clip_is_included_into_sd or shared.cmd_opts.do_not_download_clip
        ) as ctx1:
            # print(f"   🔒 初始化限制已启用 | 禁用CLIP: {ctx1.disable_clip}")
            
            with sd_disable_initialization.InitializeOnMeta() as ctx2:
                print("   💽 进入元设备初始化上下文")
                print(f"   模型配置结构: {len(sd_config.model)}个组件")
                
                # 模型实例化
                sd_model = instantiate_from_config(sd_config.model, state_dict)
                print(f"   ✅ 快速初始化成功 | 模型类: {type(sd_model).__name__}")

    except Exception as e:
        print(f"\n⚠️ 快速初始化失败！错误类型: {type(e).__name__}")
        errors.display(e, "creating model quickly", full_traceback=True)
        print("   🚨 进入慢速初始化流程...")

    if sd_model is None:
        print("\n[3/8] 🐢 启动慢速初始化方法")
        try:
            with sd_disable_initialization.InitializeOnMeta() as ctx:
                print("   🔄 重新尝试基础初始化")
                sd_model = instantiate_from_config(sd_config.model, state_dict)
                print(f"   ✅ 慢速初始化成功 | 内存占用: {torch.cuda.memory_allocated()//1e6}MB")
        except Exception as e:
            print("   ❌ 双重初始化失败！终止流程")
            raise
    else:
        print("   ✔️ 快速初始化流程完成")

    # 配置关联
    print("\n[4/8] 📌 绑定配置信息")
    sd_model.used_config = checkpoint_config
    print(f"   关联配置文件: {os.path.basename(checkpoint_config)}")
    timer.record("create model")
    # print(f"   ⏱️ 模型创建阶段耗时: {timer.get_last_time('create model'):.2f}s")

    # 精度配置
    print("\n[5/8] 🎚️ 设置权重精度")
    if shared.cmd_opts.no_half:
        weight_dtype_conversion = None
        print("   🚫 禁用半精度转换 (no_half=True)")
    else:
        weight_dtype_conversion = {
            'first_stage_model': None,
            'alphas_cumprod': None,
            '': torch.float16,
        }
        print("   🎛️ 混合精度配置:")
        print(f"    - 首阶段模型: 保持原精度")
        print(f"    - 累积参数: 保持原精度") 
        print(f"    - 默认转换: float16")

    # 权重加载
    print("\n[6/8] ⬇️ 加载模型权重")
    target_device = model_target_device(sd_model)
    print(f"   目标设备: {target_device}")
    
    with sd_disable_initialization.LoadStateDictOnMeta(
        state_dict, 
        device=target_device,
        weight_dtype_conversion=weight_dtype_conversion
    ) as loader:
        print(f"   🔧 权重加载器配置:")
        print(f"    - 状态字典条目: {len(state_dict)}")
        # print(f"    - 转换规则: {loader.weight_dtype_conversion}")
        
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
        print(f"   ✅ 权重加载完成 | 峰值内存: {torch.cuda.max_memory_allocated()//1e6}MB")
    
    timer.record("load weights from state dict")
    # print(f"   ⏱️ 权重加载耗时: {timer.get_last_time('load weights from state dict'):.2f}s")

    # 设备转移
    print("\n[7/8] 🚚 迁移模型至设备")
    prev_mem = torch.cuda.memory_allocated()
    send_model_to_device(sd_model)
    curr_mem = torch.cuda.memory_allocated()
    print(f"   💾 显存变化: {curr_mem//1e6}MB (+{(curr_mem-prev_mem)//1e6}MB)")
    timer.record("move model to device")
    # print(f"   ⏱️ 迁移耗时: {timer.get_last_time('move model to device'):.2f}s")

    # 模型劫持
    print("\n[8/8] 🎭 执行模型劫持")
    sd_hijack.model_hijack.hijack(sd_model)
    print("   🔗 劫持操作已完成")
    # print(f"\n🎉 模型初始化全流程完成！总耗时: {timer.total():.2f}s")

# def ok32324():
    print("\n🔧 开始模型后处理流程")
    
    # 记录劫持完成时间
    timer.record("hijack")
    # print(f"[1/9] ⏱️ 模型劫持完成计时 | 当前阶段耗时: {timer.get_last_time('hijack'):.2f}s")

    # 设置评估模式
    print("\n[2/9] 🧪 设置模型为评估模式")
    prev_training_mode = sd_model.training
    sd_model.eval()
    print(f"   训练模式变更: {prev_training_mode} → {sd_model.training}")

    # 存储模型引用
    print("\n[3/9] 💾 保存模型到数据管理器")
    prev_model_hash = hash(model_data.sd_model) if model_data.sd_model else None
    model_data.set_sd_model(sd_model)
    curr_model_hash = hash(sd_model)
    print(f"   模型引用变更: {prev_model_hash or 'None'} → {curr_model_hash}")

    # 更新加载状态
    print("\n[4/9] ✅ 标记模型加载状态")
    print(f"   先前加载状态: {model_data.was_loaded_at_least_once}")
    model_data.was_loaded_at_least_once = True
    print(f"   更新后状态: {model_data.was_loaded_at_least_once}")

    # 加载文本嵌入
    print("\n[5/9] 📥 重新加载文本反转嵌入")
    # print(f"   强制重载参数: {force_reload}")
    before_embeddings = len(sd_hijack.model_hijack.embedding_db.word_embeddings)
    
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(
        force_reload=True
    )
    
    after_embeddings = len(sd_hijack.model_hijack.embedding_db.word_embeddings)
    timer.record("load textual inversion embeddings")
    print(f"   嵌入数量变化: {before_embeddings} → {after_embeddings}")
    # print(f"   ⏱️ 嵌入加载耗时: {timer.get_last_time('load textual inversion embeddings'):.2f}s")

    # 执行回调函数
    print("\n[6/9] 📞 触发模型加载回调")
    # callback_count = len(script_callbacks.model_loaded_callback.callbacks)
    # print(f"   注册回调数量: {callback_count}")
    script_callbacks.model_loaded_callback(sd_model)
    timer.record("scripts callbacks")
    # print(f"   ⏱️ 回调执行耗时: {timer.get_last_time('scripts callbacks'):.2f}s")

    # 计算空提示条件
    print("\n[7/9] 🌀 计算空提示条件")
    with devices.autocast() as amp_ctx, torch.no_grad() as no_grad_ctx:
        print(f"   进入混合精度上下文: {amp_ctx.enabled}")
        print(f"   梯度计算状态: {not no_grad_ctx.enabled}")
        
        empty_cond = get_empty_cond(sd_model)
        sd_model.cond_stage_model_empty_prompt = empty_cond

        
    timer.record("calculate empty prompt")

    return sd_model


def reuse_model_from_already_loaded(sd_model, checkpoint_info, timer):
    """
    Checks if the desired checkpoint from checkpoint_info is not already loaded in model_data.loaded_sd_models.
    If it is loaded, returns that (moving it to GPU if necessary, and moving the currently loadded model to CPU if necessary).
    If not, returns the model that can be used to load weights from checkpoint_info's file.
    If no such model exists, returns None.
    Additionally deletes loaded models that are over the limit set in settings (sd_checkpoints_limit).
    """

    if sd_model is not None and sd_model.sd_checkpoint_info.filename == checkpoint_info.filename:
        return sd_model

    if shared.opts.sd_checkpoints_keep_in_cpu:
        send_model_to_cpu(sd_model)
        timer.record("send model to cpu")

    already_loaded = None
    for i in reversed(range(len(model_data.loaded_sd_models))):
        loaded_model = model_data.loaded_sd_models[i]
        if loaded_model.sd_checkpoint_info.filename == checkpoint_info.filename:
            already_loaded = loaded_model
            continue

        if len(model_data.loaded_sd_models) > shared.opts.sd_checkpoints_limit > 0:
            print(f"Unloading model {len(model_data.loaded_sd_models)} over the limit of {shared.opts.sd_checkpoints_limit}: {loaded_model.sd_checkpoint_info.title}")
            del model_data.loaded_sd_models[i]
            send_model_to_trash(loaded_model)
            timer.record("send model to trash")

    if already_loaded is not None:
        send_model_to_device(already_loaded)
        timer.record("send model to device")

        model_data.set_sd_model(already_loaded, already_loaded=True)

        if not SkipWritingToConfig.skip:
            shared.opts.data["sd_model_checkpoint"] = already_loaded.sd_checkpoint_info.title
            shared.opts.data["sd_checkpoint_hash"] = already_loaded.sd_checkpoint_info.sha256

        print(f"Using already loaded model {already_loaded.sd_checkpoint_info.title}: done in {timer.summary()}")
        sd_vae.reload_vae_weights(already_loaded)
        return model_data.sd_model
    elif shared.opts.sd_checkpoints_limit > 1 and len(model_data.loaded_sd_models) < shared.opts.sd_checkpoints_limit:
        print(f"Loading model {checkpoint_info.title} ({len(model_data.loaded_sd_models) + 1} out of {shared.opts.sd_checkpoints_limit})")

        model_data.sd_model = None
        load_model(checkpoint_info)
        return model_data.sd_model
    elif len(model_data.loaded_sd_models) > 0:
        sd_model = model_data.loaded_sd_models.pop()
        model_data.sd_model = sd_model

        sd_vae.base_vae = getattr(sd_model, "base_vae", None)
        sd_vae.loaded_vae_file = getattr(sd_model, "loaded_vae_file", None)
        sd_vae.checkpoint_info = sd_model.sd_checkpoint_info

        print(f"Reusing loaded model {sd_model.sd_checkpoint_info.title} to load {checkpoint_info.title}")
        return sd_model
    else:
        return None


def reload_model_weights(sd_model=None, info=None, forced_reload=False):
    checkpoint_info = info or select_checkpoint()

    timer = Timer()

    if not sd_model:
        sd_model = model_data.sd_model

    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = sd_model.sd_checkpoint_info
        if check_fp8(sd_model) != devices.fp8:
            # load from state dict again to prevent extra numerical errors
            forced_reload = True
        elif sd_model.sd_model_checkpoint == checkpoint_info.filename and not forced_reload:
            return sd_model

    sd_model = reuse_model_from_already_loaded(sd_model, checkpoint_info, timer)
    if not forced_reload and sd_model is not None and sd_model.sd_checkpoint_info.filename == checkpoint_info.filename:
        return sd_model

    if sd_model is not None:
        sd_unet.apply_unet("None")
        send_model_to_cpu(sd_model)
        sd_hijack.model_hijack.undo_hijack(sd_model)

    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)

    timer.record("find config")

    if sd_model is None or checkpoint_config != sd_model.used_config:
        if sd_model is not None:
            send_model_to_trash(sd_model)

        load_model(checkpoint_info, already_loaded_state_dict=state_dict)
        return model_data.sd_model

    try:
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception:
        print("Failed to load checkpoint, restoring previous")
        load_model_weights(sd_model, current_checkpoint_info, None, timer)
        raise
    finally:
        sd_hijack.model_hijack.hijack(sd_model)
        timer.record("hijack")

        if not sd_model.lowvram:
            sd_model.to(devices.device)
            timer.record("move model to device")

        script_callbacks.model_loaded_callback(sd_model)
        timer.record("script callbacks")

    print(f"Weights loaded in {timer.summary()}.")

    model_data.set_sd_model(sd_model)
    sd_unet.apply_unet()

    return sd_model


def unload_model_weights(sd_model=None, info=None):
    send_model_to_cpu(sd_model or shared.sd_model)

    return sd_model


def apply_token_merging(sd_model, token_merging_ratio):
    """
    Applies speed and memory optimizations from tomesd.
    """

    current_token_merging_ratio = getattr(sd_model, 'applied_token_merged_ratio', 0)

    if current_token_merging_ratio == token_merging_ratio:
        return

    if current_token_merging_ratio > 0:
        tomesd.remove_patch(sd_model)

    if token_merging_ratio > 0:
        tomesd.apply_patch(
            sd_model,
            ratio=token_merging_ratio,
            use_rand=False,  # can cause issues with some samplers
            merge_attn=True,
            merge_crossattn=False,
            merge_mlp=False
        )

    sd_model.applied_token_merged_ratio = token_merging_ratio
