# End to end workflow to run a Multimodal model

### Support Matrix

The following multimodal model is supported in tensorrtllm_backend:

- [BLIP2](#blip2)
- [CogVLM](#cogvlm)
- [Deplot](#deplot)
- [Fuyu](#fuyu)
- [Kosmos-2](#kosmos-2)
- [LLaVA, LLaVa-NeXT and VILA](#llava-llava-next-and-vila)
- [NeVA](#neva)
- [Nougat](#nougat)
- [Phi-3-vision](#phi-3-vision)
- [Video NeVA](#video-neva)
- [Enabling tensor parallelism for multi-GPU](#enabling-tensor-parallelism-for-multi-gpu)

First, clone the [tensorrt-demo](https://github.com/datacrunch-research/tensorrt-demo) repository:

```bash
git clone https://github.com/datacrunch-research/tensorrt-demo.git
cd tensorrt-demo
export tensorrt_demo_dir=`pwd`
```

Then, clone the [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend) repository:

```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
export tensorrtllm_backend_dir=`pwd`
git lfs install
```

For multimodal models, ensure we are using the latest version of [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend)

```bash
git fetch --all
git submodule update --init --recursive # --remote for latest upstream
```

Copy **triton_model_repo** directory from tensorrt-demo to tensorrtllm_backend: 

```bash
cp -r ${tensorrt_demo_dir}/triton_model_repo ${tensorrtllm_backend_dir}/
```

### Infrastructure Changes

[TensorRT-LLM 0.11.0 Release notes](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.11.0)

- Base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.05-py3`.
- Base Docker image for TensorRT-LLM backend is updated to `nvcr.io/nvidia/tritonserver:24.05-py3`.

Start **trt-llm-triton** docker with the latest **TensorRT-LLM backend**:

```bash
export models_dir=$HOME/models
docker run -it -d --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus all -v ${tensorrtllm_backend_dir}:/tensorrtllm_backend  -v $HOME/models:/models -v ${tensorrt_demo_dir}:/root/tensorrt-demo --name triton_server_v0_11_0 nvcr.io/nvidia/tritonserver:24.05-py3 bash

docker exec -it triton_server_v0_11_0 /bin/bash
```

Within the container (see [Installing on Linux](https://github.com/NVIDIA/TensorRT-LLM/blob/a681853d3803ee5893307e812530b5e7004bb6e1/docs/source/installation/linux.md#L4)):

```bash
# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

# Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.
# If you want to install the stable version (corresponding to the release branch), please
# remove the `--pre` option.
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Check installation
python3 -c "import tensorrt_llm"
```
#### Build TensorRT-LLM

Build TensorRT-LLM from the source.

```bash
# TensorRT-LLM is required for generating engines. You can skip this step if
# you already have the package installed. If you are generating engines within
# the Triton container, you have to install the TRT-LLM package.
(cd tensorrt_llm &&
    bash docker/common/install_cmake.sh &&
    export PATH=/usr/local/cmake/bin:$PATH &&
    python3 ./scripts/build_wheel.py --trt_root="/usr/local/tensorrt" &&
    pip3 install ./build/tensorrt_llm*.whl)
```



## CogVLM

Currently, CogVLM only support **bfloat16** precision and doesn't support `remove_input_padding` feature.

Set model params. Modify *model_type* and *model_name* to point to your model, and modify the model dtype/tp_size/max_batch_size etc... based on your requirements:

```bash
export models_dir=/models/trt # trt dir is a temporal solution

export model_type=cogvlm
export model_name=cogvlm-chat-hf
export tokenizer_name=vicuna-7b-v1.5

export model_dtype=bfloat16
export model_tp_size=1

export max_batch_size=48
export max_input_len=2048
export max_seq_len=3076

export model_path=${models_dir}/${model_name}
export trt_model_path=${models_dir}/${model_name}-trt-ckpt
export trt_engine_path=${models_dir}/${model_name}-trt-engine
```

### 1. Build the engine

Download Huggingface weights

```bash
  git clone https://huggingface.co/THUDM/${model_name} ${models_dir}/${model_name}
  git clone https://huggingface.co/lmsys/${tokenizer_name}  ${models_dir}/${tokenizer_name}
```
Because currently onnx doesn't support `xops.memory_efficient_attention`, we need to modify some source code of the huggingface CogVLM

```bash
cd ${models_dir}/${model_name}
sed -i '4s/.*//;40s/.*/        out = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()/;41s/.*//;42s/.*//' visual.py   # It will replace memory_efficient_attention with some basic ops
```

Convert hugging face checkpoint to TRT checkpoint (using scripts in `examples/cogvlm/convert_checkpoint.py`). CogVLM uses a Vit encoder as LLM encoder and a modified Llama as decoder:

```bash	
cd /tensorrtllm_backend
cd ./tensorrt_llm/examples/${model_type}

python3 convert_checkpoint.py \
    --model_dir ${model_path} \
    --dtype ${model_dtype} \
    --tp_size ${model_tp_size} \
    --output_dir ${trt_model_path} \
    --use_prompt_tuning
```

Build TensorRT-LLM engines

```bash
trtllm-build --checkpoint_dir ${trt_model_path} \
--output_dir ${trt_engine_path} \
--gemm_plugin bfloat16 \
--gpt_attention_plugin bfloat16 \
--context_fmha_fp32_acc enable \
--remove_input_padding disable \
--max_batch_size 48 \
--max_input_len 2048 \
--max_seq_len 3076 \
--paged_kv_cache disable \
--enable_xqa disable \
--bert_attention_plugin disable \
--moe_plugin disable \
--max_multimodal_len 61440 # 48 (max_batch_size) * 1280 (max_num_visual_features)
```

Generate TensorRT engines for visual components and combine everything into final pipeline.

```bash
python examples/multimodal/build_visual_engine.py --model_type ${model_type} --model_path ${model_path} --max_batch_size 48
```

```bash
python3 run.py \
--max_new_tokens 1000 \
--input_text " [INST] please describe this image in detail [/INST] " \
--hf_model_dir ${models_dir}/${TOKENIZER_NAME} \
--visual_engine_dir ${models_dir}/vision_encoder \
--llm_engine_dir ${trt_engine_path} \
--batch_size 1 \
--top_p 0.4 \
--top_k 1 \
--temperature 0.2 \
--repetition_penalty 1.2
```

### 2. Prepare Tritonserver configs 

```bash
export visual_engine_path=${models_dir}/vision_encoder
export hf_model_path=${models_dir}/${TOKENIZER_NAME}
```



```bash
cp all_models/inflight_batcher_llm/ multimodal_ifb -r

python3 tools/fill_template.py -i multimodal_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${trt_engine_path},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False

python3 tools/fill_template.py -i multimodal_ifb/preprocessing/config.pbtxt tokenizer_dir:${hf_model_path},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${visual_engine_path},engine_dir:${trt_engine_path}

python3 tools/fill_template.py -i multimodal_ifb/postprocessing/config.pbtxt tokenizer_dir:${hf_model_path},triton_max_batch_size:8,postprocessing_instance_count:1

python3 tools/fill_template.py -i multimodal_ifb/ensemble/config.pbtxt triton_max_batch_size:8

python3 tools/fill_template.py -i multimodal_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm
```

**BUG:** [TYPE_BF16](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)

### 3. Start Triton server

# Benchmarking