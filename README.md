# End to end workflow to run LLaMA

Our example targets **LLaMA 3** architecture, but previous versions are also supported. LLaMA 3.1 isn’t yet added.

The document showcases how to build and run a LLaMA model in TensorRT-LLM, on both single GPU and single node multi-GPU with **tensor parallelism**. In future, add necessary documentation to explain multi-node multi-GPU configurations.

We explore the steps and considerations while building the TensorRT engines, which offers an extensive set of hyperparameters for constructing the final engine artifact. For example, each model parallelism layout (pipeline, tensor, sequence) creates a new engine. 

1. First, clone the [tensorrt-demo](https://github.com/datacrunch-research/tensorrt-demo) repository:

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

Ensure that the version of [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend) is set to [r24.04](https://github.com/triton-inference-server/tensorrtllm_backend/tree/r24.04):

```bash
git fetch --all
git checkout -b r24.04 -t origin/r24.04

git submodule update --init --recursive
```

Copy **triton_model_repo** directory from tensorrt-demo to tensorrtllm_backend: 

```bash
cp -r ${tensorrt_demo_dir}/triton_model_repo ${tensorrtllm_backend_dir}/
```

Start **trt-llm-triton** docker **(tensorrt-llm v0.9.0**) -> nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3:

```bash
export models_dir=$HOME/models
docker run -it -d --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus all -v ${tensorrtllm_backend_dir}:/tensorrtllm_backend  -v $HOME/models:/models -v ${tensorrt_demo_dir}:/root/tensorrt-demo --name triton_server nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3 bash

docker exec -it triton_server /bin/bash
```

Set model params. Modify *model_type* and *model_name* to point to your model, and modify the model dtype/tp_size/max_batch_size etc... based on your requirements:

```bash
export models_dir=/models/trt # trt dir is a temporal solution

export model_type=llama
export model_name=Meta-Llama-3-8B-Instruct

export model_dtype=float16
export model_tp_size=1

export max_batch_size=256
export max_input_len=2048
export max_output_len=1024

export model_path=${models_dir}/${model_name}
export trt_model_path=${models_dir}/${model_name}-trt-ckpt
export trt_engine_path=${models_dir}/${model_name}-trt-engine
```

Convert hugging face checkpoint to TRT checkpoint:

```bash
cd /tensorrtllm_backend
cd ./tensorrt_llm/examples/${model_type}

python3 convert_checkpoint.py \
    --model_dir ${model_path} \
    --dtype ${model_dtype} \
    --tp_size ${model_tp_size} \
    --output_dir ${trt_model_path} \
```

Compile TRT checkpoint to TRT engine:

```bash     
# Choose to enable/disable chunked prompt
export CHUNKED_PROMPT_FLAGS=
export CHUNKED_PROMPT_FLAGS="--context_fmha=enable --use_paged_context_fmha=enable --context_fmha_fp32_acc=enable --multi_block_mode=enable"

trtllm-build \
    --checkpoint_dir=${trt_model_path} \
    --gpt_attention_plugin=${model_dtype} \
    --gemm_plugin=${model_dtype} \
    --remove_input_padding=enable \
    --paged_kv_cache=enable \
    --tp_size=${model_tp_size} \
    --max_batch_size=${max_batch_size} \
    --max_input_len=${max_input_len} \
    --max_output_len=${max_output_len} \
    --max_num_tokens=${max_output_len} \
    --opt_num_tokens=${max_output_len} \
    --output_dir=${trt_engine_path} \
    $CHUNKED_PROMPT_FLAGS
```

Copy the generated TRT engine to *triton_model_repo* as follows:

```bash     
cd /tensorrtllm_backend/triton_model_repo
cp -r ${trt_engine_path}/* ./tensorrt_llm/1
```

Modify **triton_model_repo** config files as follows:
1. Modify **ensemble/config.pbtxt**: 

| Param            | Value                                     |
| ---------------- | ----------------------------------------- |
| `max_batch_size` | Set to the value of **${max_batch_size}** |

2. Modify **preprocessing/config.pbtxt**: 

| Param            | Value                                     |
| ---------------- | ----------------------------------------- |
| `max_batch_size` | Set to the value of **${max_batch_size}** |
| `tokenizer_dir`  | Set to the value of **${model_path}**     |

3. Modify **postprocessing/config.pbtxt**: 

| Param            | Value                                     |
| ---------------- | ----------------------------------------- |
| `max_batch_size` | Set to the value of **${max_batch_size}** |
| `tokenizer_dir`  | Set to the value of **${model_path}**     |

4. Modify **tensorrt_llm/config.pbtxt**: 

| Param                            | Value                                                        |
| -------------------------------- | ------------------------------------------------------------ |
| `max_batch_size`                 | Set to the value of **${max_batch_size}**                    |
| `decoupled`                      | Ensure it is set to **true** (to allow generate_stream)      |
| `gpt_model_type`                 | Ensure it is using **inflight_fused_batching** to allow continuous batching of requests |
| `batch_scheduler_policy`         | Ensure it is using **max_utilization** to batch requests as much as possible |
| `kv_cache_free_gpu_mem_fraction` | Ensure it is set to **0.9**. This value indicates the maximum fraction of GPU memory (after loading the model) that may be used for KV cache. |


4. Modify **tensorrt_llm_bls/config.pbtxt**: 

| Param            | Value                                                   |
| ---------------- | ------------------------------------------------------- |
| `max_batch_size` | Set to the value of **${max_batch_size}**               |
| `decoupled`      | Ensure it is set to **true** (to allow generate_stream) |

Start Triton server:

```bash
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py --world_size=${model_tp_size} --model_repo=/tensorrtllm_backend/triton_model_repo
```

Ensure that the triton-server is loaded correctly by checking that the model parts are in READY state, like in this output:

```bash
I0725 10:28:00.245297 294263 server.cc:677] 
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| ensemble         | 1       | READY  |
| postprocessing   | 1       | READY  |
| preprocessing    | 1       | READY  |
| tensorrt_llm     | 1       | READY  |
| tensorrt_llm_bls | 1       | READY  |
+------------------+---------+--------+

I0725 10:28:00.295473 294263 metrics.cc:877] Collecting metrics for GPU 0: NVIDIA A100-SXM4-80GB
```
this setting requires about **75 GB**

```bash
nvidia-smi
Thu Jul 25 10:31:55 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:03:00.0 Off |                  Off |
| N/A   33C    P0             58W /  400W |   74665MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

At this point, triton-server is running inside the docker container, so we can exit the docker or go to another terminal to run the client. 

- Send request
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'

{"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"\nMachine learning is a subset of artificial intelligence (AI) that uses algorithms to learn from data and"}
```

* Run several requests at the same time

```bash
echo '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}' > tmp.txt
printf '%s\n' {1..20} | xargs -I % -P 20 curl -X POST localhost:8000/v2/models/ensemble/generate -d @tmp.txt
```

# Acknowledgments

- [neuralmagic](https://github.com/neuralmagic)/[tensorrt-demo](https://github.com/neuralmagic/tensorrt-demo)
