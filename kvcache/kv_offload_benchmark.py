import time
import torch
import os

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig

# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  
# os.environ["VLLM_TRACE_FUNCTION"] = "1"  
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
# os.environ["VLLM_DEBUG_DUMP_PATH"] = "./debug_dump" 

print(f"====SXB start")
CPU_CACHE_SIZE_GB = 100
CPU_BLOCK_SIZE = 64
NUM_DECODED_TOKENS_PER_PROMPT = 1

# MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL = "/data/models/DeepSeek-V3.2"
# sizeof(element) * head_size * num_heads * |{k,v}| * layer_count
KV_SIZE_PER_TOKEN = 2 * 128 * 8 * 2 * 32

cpu_bytes_to_use = CPU_CACHE_SIZE_GB << 30
cache_tokens_capacity = cpu_bytes_to_use // KV_SIZE_PER_TOKEN

# for throughput test
HIT_PROMPT_SIZE = 512
MISS_PROMPT_SIZE = 512
NUM_PROMPTS = 10000
HIT_PERCENTS_TO_TEST = tuple(range(0, 101, 10))

# for latency test
PROMPT_SIZES_IN_K_TO_TEST = (1,) + tuple(range(10, 101, 10))

sampling_params = SamplingParams(
    max_tokens=NUM_DECODED_TOKENS_PER_PROMPT,
    detokenize=False,
    ignore_eos=True
)
num_cpu_blocks = cache_tokens_capacity // CPU_BLOCK_SIZE
ktc = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "block_size": CPU_BLOCK_SIZE,
        "cpu_bytes_to_use": cpu_bytes_to_use,
        "num_cpu_blocks": num_cpu_blocks, # required in older versions (0.12.0)
    }
)

# kv_events_config=KVEventsConfig(enable_kv_cache_events=True)

llm = LLM(
    model=MODEL,
    enable_prefix_caching=False,
    kv_transfer_config=ktc,
    # DeepSeek-V3.2 may need these additional settings:  
    quantization="fp8",  # matching your model's quantization  
    dtype=torch.bfloat16,
    enforce_eager=True,  # Disables both torch.compile and CUDA graphs
    gpu_memory_utilization=0.7,  # Reduce from default 0.9
    max_model_len=32768,
    max_num_seqs=16,
    tensor_parallel_size=8, 
)

    # enable_expert_parallel=True,     # Enable EP for MoE layers 
    # data_parallel_size=8,            # 8-way data parallel  
    # tensor_parallel_size=1,          # No TP for attention  
    # tensor_parallel_size=2,  # Split across 2 GPUs
    # compilation_config=CompilationConfig(  
    #     cudagraph_mode=CUDAGraphMode.NONE  
    # ) 
# run latency test
max_prompt_size = max(PROMPT_SIZES_IN_K_TO_TEST) << 10
iterations_count = cache_tokens_capacity // max_prompt_size
for i, prompt_size_k in enumerate(PROMPT_SIZES_IN_K_TO_TEST):
    prompt_size = prompt_size_k << 10
   
    print("Testing prompt length:", prompt_size_k, "K")
    total_prefill_time = 0
    for j in range(iterations_count):
        prompt = TokensPrompt(
            prompt_token_ids=[i] + [j] * (prompt_size - 1)
        )
        
        # cache miss, will trigger a prefill
        start_time = time.perf_counter()
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        total_prefill_time += time.perf_counter() - start_time
        
    average_prefill_time_ms = int(
        1000 * (total_prefill_time / iterations_count)
    )
    print("Average prefill time:", average_prefill_time_ms, "ms")

    total_cpu_load_time = 0
    for j in range(iterations_count):
        prompt = TokensPrompt(
            prompt_token_ids=[i] + [j] * (prompt_size - 1)
        )

        # cache hit, will load from CPU
        start_time = time.perf_counter()
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        total_cpu_load_time += time.perf_counter() - start_time
    
    average_cpu_load_time_ms = int(
        1000 * (total_cpu_load_time / iterations_count)
    )

    print("Average CPU load time:", average_cpu_load_time_ms, "ms")
    print()

# run throughput test
max_prompt_size = max(HIT_PROMPT_SIZE, MISS_PROMPT_SIZE)
cache_prompts_capacity = cache_tokens_capacity // max_prompt_size

hit_prompts = [
    TokensPrompt(prompt_token_ids=[0] + [i+3] * (HIT_PROMPT_SIZE - 1))
    for i in range(cache_prompts_capacity)
]
miss_prompts = [
    TokensPrompt(prompt_token_ids=[1] + [i+3] * (MISS_PROMPT_SIZE - 1))
    for i in range(NUM_PROMPTS)
]
reset_cache_prompts = [
    TokensPrompt(prompt_token_ids=[2] + [i+3] * (max_prompt_size - 1))
    for i in range(cache_prompts_capacity)
]

for hit_percent in HIT_PERCENTS_TO_TEST:
    # create a pattern of hit indexes per a batch of 100 requests
    hit_indexes = set()
    if hit_percent > 0:
        hit_indexes = {
            int(i * 100 / hit_percent) for i in range(hit_percent)
        }
    assert len(hit_indexes) == hit_percent

    # build prompts mixed with hits and misses
    hit_prompts_count = cache_prompts_capacity // 100 * hit_percent
    prompts = []
    hit_idx = 0
    for i in range(NUM_PROMPTS):
        if (i%100) in hit_indexes:
            prompts.append(hit_prompts[hit_idx])
            hit_idx = (hit_idx + 1) % hit_prompts_count
        else:
            prompts.append(miss_prompts[i])
            
    # reset the cache by filling it up
    llm.generate(reset_cache_prompts, sampling_params, use_tqdm=False)

    if hit_percent:
        # fill the CPU cache
        llm.generate(
            hit_prompts[:hit_prompts_count],
            sampling_params,
            use_tqdm=False
        )

    # sleep a bit to make sure writes to cache are done
    time.sleep(1)

    print("Testing hit percent:", hit_percent)

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    total_time = time.perf_counter() - start_time

    num_tokens = sum([
        len(x.prompt_token_ids) + len(x.outputs[0].token_ids)
        for x in outputs
    ])
    print(int(num_tokens/total_time), "tokens/sec")
    print()
