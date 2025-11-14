import os, sys, json, random, string

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.log_available_gpus import log_available_gpus
from util.tool.currency_converter import currency_converter_to_huf
from util.tool.mistral_tools import generate_tool_id, TOOLS, get_tool_call_json, get_raw_tool_calls, \
    get_standard_tool_calls, run_tool_and_append_messages_with_result

log_available_gpus()

# model_id = "mistralai/Mistral-7B-Instruct-v0.2" # no tools support
model_id = "mistralai/Mistral-7B-Instruct-v0.3"  # tools support

max_memory = {
    0: "14Gib",  # Reduce memory allocation for GPU 0
    "cpu": "55Gib"
}

os.makedirs("./offload", exist_ok=True)

print("Using max_memory config:", max_memory)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    max_memory=max_memory,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
)

print(f"VRAM elosztás: {max_memory}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Modell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)


def run_conversation(messages):
    tool_map = {
        "currency_converter_to_huf": currency_converter_to_huf,
    }

    response_text = get_tool_call_json(model, tokenizer, TOOLS, messages)

    raw_tool_calls = get_raw_tool_calls(response_text)

    tool_calls = get_standard_tool_calls(raw_tool_calls)

    # Add the assistant's response with tool calls to the message history
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
    )

    # Execute the tool calls
    for tool_call in tool_calls:
        messages = run_tool_and_append_messages_with_result(tool_call, tool_map, messages)

    # Now that we have the tool outputs, we run the model again to get the final response.
    final_input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    final_inputs = tokenizer(final_input_text, return_tensors="pt").to(model.device)

    print("Végső válasz generálása a tool eredménye alapján...")
    with torch.no_grad():
        final_res = model.generate(**final_inputs, max_new_tokens=500, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)

    final_answer_ids = final_res[0][final_inputs.input_ids.shape[1]:]
    final_answer = tokenizer.decode(final_answer_ids, skip_special_tokens=True)
    print("AI végső válasz:", final_answer)


# Példa beszélgetés indítása
initial_messages = [
    {"role": "user", "content": "Mennyi 200 euró forintban?"}
]
run_conversation(initial_messages)
