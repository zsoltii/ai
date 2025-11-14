import random, string, torch, json


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "currency_converter_to_huf",
            "description": "Átvált egy összeget egy megadott pénznemről HUF-ra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Az átváltandó összeg, pl. 100",
                    },
                    "currency": {
                        "type": "string",
                        "description": "A forrás pénznem, pl. 'USD'.",
                    },
                },
                "required": ["amount", "currency"],
            },
        },
    }
]

def generate_tool_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=9))

def get_tool_call_json(model, tokenizer, tools, messages) -> any:
    # This is the first turn, with user message
    input_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("Tool hívás kérés generálás folyamatban...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    # Decode only the generated part, not the prompt
    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print("AI válasz (tool hívás):", response_text)

    # The model output might have `[TOOL_CALLS]` at the beginning, which we can strip.
    response_text = response_text.strip()
    print(f"response_text striped: {response_text}")
    if response_text.startswith("[TOOL_CALLS]"):
        response_text = response_text[len("[TOOL_CALLS]"):].strip()
    return response_text

def get_raw_tool_calls(tool_call_json) -> any:
    raw_tool_calls = []
    try:
        # Use raw_decode to handle potential trailing characters from the model
        json_decoder = json.JSONDecoder()
        raw_tool_calls, _ = json_decoder.raw_decode(tool_call_json)
    except json.JSONDecodeError as e:
        # Not a tool call, probably a direct answer.
        print(f"AI végső válasz (nincs tool hívás, JSONDecodeError: {e}):", tool_call_json)
        return

    if not raw_tool_calls:
        print("AI végső válasz (nincs tool hívás, üres tool lista):", tool_call_json)
        return

    # The model can return a single dictionary or a list of dictionaries
    if isinstance(raw_tool_calls, dict):
        raw_tool_calls = [raw_tool_calls]

    print(f"Raw tool calls: {raw_tool_calls}")

    return raw_tool_calls

def get_standard_tool_calls(raw_tool_calls) -> any:
    # The Mistral v0.3 model returns a simplified format. We need to convert it to the standard format
    # that the chat template expects, which includes an 'id', 'type', and 'function' structure.
    standard_tool_calls = []

    for raw_tool_call in raw_tool_calls:
        standard_tool_calls.append({
            "id": generate_tool_id(),
            "type": "function",
            "function": {
                "name": raw_tool_call["name"],
                # The arguments from the model are a dict, but the template expects a JSON string
                "arguments": json.dumps(raw_tool_call["arguments"])
            }
        })

    return standard_tool_calls

# def run_tool_and_append_messages_with_result(messages, tool_map, function_name, function_args, tool_call_id) -> any:
#     function_to_call = tool_map[function_name]
#     try:
#         print(f"Tool futtatása: {function_name} argumentumokkal: {function_args}")
#         tool_output = function_to_call(**function_args)
#         print(f"Tool eredménye: {tool_output}")
#
#         # Append the tool output to the messages
#         messages.append({
#             "role": "tool",
#             "tool_call_id": tool_call_id,
#             "name": function_name,
#             "content": str(tool_output),
#         })
#
#     except Exception as e:
#         print(f"Hiba a tool futtatása közben: {e}")
#         messages.append({
#             "role": "tool",
#             "tool_call_id": tool_call_id,
#             "name": function_name,
#             "content": f"Error executing tool: {e}",
#         })
#
#     return messages

def run_tool_and_append_messages_with_result(tool_call, tool_map, messages) -> any:
    function_call = tool_call['function']
    function_name = function_call['name']
    tool_call_id = tool_call['id']

    try:

        # The arguments are a JSON string, so we need to parse them
        function_args = json.loads(function_call['arguments'])

        print(f"Függvényhívás: {function_name} argumentumokkal: {function_args}")

        if function_name in tool_map:
            function_to_call = tool_map[function_name]
            try:
                print(f"Tool futtatása: {function_name} argumentumokkal: {function_args}")
                tool_output = function_to_call(**function_args)
                print(f"Tool eredménye: {tool_output}")

                # Append the tool output to the messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": str(tool_output),
                })

            except Exception as e:
                print(f"Hiba a tool futtatása közben: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": f"Error executing tool: {e}",
                })
        else:
            print(f"Ismeretlen tool: {function_name}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": f"Error: Unknown tool {function_name}",
            })
    except json.JSONDecodeError as e:
        print(f"Hiba a tool argumentumok feldolgozása közben: {e}")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_name,
            "content": f"Error parsing arguments: {e}",
        })

    return messages