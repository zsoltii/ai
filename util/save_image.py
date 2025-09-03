import datetime
import re

SEPARATOR = "_"


def save_image(prompt, image, model, step="default"):
    if not isinstance(step, str):
        step = str(step)
    safe_prompt = re.sub(r"[^a-zA-Z]", SEPARATOR, prompt)
    safe_model = re.sub(r"[^a-zA-Z]", SEPARATOR, model)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(safe_model + SEPARATOR + "steps_" + step + SEPARATOR + timestamp + SEPARATOR + safe_prompt + ".png")