import json
import os
import numpy as np
from logic.machine import Machine, AdalineMachine


class JsonSaver():
    def __init__(self):
        pass

    def save(self, machine: Machine, filename: str) -> bool:
        payload = {
            "type": "AdalineMachine" if type(machine) == AdalineMachine else "Machine",
            "w1": machine.w1,
            "w2": machine.w2,
            "bias": machine.bias,
            "tolerance": machine.tolerance,
            "epoch": machine.epoch,
            "t": machine.t,
            "error_scores": machine.error_scores,
            "data": machine.data.tolist()      if machine.data      is not None else None,
            "class_val": machine.class_val.tolist() if machine.class_val is not None else None,
        }

        if not filename.endswith(".json"): filename += ".json"
        path = os.path.join(self.catalog, filename)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception as e:
            raise Exception(f"Failed to save: {e}")

    def open(self, path) -> Machine:
        if not os.path.isfile(path):
            raise Exception(f"File not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON file: {e}")

        machine_type = payload.get("type", "Machine")
        machine = AdalineMachine() if machine_type == "AdalineMachine" else Machine()

        machine.w1 = payload["w1"]
        machine.w2 = payload["w2"]
        machine.bias = payload["bias"]
        machine.tolerance = payload["tolerance"]
        machine.epoch = payload["epoch"]
        machine.t = payload["t"]
        machine.error_scores = payload["error_scores"]

        machine.data = np.array(payload["data"]) if payload["data"] is not None else None
        machine.class_val = np.array(payload["class_val"]) if payload["class_val"] is not None else None

        return machine