import os
import json
from collections.abc import Iterable

__all__ = ["Orchestrator"]


class Orchestrator:

    def __init__(self, orchestrator_type="kubeflow", storage_path="./"):
        self.type = orchestrator_type
        self.storage_path = storage_path
        assert orchestrator_type in ("step_functions", "kubeflow")

        print(f"Initialized {self}")
    
    def __repr__(self):
        return f"Orchestrator(type={self.type})"
    
    def _serialize_value(self, value, extension):
        if extension == "json":
            return json.dumps(value)
        return str(value)

    def _write_file(self, filename, content):
        with open(filename, "w+") as file: 
            file.write(content)

    def export_meta(self, key, value, extension=None):
        if extension is not None:
            key = f"{key}.{extension}"
            value = self._serialize_value(value, extension)

        self._write_file(
            filename=os.path.join(self.storage_path, key), 
            content=value)