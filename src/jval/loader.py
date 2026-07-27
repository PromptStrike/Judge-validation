import json
from jval.models import Sample

def load_samples_jsonl(path: str) -> list[Sample]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(Sample(
                id=d.get("id", f"row_{i}"),
                prompt=d["prompt"],
                response=d["response"],
                source_meta=d.get("source_meta", {}),
            ))
    return samples
