import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from quant_utils import (  # noqa: E402
    QuantizedWeights,
    CodebookQuantizedWeights,
    WeightQuantizer,
)


def _storage_bytes(tensor):
    return tensor.nelement() * tensor.element_size()


def run_roundtrip_tests():
    torch.manual_seed(0)
    x = torch.randn(16, 16, dtype=torch.float32)
    configs = [
        ("4bit_sym", dict(bits=4, perchannel=True, sym=True, w_quant_scheme="sym")),
        ("4bit_asym", dict(bits=4, perchannel=True, sym=False, w_quant_scheme="asym")),
        ("3bit_sym", dict(bits=3, perchannel=True, sym=True, w_quant_scheme="sym")),
        ("3bit_asym", dict(bits=3, perchannel=True, sym=False, w_quant_scheme="asym")),
        ("2bit_sym", dict(bits=2, perchannel=True, sym=True, w_quant_scheme="sym")),
        ("2bit_asym", dict(bits=2, perchannel=True, sym=False, w_quant_scheme="asym")),
        ("1bit_binary", dict(bits=1, perchannel=True, sym=True, w_quant_scheme="binary")),
        ("ternary", dict(bits=2, perchannel=True, sym=True, w_quant_scheme="ternary")),
    ]

    for name, kwargs in configs:
        quantizer = WeightQuantizer()
        quantizer.configure(**kwargs)
        quantizer.find_params(x)
        direct = quantizer(x)
        packed = quantizer.quantize(x, qat=False)
        unpacked = packed()
        if not torch.equal(direct, unpacked):
            raise AssertionError(f"{name}: packed round-trip changed dequantized weights")
        if hasattr(packed, "unpack_weight_q"):
            q = packed.unpack_weight_q()
            if name == "ternary":
                allowed = {-1, 0, 1}
            elif name == "1bit_binary":
                allowed = {-1, 1}
            else:
                allowed = None
            if allowed is not None:
                unique = set(torch.unique(q).tolist())
                if unique - allowed:
                    raise AssertionError(f"{name}: unexpected unpacked codes {sorted(unique)}")


def run_storage_test():
    torch.manual_seed(0)
    layer = torch.nn.Linear(1024, 1024, bias=False, dtype=torch.float16)
    weight = layer.weight.detach().to(torch.float32)

    checks = [
        ("4bit_sym", dict(bits=4, perchannel=True, sym=True, w_quant_scheme="sym")),
        ("4bit_asym", dict(bits=4, perchannel=True, sym=False, w_quant_scheme="asym")),
        ("3bit_sym", dict(bits=3, perchannel=True, sym=True, w_quant_scheme="sym")),
        ("3bit_asym", dict(bits=3, perchannel=True, sym=False, w_quant_scheme="asym")),
        ("2bit_sym", dict(bits=2, perchannel=True, sym=True, w_quant_scheme="sym")),
        ("2bit_asym", dict(bits=2, perchannel=True, sym=False, w_quant_scheme="asym")),
        ("1bit_binary", dict(bits=1, perchannel=True, sym=True, w_quant_scheme="binary")),
        ("ternary", dict(bits=2, perchannel=True, sym=True, w_quant_scheme="ternary")),
    ]

    original_fp16_bytes = _storage_bytes(layer.weight)
    print(f"original_fp16_bytes={original_fp16_bytes}")

    for name, kwargs in checks:
        quantizer = WeightQuantizer()
        quantizer.configure(**kwargs)
        quantizer.find_params(weight)
        packed = quantizer.quantize(weight, qat=False)
        if isinstance(packed, (QuantizedWeights, CodebookQuantizedWeights)):
            packed_bytes = packed.packed_storage_size_bytes()
            unpacked_q = packed.unpack_weight_q()
            theoretical_q_bytes = (unpacked_q.numel() * kwargs["bits"] + 7) // 8
            print(
                f"{name}: packed_bytes={packed_bytes}, "
                f"packed_payload_bytes={_storage_bytes(packed.packed_weight_q)}, "
                f"theoretical_q_bytes={theoretical_q_bytes}, "
                f"scale_bytes={_storage_bytes(packed.scale)}"
            )
            if packed_bytes >= original_fp16_bytes:
                raise AssertionError(f"{name}: packed storage did not get smaller")
        else:
            raise AssertionError(f"{name}: unexpected packed object type {type(packed)}")


if __name__ == "__main__":
    run_roundtrip_tests()
    run_storage_test()
    print("packing tests passed")
