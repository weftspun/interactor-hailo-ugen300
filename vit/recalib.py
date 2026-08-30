"""Same graph, two calibration sets: uniform [0,1) and the standard normal the input is drawn from.

The first HEF scored 65% relative error against the fp32 control. If the calibration
distribution is the cause, matching it to the input distribution collapses that error and
nothing else changes -- same weights, same ONNX, same compiler settings.
"""
import time
import numpy as np
from hailo_sdk_client import ClientRunner

S, C = 64, 64
for tag, calib in (("uniform", np.random.rand(256, 1, S, C).astype(np.float32)),
                   ("normal", np.random.default_rng(7).standard_normal((256, 1, S, C)).astype(np.float32))):
    r = ClientRunner(hw_arch="hailo10h")
    r.translate_onnx_model("/work/rb_batched.onnx", "rb_" + tag, start_node_names=["x"],
                           end_node_names=["y"], net_input_shapes={"x": [1, S, C]})
    t = time.time()
    r.optimize(calib)
    hef = r.compile()
    open("/work/rb_%s.hef" % tag, "wb").write(hef)
    print("%-8s calib mean %+.3f std %.3f -> %d bytes in %.1f s"
          % (tag, calib.mean(), calib.std(), len(hef), time.time() - t))
