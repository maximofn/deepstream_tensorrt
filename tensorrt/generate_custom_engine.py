import tensorrt as trt

# execute bash command
import subprocess
command = "/usr/src/tensorrt/bin/trtexec --onnx=resnet50custom.onnx --saveEngine=resnet50custom_terminal.engine  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
subprocess.call(command, shell=True)

print("Export no cuda model")
# To create a builder, you must first create a logger
loggerNoCuda = trt.Logger(trt.Logger.WARNING)

# Can then create a builder
builderNoCuda = trt.Builder(loggerNoCuda)

# 1.1.  Creating a Network Definition in Python
networkNoCuda = builderNoCuda.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 1.2.  Importing a Model Using the ONNX Parser
# The network definition must be populated from the ONNX representation
parserNoCuda = trt.OnnxParser(networkNoCuda, loggerNoCuda)

# Then, read the model file and process any errors
model_path = "resnet50custom.onnx"
successNoCuda = parserNoCuda.parse_from_file(model_path)
for idx in range(parserNoCuda.num_errors):
    print(parserNoCuda.get_error(idx))
if not successNoCuda:
    print(f"Error reading the model {model_path} and procesing errors")
else:
    print(f"Reading the model {model_path} success")

# 1.3.  Building an Engine
configNoCuda = builderNoCuda.create_builder_config()
configNoCuda.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

# After the configuration has been specified, the engine can be built and serialized
serializedEngineNoCuda = builderNoCuda.build_serialized_network(networkNoCuda, configNoCuda)

# save the engine
engineNameNoCuda = "resnet50custom_script.engine"
with open(engineNameNoCuda, "wb") as f:
    f.write(serializedEngineNoCuda)
print(f"Serialized engine saved to {engineNameNoCuda}")



# print("Export cuda model")
# # To create a builder, you must first create a logger
# loggerCuda = trt.Logger(trt.Logger.WARNING)

# # Can then create a builder
# builderCuda = trt.Builder(loggerCuda)

# # 1.1.  Creating a Network Definition in Python
# networkCuda = builderCuda.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# # 1.2.  Importing a Model Using the ONNX Parser
# # The network definition must be populated from the ONNX representation
# parserCuda = trt.OnnxParser(networkCuda, loggerCuda)

# # Then, read the model file and process any errors
# model_path = "resnet50customCuda.onnx"
# successCuda = parserCuda.parse_from_file(model_path)
# for idx in range(parserCuda.num_errors):
#     print(parserCuda.get_error(idx))
# if not successCuda:
#     print(f"Error reading the model {model_path} and procesing errors")
# else:
#     print(f"Reading the model {model_path} success")

# # 1.3.  Building an Engine
# configCuda = builderCuda.create_builder_config()
# configCuda.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

# # After the configuration has been specified, the engine can be built and serialized
# serializedEngineNoCuda = builderCuda.build_serialized_network(networkCuda, configCuda)

# # save the engine
# engineNameCuda = "resnet50customCuda_script.engine"
# with open(engineNameCuda, "wb") as f:
#     f.write(serializedEngineNoCuda)
# print(f"Serialized engine saved to {engineNameCuda}")
