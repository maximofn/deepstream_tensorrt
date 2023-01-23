import tensorrt as trt

print("Export model")
# To create a builder, you must first create a logger
logger = trt.Logger(trt.Logger.WARNING)

# Can then create a builder
builder = trt.Builder(logger)

# 1.1.  Creating a Network Definition in Python
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 1.2.  Importing a Model Using the ONNX Parser
# The network definition must be populated from the ONNX representation
parser = trt.OnnxParser(network, logger)

# Then, read the model file and process any errors
model_path = "segmentation_custom.onnx"
successNoCuda = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not successNoCuda:
    print(f"Error reading the model {model_path} and procesing errors")
else:
    print(f"Reading the model {model_path} success")

# 1.3.  Building an Engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

# After the configuration has been specified, the engine can be built and serialized
serializedEngine = builder.build_serialized_network(network, config)

# save the engine
engineName = "segmentation_custom_script.engine"
with open(engineName, "wb") as f:
    f.write(serializedEngine)
print(f"Serialized engine saved to {engineName}")