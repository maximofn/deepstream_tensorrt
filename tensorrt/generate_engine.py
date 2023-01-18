import tensorrt as trt

############################# 1. Build phase #############################

# To create a builder, you must first create a logger
logger = trt.Logger(trt.Logger.WARNING)

# Can then create a builder
builder = trt.Builder(logger)
# print("Create builder:")
# print(f"\tbuilder max_batch_size {builder.max_batch_size}")
# print(f"\tbuilder platform_has_tf32 {builder.platform_has_tf32}")
# print(f"\tbuilder platform_has_fast_fp16 {builder.platform_has_fast_fp16}")
# print(f"\tbuilder platform_has_fast_int8 {builder.platform_has_fast_int8}")
# print(f"\tbuilder max_DLA_batch_size {builder.max_DLA_batch_size}")
# print(f"\tbuilder num_DLA_cores {builder.num_DLA_cores}")
# print(f"\tbuilder error_recorder {builder.error_recorder}")
# # print(f"\tbuilder gpu_allocator {builder.gpu_allocator}")
# print(f"\tbuilder logger {builder.logger}")
# print(f"\tbuilder max_threads {builder.max_threads}")

# 1.1.  Creating a Network Definition in Python
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# print("Create network:")
# print(f"\tnetwork num_layers {network.num_layers}")
# print(f"\tnetwork num_inputs {network.num_inputs}")
# print(f"\tnetwork num_outputs {network.num_outputs}")
# print(f"\tnetwork name {network.name}")
# print(f"\tnetwork has_implicit_batch_dimension {network.has_implicit_batch_dimension}")
# print(f"\tnetwork has_explicit_precision {network.has_explicit_precision}")
# print(f"\tnetwork error_recorder {network.error_recorder}")
# print(f"\tnetwork 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) {1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)}")
# The EXPLICIT_BATCH flag is required in order to import models using the ONNX parser

# 1.2.  Importing a Model Using the ONNX Parser
# The network definition must be populated from the ONNX representation
# network = "resnet50_pytorch_BS1.onnx"
parser = trt.OnnxParser(network, logger)

# Then, read the model file and process any errors
model_path = "resnet50_pytorch_BS1.onnx"
success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    print(f"Error reading the model {model_path} and procesing errors")
else:
    print(f"Reading the model {model_path} success")

# 1.3.  Building an Engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

# After the configuration has been specified, the engine can be built and serialized
serialized_engine = builder.build_serialized_network(network, config)
# print("built and serialized network:")
# print(f"\tnetwork num_layers {network.num_layers}")
# print(f"\tnetwork num_inputs {network.num_inputs}")
# print(f"\tnetwork num_outputs {network.num_outputs}")
# print(f"\tnetwork name {network.name}")
# print(f"\tnetwork has_implicit_batch_dimension {network.has_implicit_batch_dimension}")
# print(f"\tnetwork has_explicit_precision {network.has_explicit_precision}")
# print(f"\tnetwork error_recorder {network.error_recorder}")

# save the engine
engine_name = "resnet50_pytorch_BS1_script.engine"
with open(engine_name, "wb") as f:
    f.write(serialized_engine)
print(f"Serialized engine saved to {engine_name}")
