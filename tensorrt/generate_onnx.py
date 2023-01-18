import torchvision
import torch
from torch.onnx import OperatorExportTypes

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

# Export to ONNX
BATCH_SIZE=1
onnx_file = f"resnet50_pytorch_BS{BATCH_SIZE}.onnx"
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
output = torch.onnx.export(
    model=model,
    args=dummy_input,
    f=onnx_file,
    export_params=True,
    verbose=False,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=["input"],
    output_names=["output"],
    operator_export_type=OperatorExportTypes.ONNX_EXPLICIT_BATCH)

output = torch.onnx.export(
    model=model,                # model being run
    args=dummy_input,           # model input (or a tuple for multiple inputs)
    f=onnx_file,                # where to save the model (can be a file or file-like object)
    export_params=True,         # if True, all parameters will be exported. Set this to False if you want to export an untrained model
    verbose=False,              # if True, prints a description of the model being exported to stdout. 
                                # In addition, the final ONNX graph will include the field doc_string` 
                                # from the exported model which mentions the source code locations for model. 
                                # If True, ONNX exporter logging will be turned on
    training=torch.onnx.TrainingMode.EVAL,  # TrainingMode.EVAL: export the model in inference mode.
                                            # TrainingMode.PRESERVE: export the model in inference mode if model.training is
                                            #   False, and export the model in training mode if model.training is True.
                                            # TrainingMode.TRAINING: export the model in training mode. Disables optimizations
                                            #   which might interfere with training
    input_names=["input"],      # names to assign to the input nodes of the graph
    output_names=["output"],    # names to assign to the output nodes of the graph
    operator_export_type=OperatorExportTypes.ONNX_EXPLICIT_BATCH,   # OperatorExportTypes.ONNX: indica que se deben utilizar solo los 
                                                                    #   operadores ONNX disponibles. Si un operador no está disponible 
                                                                    #   en ONNX, la exportación fallará. Este es el valor predeterminado.
                                                                    # OperatorExportTypes.ONNX_FALLTHROUGH: indica que se deben utilizar 
                                                                    #   los operadores ONNX disponibles y, si un operador no está 
                                                                    #   disponible, se debe utilizar el operador PyTorch equivalente.
                                                                    # OperatorExportTypes.ONNX_ATEN_FALLBACK: indica que se deben utilizar 
                                                                    #   los operadores ONNX disponibles y, si un operador no está 
                                                                    #   disponible, se debe utilizar el operador ATen equivalente.
                                                                    #   ATen es una biblioteca de código abierto que proporciona una interfaz 
                                                                    #   de alto nivel para la manipulación de tensores y la implementación de 
                                                                    #   operaciones matemáticas en PyTorch. Es la capa subyacente que se utiliza 
                                                                    #   para implementar muchas de las operaciones matemáticas y de manipulación 
                                                                    #   de tensores en PyTorch. Por ejemplo, cuando invocas una función como 
                                                                    #   torch.add() o torch.mm(), en realidad estás llamando a una función de 
                                                                    #   ATen debajo de la capa
                                                                    # OperatorExportTypes.ONNX_EXPLICIT_BATCH: este valor es una variante 
                                                                    #   de OperatorExportTypes.ONNX y solo se aplica a operadores que 
                                                                    #   tienen un atributo "batch" o "batch_size". Cuando se utiliza esta 
                                                                    #   opción, se inserta un operador ONNX "Unsqueeze" para añadir una 
                                                                    #   dimensión "batch" a estos operadores. Esto es útil cuando se quiere 
                                                                    #   exportar un modelo que fue entrenado con un tamaño de lote mayor que 1.
    opset_version=17,           # The version of the ONNX operator set to export the model to.
                                # Según https://github.com/onnx/onnx-tensorrt/blob/1da7332349d5b1196ccfa6dc719b839876f1e83e/docs/operators.md
                                # la última versión de ONNX compatible con tensorrt 8.4 es la 17
    #do_constant_folding=True,  # Especifica si se deben realizar plegados de constantes al exportar el modelo a ONNX.El plegado de constantes 
                                # es una técnica de optimización que consiste en reemplazar una expresión por su resultado durante el tiempo 
                                # de compilación en lugar de evaluarla en tiempo de ejecución. Esto puede mejorar el rendimiento de un modelo 
                                # al reducir el número de operaciones que se deben realizar en tiempo de ejecución.
    #dynamic_axes=None,         # a dictionary specifying the dynamic axes of the input and output tensors
    #keep_initializers_as_inputs=None,  # If True, keep the initializers of the model as inputs of the graph.
    #custom_opsets=None,        # a dictionary mapping domain names to operator set versions
    #export_modules_as_functions=False, # If True, export modules as functions
)
print(f"Done exporting to ONNX, type of output: {type(output)}")

# Export to TensorRT
# !/usr/src/tensorrt/bin/trtexec --onnx=../../tensorrt/resnet50_pytorch_BS1.onnx --saveEngine=resnet50_engine_pytorch_BS1.engine --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16