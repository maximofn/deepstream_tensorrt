# Programa que detecta las 4 clases
Usar dstest1_pgie_config_4classes.txt
Ejecutar:
python3 objDetect.py -i ../samples/sample_qHD.h264

# Programa que detecta solo 2 clases
Usar dstest1_pgie_config_2classes.txt
 * Cambiar *num-detected-classes* a 2
 * Procurar que *[class-attrs-i]* no pase de 1 (1 o 0)
Ejecutar:
python3 objDetect.py -i ../samples/sample_qHD.h264 -c dstest1_pgie_config_2classes.txt