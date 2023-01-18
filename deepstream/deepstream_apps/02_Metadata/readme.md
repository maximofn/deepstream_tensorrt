En la función *osd_sink_pad_buffer_probe* se obtiene la información de cada batch: **batch_meta** que es de tipo *NvDsBatchMeta*

<figure>
    <img src="../../deepstream_nano/images/DS_plugin_metadata_720.png"
         alt="metadata">
    <figcaption style="text-align:center;"><b>Metadata Structure Diagram</b></figcaption>
</figure>

 * De ahí se obtiene la información de cada frame: **frame_meta** que es de tipo *NvDsBatchMeta.NvDsFrameMeta*
    * De cada frame se obtiene:
        * **frame_number**: Número de frame, es de tipo *NvDsBatchMeta.NvDsFrameMeta.frame_num*
        * **num_rects**: Número de objetos, es de tipo *NvDsBatchMeta.NvDsFrameMeta.num_obj_meta*
        * **l_obj**: Lista de objetos, es de tipo *NvDsBatchMeta.NvDsFrameMeta.NvDsObjectMeta*. En el momento que se tiene cada objeto se puede sacar información de el, como la posición del cuadro delimitador con:
            * **obj_meta.rect_params.height**
            * **obj_meta.rect_params.width**
            * **obj_meta.rect_params.left**
            * **obj_meta.rect_params.top**