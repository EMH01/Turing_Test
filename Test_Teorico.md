## Caso 1: 
- Usuario tiene un gran número de facturas hechas a mano. Necesita que se le desarrolle un servicio que digitalice esas facturas y devuelva un json con el identificador de cada campo y su valor. 

### Propuesta de solución
A partir todos los documentos escaneados, se puede utilizar un servicio en la nube que automatizará la extracción de datos con IA como Form Recognizer de Azure Document Intelligence.
Dentro de este servicio, se puede optar por utilizar un modelo ya pre-entrenado para extraer datos de documentos de facturas, el cual será capaz de extraer los datos de diversos formatos en que se encuentren, en este caso como imagen y podremos obtener directamente la salida estructurada en formato JSON con los campos determinados.

- Técnicas y algoritmos/modelos utilizados
En Azure Document Intelligence se utilizan varias tecnicas en dependencia del modelo y los datos, en este caso al utilizar el modelo pre-entrenado "Invoice".

- Fuentes de la información encontrada
https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/invoice?view=doc-intel-4.0.0
https://github.com/Azure-Samples/document-intelligence-code-samples/blob/main/schema/2024-11-30-ga/invoice.md
https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/accuracy-confidence?view=doc-intel-4.0.0

- Justificación de uso
El modelo "Invoice" está diseñado para problemas relacionados con documentos financieros, como facturas, y extrae automaticamente campos clave. El uso de este seria eficiente y llevaria mucho menos tiempo que entrenar un modelo para la tarea especificada.

- Enlace a Documentación o ejemplo para su uso
https://github.com/Azure-Samples/document-intelligence-code-samples/blob/main/Python(v4.0)/Prebuilt_model/README.md
https://python.langchain.com/docs/integrations/document_loaders/azure_document_intelligence/

- Utilizando la solucion propuesta no es necesario el entrenamiento o fine tune del modelo. 
- Métricas utilizadas para poder medir el desempeño del modelo: 
El valor de "confidence" asociado a cada campo extraido.
Calcular el accuracy del flujo a partir de un subconjunto que se conforme de los datos a extraer como ground truth y comparar los resultados con los del modelo.

- Justificar posible éxito de la solución
La solucion propuesta sería efectiva y rapida de evaluar para resolver el caso planteado, donde obtienen resultados estructurados cumpliendo los objetivos planteados.

## Caso 2: 
- Usuario necesita un sistema que, haciendo uso de imágenes satelitales compruebe si la dirección tiene casa y si tiene casa, que devuelva si en esa dirección hay piscina y el tamaño de la misma. Podemos suponer que hay una imagen con cada dirección posible.
Extra: Al realizar el sistema, tenemos problema con los lagos y ríos, ya que los confunde con piscinas, qué solución plantearías para minimizar el impacto de estas masas de agua y evitar falsos positivos de piscinas?

### Propuesta de solución
Utilizar un modelo LLM multimodal para analizar una imagen y aplicar la tecnica de zero-shot para obtener si existe o no una casa en la direccion determinada.
Si hay una casa, la siguiente etapa seria la segmentacion de la imagen para detectar los cuerpos de agua presentes y determinar si hay una piscina.
Una vez confirmada la existencia de piscina podria calcularse su tamaño usando las proporciones de los pixeles llevandolas a la proporcion real.

- Técnicas y algoritmos/modelos. Entrenamiento o fine tune si aplica, razones y enfoque.
Uso del LLM por ejemplo de OpenAI GPT-4o, definiendole un prompt para describir la tarea y lo que se quiere obtener.
Uso del modelo SAM para la segmentacion, y de esta forma identificar cuerpos de agua, para definir si es una piscina y reducir los falsos positivos se puede analizar los pixeles delimitados en cuanto a tamaño, color y cercanía a la casa.

- Fuentes de la información encontrada
https://learn.microsoft.com/es-es/dotnet/ai/conceptual/zero-shot-learning
https://www.kaggle.com/code/skalskip/how-to-use-segment-anything-model-sam

- Justificación de uso
Las tecnicas y modelos propuestos son capaces de resolver el caso en muy poco tiempo en comparacion a entrenar modelos para la tarea especifica

- Enlace a Documentación o ejemplo para su uso
https://github.com/facebookresearch/segment-anything
https://medium.com/@abonia/sam-2-advanced-object-segmentation-for-images-and-videos-4e9a3eb48abc
https://github.com/EMH01/em_projects/blob/main/Zero_Shot_Example.ipynb
https://samgeo.gishub.org/examples/swimming_pools/

- Métricas utilizadas para poder medir el desempeño del modelo
Preparar un subconjunto de datos con respuestas correctas o esperadas y comparando con los resultados del LLM calcular F1-Score, Accuracy
Para la segmentacion utilizar IoU

- Justificar posible éxito de la solución
El uso de LLM y modelos de segmentacion, se utiliza en muchos y variados problemas similares con buenos resultados. Obtener mayor o menor exito dependera del enfoque del modelo, las imagenes utilizadas y la evaluacion de los resultados obtenidos, pero es una solucion viable y escalable ya que se divide el problema para garantizar que cada etapa se ejecute correctamente