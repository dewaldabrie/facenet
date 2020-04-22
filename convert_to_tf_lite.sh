# Convert a TF .pb model into TFLite with optional quantization for reduction in model size.
#
# Overall repo cloned from https://github.com/davidsandberg/facenet.git
# Modications commit to git history in this repo.
# Taken from https://medium.com/analytics-vidhya/facenet-on-mobile-cb6aebe38505

source .venv/bin/activate
pip install -r requirements.txt
python inference_graph.py models/ model_inference/
python src/freeze_graph.py model_inference/ facenet_frozen.pb
tflite_convert --output_file model_mobile/my_facenet.tflite --graph_def_file facenet_frozen.pb --input_arrays "input" --input_shapes "1,160,160,3" --output_arrays "embeddings" --output_format TFLITE --std_dev_values 128 --mean_values 128 --default_ranges_min 0 --default_ranges_max 6 --inference_type QUANTIZED_UINT8 --inference_input_type QUANTIZED_UINT8
python model_mobile/verify_model.py