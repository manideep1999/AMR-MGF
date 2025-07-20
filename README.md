# EMGF: Extensible Multi-Granularity Fusion Network for Aspect-based Sentiment Analysis

# You can download EMGF's model weights from [Google Drive](https://drive.google.com/drive/folders/1A3ZtkLyrzMwijoFXfyx_5u2rhh341kKw?usp=sharing) to reproduce the results.


# Preprocessing Part
    1. Copy the train and test paths of the old datasets of the amrbart (containing the amr_edges and amr_tokens) to the EMGF dataset directory
    2. Modify the path accordingly for the train and tests to merge both the data and overwrite it in the train_new.json and test_new.json files
    3. For this, run the cmd : python merge_json.py 

