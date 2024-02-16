import torch
import torch.nn as nn

from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage

from transformers import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer
from transformers import FLMRModelForRetrieval
from transformers import FLMRConfig, FLMRVisionConfig, FLMRTextConfig

if __name__ == '__main__':
    from transformers import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, AutoImageProcessor

    checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-G"
    image_processor_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer")

    model = FLMRModelForRetrieval.from_pretrained(checkpoint_path, 
                                                    query_tokenizer=query_tokenizer, 
                                                    context_tokenizer=context_tokenizer,
                                                    ).to("cuda")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

    Q_encoding = query_tokenizer(["Using the provided image, obtain documents that address the subsequent question: What is the capital of France?", "Extract documents linked to the question provided in conjunction with the image: What is the capital of France?"])
    D_encoding = context_tokenizer(["Paris is the capital of France.", "Beijing is the capital of China.",
                                "Paris is the capital of France.", "Beijing is the capital of China."])
    print(query_tokenizer.batch_decode(Q_encoding['input_ids'], skip_special_tokens=False))
    print(query_tokenizer.batch_decode(D_encoding['input_ids'], skip_special_tokens=False))
    input()
    
    Q_pixel_values = torch.randn(2, 3, 224, 224)

    inputs = dict(
        query_input_ids=Q_encoding['input_ids'],
        query_attention_mask=Q_encoding['attention_mask'],
        query_pixel_values=Q_pixel_values,
        context_input_ids=D_encoding['input_ids'],
        context_attention_mask=D_encoding['attention_mask'],
        use_in_batch_negatives=True,
    )
    
    forward_results = model.forward(**inputs)
    print(forward_results)
    exit()



    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained("./PreFLMR_ViT-G/query_tokenizer")
    Q_encoding = query_tokenizer(["This is instruction: What is the capital of France?", "This is instruction: This is a test sentence."])
    # print(res)
    # print(question_tokenizer.batch_decode(res["input_ids"]))

    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained("./PreFLMR_ViT-G/context_tokenizer")
    D_encoding = context_tokenizer(["Paris is the capital of France.", "Paris is not the capital of France.",
                                    "This is a test sentence.", "This is a test negative sentence."])
    # print(res)
    # print(context_tokenizer.batch_decode(res["input_ids"]))

    # flmr_model = FLMRModelForRetrieval.from_pretrained("bert-base-uncased", 
    #                                                     query_tokenizer=query_tokenizer, 
    #                                                     context_tokenizer=context_tokenizer,
    #                                                     query_concat_output_from_vision_encoder=True,
    #                                                     query_concat_output_from_text_encoder=True,
    #                                                     context_concat_output_from_vision_encoder=True,
    #                                                     context_concat_output_from_text_encoder=True,
    #                                                     )
    
    # text_config = FLMRTextConfig.from_pretrained("bert-base-uncased")
    # vision_config = FLMRVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
    # flmr_config = FLMRConfig.from_text_vision_configs(text_config=text_config, vision_config=vision_config)

    text_config = FLMRTextConfig()
    vision_config = FLMRVisionConfig()
    flmr_config = FLMRConfig.from_text_vision_configs(text_config=text_config, vision_config=vision_config)

    # print(flmr_config)
    flmr_model = FLMRModelForRetrieval(config=flmr_config,
                                        query_tokenizer=query_tokenizer,
                                        context_tokenizer=context_tokenizer,
                                       )

    # input()


    # flmr_model = FLMRModelForRetrieval.from_pretrained("./PreFLMR_ViT-G", 
    #                                                     query_tokenizer=query_tokenizer, 
    #                                                     context_tokenizer=context_tokenizer,
    #                                                     # query_concat_output_from_vision_encoder=True,
    #                                                     # query_concat_output_from_text_encoder=True,
    #                                                     # context_concat_output_from_vision_encoder=False,
    #                                                     # context_concat_output_from_text_encoder=True,
    #                                                     # use_transformer_mapping_network=True,
    #                                                     # transformer_mapping_config_base="bert-base-uncased",
    #                                                     # transformer_mapping_num_hidden_layers=1,
    #                                                     # separate_query_and_context_text_encoder=True,
    #                                                 )

    flmr_model = flmr_model.cuda()

    Q_image_features = torch.randn(2, 768)
    D_image_features = torch.randn(4, 768)
    Q_pixel_values = torch.randn(2, 3, 224, 224)
    D_pixel_values = torch.randn(4, 3, 224, 224)
    # FLMR forward
    inputs = dict(
        query_input_ids=Q_encoding['input_ids'],
        query_attention_mask=Q_encoding['attention_mask'],
        query_pixel_values=Q_pixel_values,
        query_image_features=None,
        context_input_ids=D_encoding['input_ids'],
        context_attention_mask=D_encoding['attention_mask'],
        context_pixel_values=D_pixel_values,
        context_image_features=None,
        query_concat_output_from_vision_encoder=True,
        query_concat_output_from_text_encoder=True,
        context_concat_output_from_vision_encoder=True,
        context_concat_output_from_text_encoder=True,
        use_in_batch_negatives=True,
    )
    # # FLMR WIT pretraining Forward
    # inputs = dict(
    #     query_input_ids=None,
    #     query_attention_mask=None,
    #     query_pixel_values=Q_pixel_values,
    #     query_image_features=None,
    #     context_input_ids=D_encoding['input_ids'],
    #     context_attention_mask=D_encoding['attention_mask'],
    #     context_pixel_values=None,
    #     context_image_features=None,
    #     query_concat_output_from_vision_encoder=True,
    #     query_concat_output_from_text_encoder=False,
    #     context_concat_output_from_vision_encoder=False,
    #     context_concat_output_from_text_encoder=True,
    # )

    forward_results = flmr_model.forward(**inputs)
    print(forward_results)

    # flmr_model.save_pretrained("./test_preflmr_model")
    # query_tokenizer.save_pretrained("./test_preflmr_model/query_tokenizer")
    # context_tokenizer.save_pretrained("./test_preflmr_model/context_tokenizer")
    
    exit()
    

    # input("save done.")
    # flmr_model = FLMRModelForIndexing.from_pretrained("bert-base-uncased", 
    #                                                    query_tokenizer=query_tokenizer, 
    #                                                    context_tokenizer=context_tokenizer)
    # print("loading finished")
    # input()

    ## Call ColBERT indexing to index passages
    # generate 1000 passages
    passage_contents = ["Test sentence {}".format(i) for i in range(1000)]
    # generate 1000 random images for each passage, convert them into jpg and save to path
    random_images = torch.randn(1000, 3, 224, 224)
    # convert to jpg with PIL
    to_img = ToPILImage()
    for i, image in enumerate(random_images):
        image = to_img(image)
        image.save(os.path.join("./test_images", "{}.jpg".format(i)))
    
    image_paths = [os.path.join("./test_images", "{}.jpg".format(i)) for i in range(1000)]
    # random image features and convert to numpy
    passage_image_features = np.random.rand(1000, 768)

    from colbert.infra import Run, RunConfig, ColBERTConfig

    # from models.retriever.colbert_utils import MultiModalIndexer
    from colbert import Indexer

    multimodal_docs = False

    # Launch indexer
    with Run().context(RunConfig(nranks=1, root=".", experiment=f"temp_index")):
        nbits = 2
        config = ColBERTConfig(
            nbits=nbits,
            doc_maxlen=512,
        )
        print("indexing with", nbits, "bits")
        if multimodal_docs:
            # custom_collection = [
            #     (passage_content, passage_image_feature, None) for passage_content, passage_image_feature in zip(passage_contents, passage_image_features)
            # ]
            custom_collection = [
                (passage_content, None, image_path) for passage_content, image_path in zip(passage_contents, image_paths)
            ]
        else:
            custom_collection = passage_contents
        
        indexer = Indexer(
            checkpoint="./test_flmr_model", 
            config=config
        )
        indexer.index(
            name=f"temp_index.nbits={nbits}", 
            collection=custom_collection, 
            batch_size=128,
            overwrite=True)
        index_path = indexer.get_index()
        del indexer