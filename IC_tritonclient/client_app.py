import argparse
from functools import partial
import os
import sys

from PIL import Image
import numpy as np
from attrdict import AttrDict

import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from asyncio import streams
from urllib import response
import requests
import time
import urllib.request

FLAGS = None

def parse_model(model_metadata, model_config):

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    
    FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
    input_config.format = FORMAT_ENUM_TO_INT[input_config.format]
    
    h = input_metadata.shape[1 if input_batch_dim else 0]
    w = input_metadata.shape[2 if input_batch_dim else 1]
    c = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, c, h, w, input_config.format,
            input_metadata.datatype)


def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
   
    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return scaled


def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """
    output_array = results.as_numpy(output_name)

    # Include special handling for non-batching models
    for results in output_array:
        for result in results:
            if output_array.dtype.type == np.object_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
    client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(output_name, class_count=FLAGS.classes)
    ]

    yield inputs, outputs, model_name, ""


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


if __name__ == '__main__':
    model_name = "inception_graphdef"

    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=3,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    FLAGS = parser.parse_args()

    try:
    # No async requests => concurrency=1
    	triton_client = httpclient.InferenceServerClient(
        	url=FLAGS.url, verbose=FLAGS.verbose, concurrency=1)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    model_metadata, model_config = convert_http_metadata_config(
        model_metadata, model_config)

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config)

    # Preprocess the images into input data according to model
    # requirements
    
    #img = Image.open("casa.png")
    
    url = "http://192.168.0.33/take_photo"
    try:
    	requests.get(url)
    except Exception:
    	pass

    time.sleep(0.5)

    url = "http://192.168.0.33/photo"

    file_name = "pic.jpg"
    urllib.request.urlretrieve(url, file_name)

    img = Image.open(file_name)
    image_data = preprocess(img, format, dtype, c, h, w, 'INCEPTION',
		   'HTTP')

    requests = []
    responses = []
    result_filenames = []
    request_ids = []


    repeated_image_data = []

    repeated_image_data.append(image_data)

    batched_image_data = np.stack(repeated_image_data, axis=0)

    # Send request
    for inputs, outputs, model_name, model_version in requestGenerator(
            batched_image_data, input_name, output_name, dtype, FLAGS):
                
        responses.append(
             triton_client.infer(model_name,
                                 inputs,
                                 request_id="1",
                                 model_version="",
                                 outputs=outputs))


    for response in responses:    
        this_id = response.get_response()["id"]
        postprocess(response, output_name, 1, max_batch_size > 0)

    print("PASS")
