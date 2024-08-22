#!/usr/bin/env python3
"""
 Copyright (c) 2018-2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from openvino import Core, get_version
import cv2 as cv
import numpy as np
import logging as log
from time import perf_counter
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_arg():
    parser = ArgumentParser(add_help=False)
    in_args = parser.add_argument_group('Options')
    in_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Help with the script.')
    in_args.add_argument("-m", "--model", help="Required. Path to .xml file with pre-trained model.",
                         required=True, type=Path)
    in_args.add_argument("-d", "--device",
                         help="Optional. Specify a device to infer on (the list of available devices is shown below). Use "
                              "'-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use "
                              "'-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU",
                         default="CPU", type=str)
    in_args.add_argument('-i', "--input", required=True,
                         help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    in_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    in_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    in_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    in_args.add_argument("--no_show", help="Optional. Don't show output.",
                         action='store_true', default=False)
    in_args.add_argument("-u", "--utilization_monitors", default="", type=str,
                         help="Optional. List of monitors to show initially.")
    return parser

##  #FF0000 ->  #ff6600 으로 변환, #00FF00 -> #009966 욿 변환하는 함수 
## Red-Green Color Blindness
## 변환된 픽셀의 수와 비율 계산
def rg_replace_colors(image):
    # 색상 범위 정의 (BGR 형식)
    lower_red = np.array([0, 0, 150 ])  # 빨간색 범위를 넓게 설정
    upper_red = np.array([100, 100, 255])

    lower_green = np.array([0, 200, 0])  # 연두색 범위를 넓게 설정
    upper_green = np.array([100, 255, 100])

    # 원본 이미지의 전체 픽셀 수
    total_pixels = image.shape[0] * image.shape[1]

    # 빨간색을 찾아서 #ff6600 으로 변환
    mask_red = cv.inRange(image, lower_red, upper_red)
    red_pixels_count = np.count_nonzero(mask_red)
    image[mask_red != 0] = [0, 105, 255]

    # 연두색을 찾아서 #009966 (BGR: [115, 158, 0])으로 변환
    mask_green = cv.inRange(image, lower_green, upper_green)
    green_pixels_count = np.count_nonzero(mask_green)
    image[mask_green != 0] = [115, 158, 0]


    # 변환된 픽셀의 수를 합산
    rg_total_changed_pixels = red_pixels_count + green_pixels_count

    # 변환된 픽셀 비율 계산
    rg_percentage_changed = (rg_total_changed_pixels / total_pixels) * 100

    # 결과 출력
    print(f"Red pixels changed: {red_pixels_count}")
    print(f"Green pixels changed: {green_pixels_count}")
    print(f"Total Red and Green pixels changed: {rg_total_changed_pixels} ({rg_percentage_changed:.2f}%)")


    return image

##  blue ->  보라색 으로 변환, 노랑 -> 진녹색 변환하는 함수 
## Blue-Yellow Color Blindness
def by_replace_colors(image):
    # 색상 범위 정의 (BGR 형식)
    lower_blue = np.array([50 , 0, 0])  # 파랑색 범위를 넓게 설정
    upper_blue = np.array([255, 100, 100])

    lower_yellow = np.array([0, 150 , 150 ])  # 연두색 범위를 넓게 설정
    upper_yellow = np.array([100, 255, 255])

    # 원본 이미지의 전체 픽셀 수
    total_pixels = image.shape[0] * image.shape[1]

    # 파랑색찾아서 찾아서 #보라색 으로 변환
    mask_blue = cv.inRange(image, lower_blue, upper_blue)
    blue_pixels_count = np.count_nonzero(mask_blue)
    image[mask_blue != 0] = [128, 0, 128]       #990099

    # 노랑색을 찾아서 #진녹색으로 변환
    mask_yellow = cv.inRange(image, lower_yellow, upper_yellow)
    yellow_pixels_count = np.count_nonzero(mask_yellow)
    image[mask_yellow != 0] = [0, 128, 0]       #009900

    
    # 변환된 픽셀의 수를 합산
    by_total_changed_pixels = blue_pixels_count + yellow_pixels_count

    # 변환된 픽셀 비율 계산
    by_percentage_changed = (by_total_changed_pixels / total_pixels) * 100

    print(f"Blue pixels changed: {blue_pixels_count}")
    print(f"Yellow pixels changed: {yellow_pixels_count}")
    print(f"Total Blue and Yellow pixels changed: {by_total_changed_pixels} ({by_percentage_changed:.2f}%)")

    return image

def main(args):
    cap = open_images_capture(args.input, args.loop)

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    log.info('Reading model {}'.format(args.model))
    model = core.read_model(args.model)

    input_tensor_name = 'data_l'
    input_shape = model.input(input_tensor_name).shape
    assert input_shape[1] == 1, "Expected model input shape with 1 channel"

    inputs = {}
    for input in model.inputs:
        inputs[input.get_any_name()] = np.zeros(input.shape)

    assert len(model.outputs) == 1, "Expected number of outputs is equal 1"

    compiled_model = core.compile_model(model, device_name=args.device)
    output_tensor = compiled_model.outputs[0]
    infer_request = compiled_model.create_infer_request()
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    _, _, h_in, w_in = input_shape

    frames_processed = 0
    imshow_size = (640, 480)
    graph_size = (imshow_size[0] // 2, imshow_size[1] // 4)
    presenter = monitors.Presenter(args.utilization_monitors, imshow_size[1] * 2 - graph_size[1], graph_size)
    metrics = PerformanceMetrics()

    video_writer = cv.VideoWriter()
    if args.output and not video_writer.open(args.output, cv.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (imshow_size[0] * 2, imshow_size[1] * 2)):
        raise RuntimeError("Can't open video writer")

    start_time = perf_counter()
    original_frame = cap.read()
    if original_frame is None:
        raise RuntimeError("Can't read an image from the input")

    while original_frame is not None:
        (h_orig, w_orig) = original_frame.shape[:2]

        if original_frame.shape[2] > 1:
            frame = cv.cvtColor(cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
        else:
            frame = cv.cvtColor(original_frame, cv.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l_rs = cv.resize(img_lab.copy(), (w_in, h_in))[:, :, 0]

        inputs[input_tensor_name] = np.expand_dims(img_l_rs, axis=[0, 1])

        res = infer_request.infer(inputs)[output_tensor]

        update_res = np.squeeze(res)

        out = update_res.transpose((1, 2, 0))
        out = cv.resize(out, (w_orig, h_orig))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

        #! 원본 이미지에서 특정 색상 교체
        rg_color_changed_frame = rg_replace_colors(original_frame.copy())
        by_color_changed_frame = by_replace_colors(original_frame.copy())


        original_image = cv.resize(original_frame, imshow_size)
        grayscale_image = cv.resize(frame, imshow_size)
        colorize_image = (cv.resize(img_bgr_out, imshow_size) * 255).astype(np.uint8)
        lab_image = cv.resize(img_lab_out, imshow_size).astype(np.uint8)
        rg_color_changed_image = cv.resize(rg_color_changed_frame, imshow_size)
        by_color_changed_image = cv.resize(by_color_changed_frame, imshow_size)


        original_image = cv.putText(original_image, 'Original', (25, 50),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (115, 115, 115), 2, cv.LINE_AA)
        grayscale_image = cv.putText(grayscale_image, 'Grayscale', (25, 50),
                                     cv.FONT_HERSHEY_SIMPLEX, 1, (115, 115, 115), 2, cv.LINE_AA)
        colorize_image = cv.putText(colorize_image, 'Colorize', (25, 50),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (115, 115, 115), 2, cv.LINE_AA)
        lab_image = cv.putText(lab_image, 'LAB interpretation', (25, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (115, 115, 115), 2, cv.LINE_AA)
        rg_color_changed_image = cv.putText(rg_color_changed_image, 'Red-Green Color Blindness', (25, 50),
                                         cv.FONT_HERSHEY_SIMPLEX, 1, (115, 115, 115), 2, cv.LINE_AA)
        by_color_changed_image = cv.putText(by_color_changed_image, 'Blue-Yellow Color Blindness', (25, 50),
                                         cv.FONT_HERSHEY_SIMPLEX, 1, (115, 115, 115), 2, cv.LINE_AA)


        # 이미지 가로에 2개씩
        # ir_image = [cv.hconcat([original_image, grayscale_image]),
        #             cv.hconcat([lab_image, colorize_image]),
        #             cv.hconcat([rg_color_changed_image, rg_color_changed_image])]
        # final_image = cv.vconcat(ir_image)
        
        
        # 이미지 가로에 3개씩
        first_row = cv.hconcat([original_image, grayscale_image, colorize_image])
        second_row = cv.hconcat([lab_image, rg_color_changed_image, by_color_changed_image])
        final_image = cv.vconcat([first_row, second_row])


        metrics.update(start_time, final_image)

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(final_image)

        presenter.drawGraphs(final_image)
        if not args.no_show:
            cv.imshow('Colorization Demo', final_image)
            key = cv.waitKey(0)
            if key in {ord("q"), ord("Q"), 27}:
                break
            presenter.handleKey(key)
        start_time = perf_counter()
        original_frame = cap.read()

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)

if __name__ == "__main__":
    args = build_arg().parse_args()
    sys.exit(main(args) or 0)
