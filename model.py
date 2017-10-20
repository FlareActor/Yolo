# -*- coding: UTF-8 -*-
"""Run a YOLO_v2 style detection model on test images."""
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from yad2k.models.keras_yolo import yolo_eval, yolo_head


class Yolo:
    def __init__(self, model=None, score_threshold=0.3, iou_threshold=0.5):
        """
        初始化
        :param model:  yolo模型
        :param test_path: 测试数据集文件夹路径
        :param output_path: 结果输出文件夹路径
        :param score_threshold: bbox分类置信度
        :param iou_thresshold: bbox面积重叠率IOU
        """
        self.model = model if model is not None else load_model('yolo.h5')
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._classes_path = 'coco_classes.txt'
        self._anchors_path = 'yolo_anchors.txt'
        pass

    def predict(self, test_path, output_path=None, draw_box=True, targets=None, targets_threshold=0.8):
        """
        检测图像
        :param test_path: 要检测的文件夹路径
        :param output_path: 输出文件夹路径
        :return: bboxes list
        """
        assert test_path is not None
        if targets is not None:
            assert isinstance(targets,list)

        sess = K.get_session()

        # 加载类别
        with open(self._classes_path) as f:
            class_names = [c.strip() for c in f.readlines()]

        # 加载anchor尺寸
        with open(self._anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)

        if output_path is not None and not os.path.exists(output_path):
            os.mkdir(output_path)

        num_classes = len(class_names)  # default:80
        num_anchors = len(anchors)  # default:5*2
        # TODO: Assumes dim ordering is channel last
        model_output_channels = self.model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'

        # Check if model is fully convolutional, assuming channel last order.
        model_image_size = self.model.layers[0].input_shape[1:3]
        is_fixed_size = model_image_size != (None, None)

        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(self.model.output, anchors, len(class_names))
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            yolo_outputs,
            input_image_shape,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        images_bboxes = {}

        for image_file in tqdm(os.listdir(test_path)):
            try:
                image_type = imghdr.what(os.path.join(test_path, image_file))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue

            image = Image.open(os.path.join(test_path, image_file))
            if is_fixed_size:  # TODO: When resizing we can use minibatch input.
                resized_image = image.resize(
                    tuple(reversed(model_image_size)), Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')
            else:
                # Due to skip connection + max pooling in YOLO_v2, inputs must have
                # width and height as multiples of 32.
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                resized_image = image.resize(new_image_size, Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    self.model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            # print('Found {} boxes for {}'.format(len(out_boxes), image_file))

            font = ImageFont.truetype(
                font='FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            bboxes = []

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                
                if targets is not None:
                    if c not in targets or score< targets_threshold:
                        continue

                label = '{} {:.2f}'.format(predicted_class, score)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                # print(label, (left, top), (right, bottom))

                bboxes.append((left, top, right - left, bottom - top, c, predicted_class, score))
                # 绘制BBOX
                if output_path is not None and draw_box:
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for j in range(thickness):
                        draw.rectangle(
                            [left + j, top + j, right - j, bottom - j],
                            outline=colors[c])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=colors[c])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw
            # 有符合条件的bbox才保存并记录
            if len(bboxes)>0:
                images_bboxes.update({image_file: bboxes})
                if output_path is not None:
                    image.save(os.path.join(output_path, image_file), quality=100)

        # sess.close()
        return images_bboxes

# class BBox:
#     def __init__(self, x, y, w, h, class_index, class_name, score):
#         """
#         存储BBOX相关信息
#         :param x: 左上角x坐标
#         :param y: 左上角y坐标
#         :param w: bbox宽度
#         :param h: bbox高度
#         :param class_index: bbox所属类标号
#         :param class_name: bbox所属类名称
#         :param score: 所属分类置信度
#         """
#         self.x = x
#         self.y = y
#         self.w = w
#         self.h = h
#         self.class_index = class_index
#         self.class_name = class_name
#         self.score = score
