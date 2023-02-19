# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2
import paddle


def process(src_image_dir, save_dir):
    # load
    model = paddle.jit.load('./mymodel/model')
    model.eval()

    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    result = {}
    for image_path in image_paths:
        filename = os.path.split(image_path)[1]
        # do something
        img = cv2.imread(image_path)
        # get img infor
        h, w, c = img.shape
        # pre-process of img
        img = paddle.vision.transforms.resize(img, (512,512), interpolation='bilinear')
        img = img.transpose((2,0,1))
        img = img/255
        img = paddle.to_tensor(img).astype('float32')
        img = paddle.reshape(img, [1]+img.shape)

        pre = model(img)
        pre = pre[0]
        pre[pre>1]=1
        pre[pre<0]=0
        pre = pre.tolist()
        x1, y1, x2, y2, x3, y3, x4, y4 = pre
        x1, x2, x3, x4 = [int(x*w) for x in [x1, x2, x3, x4]]
        y1, y2, y3, y4 = [int(y*h) for y in [y1, y2, y3, y4]]

        xmin = min(x1,x2,x3,x4)
        xmax = max(x1,x2,x3,x4)
        ymin = min(y1,y2,y3,y4)
        ymax = max(y1,y2,y3,y4)

        if filename not in result:
            result[filename] = []
        result[filename].append({
            "box": [xmin, ymin, xmax, ymax],
            "lb": [x1, y1],
            "lt": [x2, y2],
            "rt": [x3, y3],
            "rb": [x4, y4],
        })
    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)