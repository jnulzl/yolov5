import argparse
import cv2
import os

def main(dir_root,sub_dir='train',ext_name = 'jpg', wait_time = 0):
    img_root = os.path.join(dir_root, 'images', sub_dir)
    img_list = os.listdir(img_root)
    for img_name in img_list:
        img_path = os.path.join(img_root, img_name)
        label_path = img_path.replace('images','labels').replace(ext_name,'txt')
        if not os.path.exists(label_path):
            print('Passing ' + img_path)
            continue
        img = cv2.imread(img_path)
        if img is None:
            print('Passing ' + img_path)
            continue
        print('Show image : ' + img_path)
        height, width = img.shape[:2]
        with open(label_path,'r') as fpR:
            lines = fpR.readlines()
            for line in lines:
                _line = line.split()
                line = [float(item) for item in _line]
                roi_width, roi_height = int(line[3] * width), int(line[4] * height)
                min_x = int(line[1] * width  - roi_width  / 2)
                min_y = int(line[2] * height - roi_height / 2)
                # print(min_x, min_y)
                cv2.rectangle(img, (min_x, min_y), (min_x + roi_width, min_y + roi_height),
                              (255, 0, 255), 2, cv2.FONT_HERSHEY_PLAIN)
        cv2.imshow("Demo", img)
        if 27 == cv2.waitKey(wait_time):
            break                
                                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show Yolov5 train image')
    # general
    parser.add_argument('--root_dir', 
                         default='', 
                         type = str,
                         help='The root directory of yolov5 train dataset.')
    parser.add_argument('--sub_dir',
                        default='train',
                        type = str,
                        help='train or val directory.')
    parser.add_argument('--ext_name', 
                         default='jpg', 
                         type = str,
                         help='The extension name of image.')
    parser.add_argument('--wait_time', 
                         default=0, 
                         type = int,
                         help='The wait time of show image.')
    args = parser.parse_args()
    dir_root = args.root_dir
    sub_dir = args.sub_dir
    main(args.root_dir, args.sub_dir, args.ext_name, args.wait_time)
