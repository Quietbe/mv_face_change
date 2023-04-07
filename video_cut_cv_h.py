from pathlib import Path
import cv2
import os


# 将视频video_path分割成图片和音频文件，保存到save_path文件夹中
def video2mp3_img(video_path, save_path):
    def video_split(video_path, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(save_path + '/' + str(i) + '.jpg', frame)
                i += 1
            else:
                break
        cap.release()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 视频分割
    video_split(video_path, save_path)
    # 视频转音频
    os.system("ffmpeg -i {} -vn -acodec copy {}/audio.mp3".format(video_path, save_path))
    # 音频转wav
    # os.system("ffmpeg -i {}/audio.mp3 {}/audio.wav".format(save_path, save_path))

# video2mp3_img(video_path, save_path)

"""
人脸替换修复处理
"""
def face_replace(user_path=""):
    import threading
    from pathlib import Path

    import cv2
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    max_workers = 2
    semaphore = threading.Semaphore(max_workers)

    def my_function(img_path):

        with semaphore:
            print(f"{img_path}  开始")
            image_face_fusion = pipeline(Tasks.image_face_fusion,
                                   model='damo/cv_unet-image-face-fusion_damo')
            template_path = img_path
            filename = os.path.splitext(os.path.basename(img_path))[0]
            # 替换面部依赖
            # user_path = "刘德华.jpg"
            result = image_face_fusion(dict(template=template_path, user=user_path))
            cv2.imwrite(f'video_imgout/{filename}.jpg', result[OutputKeys.OUTPUT_IMG])
            print(f"{filename}.png ok")
    threads = []
    BASE_PATH = os.path.dirname(__file__)
    for dirpath, dirnames, filenames in os.walk(BASE_PATH + "/video_img"):
        for filename in filenames:
            print(filename)
            if filename.endswith('.jpg'):
                file_path = Path(os.path.join(dirpath, filename))
                t = threading.Thread(target=my_function, args=(str(file_path),)).start()
                # threads.append(t)



# 将video_imgout文件夹中的图片合成视频并且添加音频文件video_img/audio.mp3
def img2mp4(video_path, save_name):
    BASE_PATH = os.path.dirname(__file__)
    # 读取img size
    img = cv2.imread("video_imgout/0.jpg")
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    print(size)
    # videoWrite = cv2.VideoWriter('img_dir//abcd.mp4', -1, 25, size)  # 写入对象 1 file name  3: 视频帧率
    files = []
    for dirpath, dirnames, filenames in os.walk(BASE_PATH + "/video_imgout"):
        for filename in filenames:
            fileName = Path(os.path.join(dirpath, filename))
            files.append(os.path.join(dirpath, filename))

    files = [file.replace('\\', '/') for file in files]
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    print(files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWrite = cv2.VideoWriter(f'{BASE_PATH}/videos/ldh.mp4', fourcc, 25, size)  # 写入对象 1 file name  3: 视频帧率
    for i in files:
        img = cv2.imread(str(i))
        videoWrite.write(img)
    print(f'{BASE_PATH}/videos/{save_name}.mp4')
    # 将video_img中的音频文件添加到视频中
    # os.system("ffmpeg -i {}/videos/ldh.mp4 -i {}/video_img/audio.mp3 -c:v copy -c:a aac -strict experimental {}/videos/ldh.mp4".format(BASE_PATH, BASE_PATH, BASE_PATH))


if __name__ == '__main__':
    BASE = os.path.dirname(__file__)
    video_path = os.path.join(BASE, "videos/demo.mp4")  # 视频路径
    save_path = os.path.join(BASE, "video_img")  # 保存路径

    # 视频  ==> imgs
    # video2mp3_img(video_path, save_path)
    # # 人脸替换
    # face_replace(user_path='zsy.jpg')
    # # imgs ==> 视频
    # img2mp4(video_path, save_name='zsy')