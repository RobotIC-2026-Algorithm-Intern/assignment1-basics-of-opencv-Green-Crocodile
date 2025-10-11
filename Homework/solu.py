import cv2
import numpy as np
import json
import os
    
config={}
try:
    with open('Homework/config.json', 'r') as f:#打开文件,只读
        config=json.load(f)#读取JSON数据,并将其转换为Python对象
except FileNotFoundError:
    print(f"Config file not found at path")

class ColorRange:
    def __init__(self):
        self.color_ranges = {}
        self.config=config
        self.load_color_ranges()

    def load_color_ranges(self):#存储从JSON文件读取的颜色范围的数据
        color_ranges = self.config['color_ranges']#获取ConfigManager的实例的load_config的返回值(字典)的'color_ranges'键对应的值
        for color, range_data in color_ranges.items():
            self.color_ranges[color] = [#键：颜色名称;值：颜色范围
                tuple(range_data['lower_hsv']), 
                tuple(range_data['upper_hsv'])
            ]

    def create_color_mask(self, hsv_frame, color_name):#制造掩膜
        if color_name == 'red':#红色的组合掩膜
            mask1 = cv2.inRange(hsv_frame, np.array(self.color_ranges['red1'][0]), np.array(self.color_ranges['red1'][1]))
#检查每个像素是否位于指定的下限 (lower) 和上限 (upper) 范围内。符合条件的像素在输出的二值图像中会被设置为白色 (255)，不符合的则被设置为黑色 (0)
            mask2 = cv2.inRange(hsv_frame, np.array(self.color_ranges['red2'][0]), np.array(self.color_ranges['red2'][1]))
            return cv2.bitwise_or(mask1, mask2)#取并集
        lower, upper = self.color_ranges[color_name]#其他颜色的单一掩膜
        return cv2.inRange(hsv_frame, np.array(lower), np.array(upper))

class BallDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)#cv2.VideoCapture的构造函数,用于后续从视频文件或摄像头捕获视频
        self.config=config#BallDetector是使用ColorRange的上层组件,选择在这里实例化ConfigManager
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        self.color_range = ColorRange()#调用ColorRange的构造函数
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:#0不能作为除数
            print("Warning:30 fps.")
            self.fps = 30
        self.wait_time = max(1, int(1000 / self.fps))#求原视频帧之间的等待时间
        #在main中使用cv2.waitKey(self.wait_time)来控制速度
        
        self.crop_top_ratio = self.config['roi']['crop_top_ratio'] # 从配置文件读取ROI参数
        self.crop_side_ratio = self.config['roi']['crop_side_ratio']
        self.detection_threshold = self.config['detection_threshold']

    def process_frame(self, frame):#处理每一帧
        frame = cv2.flip(frame, 0)
        
        crop_top = int(self.height * self.crop_top_ratio)#裁剪区域
        crop_side = int(self.width * self.crop_side_ratio)
        roi = frame[crop_top:self.height-crop_top, crop_side:self.width-crop_side]#切片
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)#获取HSV图像
        
        red_mask = self.color_range.create_color_mask(hsv, 'red')#制造掩膜
        purple_mask = self.color_range.create_color_mask(hsv, 'purple')
        blue_mask = self.color_range.create_color_mask(hsv, 'blue')
       
        red_ratio = cv2.countNonZero(red_mask) / (roi.shape[0] * roi.shape[1])#计算掩膜中像素的比例
        purple_ratio = cv2.countNonZero(purple_mask) / (roi.shape[0] * roi.shape[1])
        blue_ratio = cv2.countNonZero(blue_mask) / (roi.shape[0] * roi.shape[1])
        
        ball_text = 'no ball'
        max_ratio = max(red_ratio, purple_ratio, blue_ratio)

        if max_ratio > self.detection_threshold:# 使用配置文件中的检测阈值
            if max_ratio == red_ratio:
                ball_text = 'red ball'
            elif max_ratio == purple_ratio:
                ball_text = 'purple ball'
            else:
                ball_text = 'blue ball'
            
        cv2.putText(frame, ball_text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,0,0), 3)#将文字写入视频
        
        return frame#返回处理后的帧

    def process_video(self):
        while True:#循环读取视频的每一帧
            ret, frame = self.cap.read()#读取下一帧
            if not ret:
                break  
            processed_frame = self.process_frame(frame)#调用单帧处理函数    
            cv2.imshow('Ball Detection', processed_frame)#在Ball Detection窗口显示图像
                
            key = cv2.waitKey(self.wait_time)#使视频原速播放
            if key & 0xFF == ord('q'):#创建退出按钮
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = r'res\output1.avi'  
    detector = BallDetector(video_path)
    detector.process_video()

if __name__ == "__main__":
    main()