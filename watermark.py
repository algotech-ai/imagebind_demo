from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import math

def generate_text_watermark(text, width, height, output_path, spacing=20):
    # 创建一个稍大的画布，确保旋转后覆盖整个区域
    diagonal = int(math.hypot(width, height) * 1.5)
    img = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = 15
    try:
        font = ImageFont.truetype("/path/to/font.ttf", font_size)
    except:
        font = ImageFont.load_default(size=font_size)  # 备用默认字体
    
    # 获取文字边界框
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 计算每行和每列的文字数量，带间隔
    cols = math.floor((diagonal - text_width) / (text_width + spacing))
    rows = math.floor((diagonal - text_height) / (text_height + spacing))
    
    # 水平排列文字
    for i in range(rows):
        for j in range(cols):
            x = j * (text_width + spacing)
            y = i * (text_height + spacing)
            draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255*0.1)))  # 白色，50% 透明
    
    # 裁剪并旋转45度
    img = img.rotate(45, expand=True)
    # 裁剪到视频尺寸，居中
    left = (img.width - width) // 2
    top = (img.height - height) // 2
    img = img.crop((left, top, left + width, top + height))
    img.save(output_path)

def apply_watermark_to_video(video_path, watermark_path, output_path):
    video = VideoFileClip(video_path)
    watermark = (ImageClip(watermark_path)
                 .with_duration(video.duration))
    final = CompositeVideoClip([video, watermark])
    final.write_videofile(output_path)

def main():
    video_path = "input.mp4"
    text = "NEZA"  # 文字 logo
    watermark_path = "watermark.png"
    output_path = "output.mp4"
    
    # 获取视频尺寸
    video = VideoFileClip(video_path)
    width, height = int(video.w), int(video.h)
    
    # 生成文字水印图片
    generate_text_watermark(text, width, height, watermark_path, spacing=50)
    
    # 应用水印到视频
    apply_watermark_to_video(video_path, watermark_path, output_path)
    print(f"水印已添加到视频，输出为 {output_path}")

if __name__ == "__main__":
    main()