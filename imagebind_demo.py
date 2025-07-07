# apt-get -y update && apt-get install -y ffmpeg

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torch.nn.functional as F
from utils import copy_audio_wav
import random

text_list=["A man."]
video_paths=["video/output.mp4", "video/output3.mp4"]

for video in video_paths:
    copy_audio_wav(video, video.replace('mp4', 'wav'))

audio_paths=["video/output.wav", "video/output3.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(embeddings[ModalityType.VISION])
print(embeddings[ModalityType.TEXT])
print(embeddings[ModalityType.AUDIO])

# 计算余弦相似度（自动处理批量维度）
video_cos_sim = F.cosine_similarity(embeddings[ModalityType.VISION][0], embeddings[ModalityType.VISION][1], dim=-1)

# 余弦距离 = 1 - 余弦相似度
video_cos_distance = 1 - video_cos_sim

print(f"video cos Similarity: {video_cos_sim.item():.4f}")
print(f"video cos Distence: {video_cos_distance.item():.4f}")

# 计算余弦相似度（自动处理批量维度）
audio_cos_sim = F.cosine_similarity(embeddings[ModalityType.AUDIO][0], embeddings[ModalityType.AUDIO][1], dim=-1)

# 余弦距离 = 1 - 余弦相似度
audio_cos_distance = 1 - audio_cos_sim

print(f"audio cos Similarity: {audio_cos_sim.item():.4f}")
print(f"audio cos Distence: {audio_cos_distance.item():.4f}")

video_score_weight = 0.4
audio_score_weight = 0.4
runtime_weight = 0.2

base_runtime = 10
random_runtime = round(random.uniform(5, 20))
diff_runtime = random_runtime - 10

score = video_score_weight*video_cos_sim + audio_score_weight*audio_cos_sim + runtime_weight*diff_runtime
print(video_score_weight*video_cos_sim, audio_score_weight*audio_cos_sim, runtime_weight*diff_runtime, score)
# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
