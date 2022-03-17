# Gesture Recognition

## Description

BlazePalm에서 추론된 포인트로 제스처 인식하기

정해진 제스처 = swipe(상.하.좌.우) 제스처, pinch 제스처, quiet 제스처, grab 제스처 


## Requirements
```
pip install -r requirements.txt
```


## 모델 구축

### 1. collect dataset

   - 연속 동영상 촬영해서 손 keypoint 좌표를 뽑고 각 keypoint들의 각도를 feature로 저장
 
   - 총 226,280개 데이터 → './dataset/raw_data.npy'  

  -  `actions = ['palm', 'quiet', 'grab', 'pinch']`

    
### 2. train

   - 학습된 모델 : gesture_model.h5 → onnx모델로 변경해서 사용

   - 학습 정확도 - 99%


## Demo

   - 얼굴 detect하고, 제스쳐 영역 만들어 영역 내에서만 제스처인식이 가능하게
   - quiet 제스처는 입 주변 roi에서만 가능하게
   - grab 제스처는 회전 각도 계산
   - swipe 제스처는 palm 제스처가 인식 되면 이동 알고리즘을 사용해서 상.하.좌.우 판단 → 손바닥 이동 좌표 비율로 계산

   ![demo](gesture.gif)