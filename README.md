# Anyractive Gesture

## Description

BlazePalm에서 추론된 포인트나 손 이미지로 제스처 인식하기

정해진 제스처 = 좌.우.위.아래 swipe 제스처, pinch 제스처, Palm 제스처, quiet 제스처, grab 제스처 

거리는 1~2미터도 가능하게


## Requirements
```
pip install -r requirements.txt
```


## 모델 구축

### 1. collect dataset

   [create_dataset.py](https://github.com/mjw2705/Anyractive/blob/main/create_dataset.py)

   연속 동영상 촬영해서 손가락 포인트 좌표 뽑기 / swipe 동작은 알고리즘으로 
 
   총 201,926개 데이터 - './dataset/raw_data.npy'  

   `actions = ['palm', 'quiet', 'grab', 'pinch']`

    
### 2. train
   
   [train.ipynb](https://github.com/mjw2705/Anyractive/blob/main/train.ipynb)

   모델 : sequential

   학습 모델 - gesture_model.h5 → onnx모델로 변경해서 사용

  
### 3. test & demo
   
   [test.py](https://github.com/mjw2705/Anyractive/blob/main/test.py)

   얼굴 detect하고, 제스쳐 영역 만들기
   - 얼굴 detect 방법
       1. mediapipe pose 사용 → 결정
       2. facedetector.onnx 모델 사용
    
   제스쳐 영역 내에서만 제스쳐 인식 할 수 있게, quiet 제스쳐는 입 주변에서만 가능하도록 
   
   grab 제스쳐는 회전 각도 계산
   
   swipe은 palm 제스쳐 상태에서 상.하.좌.우 이동 알고리즘 → 손바닥 이동 좌표 비율로 계산
   
    
   
    
