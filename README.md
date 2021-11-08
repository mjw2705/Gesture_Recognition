# Anyractive Gesture

## Description

BlazePalm에서 추론된 포인트나 손 이미지로 제스처 인식하기

정해진 제스처 = 좌.우.위.아래 스와이프 제스처, pinch 제스처, Palm 제스처, quiet 제스처, grab 제스처 

거리는 1~2미터도 가능하게

## 모델 구축 계획

1. 데이터셋 수집
    
    연속 동영상 촬영해서 손가락 포인트 좌표 뽑기 / swipe 동작은 알고리즘 짜는게 더 정확할 듯
    
    총 201,926개 data - './dataset/raw_data.npy'  

`actions = ['palm', 'quiet', 'grab', 'pinch']`

    
2. 모델 생성 & 학습
    
    모델 : swipe 학습 할 때는 LSTM 아니면 CNN
    
    사용 모델 - gesture_model.h5
    
  
3. 테스트
    
    얼굴 detect하고, 제스쳐 영역 만들기
    
    제스쳐 영역 내에서만 제스쳐 인식 할 수 있게, quiet 제스쳐는 입 주변에서만 
    
    얼굴 detect 방법
    1. mediapipe pose 사용 → 결정
    2. facedetector.onnx 모델 사용
