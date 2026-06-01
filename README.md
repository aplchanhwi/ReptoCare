# ReptoCare — 반려 파충류 종 분류 iOS 앱

> CoreML 기반 온디바이스 AI로 반려 파충류 종을 판별하고, LLM을 통해 사육 정보를 제공하는 iOS 앱

---

## 문제 정의

반려 파충류 시장은 연평균 7.3% 성장 중이나, 강아지·고양이와 달리 종별 사육 정보를 빠르게 확인할 수 있는 모바일 서비스가 부재합니다.

본 프로젝트는 파충류 사진 한 장으로 종을 자동 분류하고, 해당 종의 사육 방법을 LLM으로 제공하는 iOS 앱을 구현합니다.

**핵심 파이프라인**
```
사진 촬영 / 갤러리 선택
        ↓
CoreML 온디바이스 추론 (종 분류)
        ↓
LLM API (사육 정보 생성)
        ↓
SwiftUI 결과 화면 렌더링
```

---

## 📦 데이터셋

**출처**: [Roboflow — ReptilesDataset](https://universe.roboflow.com/pet-b3kq0/reptilesdataset)

| 항목 | 내용 |
|------|------|
| 전체 클래스 | 45종 |
| 사용 클래스 | **15종** (반려용 위주 필터링) |
| 원본 이미지 수 | 약 4,823장 (15종 기준) |
| Train / Valid / Test | 3,397 / 946 / 480장 |
| 이미지 포맷 | JPG, PNG |
| 라벨 구조 | 폴더 구조 (ImageFolder 호환) |

**선별 기준**: 실제 반려동물로 키우는 종 위주로 15종 선별

| 카테고리 | 종 |
|---|---|
| 도마뱀 (7종) | Leopard Gecko, African Fat-tail Gecko, Crested Gecko, Central Bearded Dragon, Common Bluetongue, Common Green Iguana, Argentine Black and White Tegu |
| 뱀 (5종) | Ball Python, Boa Constrictor, Corn Snake, Milk Snake, Western Hognose Snake |
| 거북이 (3종) | Red-eared Slider, Leopard Tortoise, Red-footed Tortoise |

**전처리 방법**

Train 데이터에만 실시간 증강 적용 (Valid/Test는 원본 유지)

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## 🤖 사용 모델

### MobileNetV2
- ImageNet 사전학습 모델 기반 Transfer Learning
- 마지막 Classifier Layer를 15개 클래스로 교체
- 모델 크기: 약 14MB → iOS 온디바이스 탑재 최적화

```
Input (224×224×3)
    ↓
MobileNetV2 Backbone (Pretrained, ImageNet)
    ↓
Classifier → Linear(1280, 15)
    ↓
Output (15 classes)
```

### ResNet34
- ImageNet 사전학습 모델 기반 Transfer Learning
- FC Layer를 15개 클래스로 교체
- 모델 크기: 약 87MB

```
Input (224×224×3)
    ↓
ResNet34 Backbone (Pretrained, ImageNet)
    ↓
FC → Linear(512, 15)
    ↓
Output (15 classes)
```

---

## ⚙️ 학습 설정 (Hyperparameter)

| 항목 | 값 |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 32 |
| Epochs | 20 |
| Loss Function | CrossEntropyLoss |
| Pretrained Weight | ImageNet1K_V1 |
| 학습 환경 | Google Colab (GPU) |

---

## 📊 성능 비교

### 최종 Test Accuracy

| 모델 | Test Accuracy | 모델 크기 |
|---|---|---|
| **MobileNetV2** | **90.83%** | 14MB |
| ResNet34 | 83.75% | 87MB |

### 성능 측정 기준
- Validation Accuracy (학습 중 매 Epoch)
- Test Accuracy (최종 평가, 모델이 한 번도 보지 않은 데이터)
- Confusion Matrix (클래스별 오분류 패턴 분석)

### MobileNetV2 Confusion Matrix
> 90.83% 달성, 데이터 부족 클래스(Boa constrictor 126장)에서 상대적으로 낮은 성능

### ResNet34 Confusion Matrix
> 83.75% 달성, 생김새가 유사한 종(Gecko 계열) 간 혼동 발생 (Leopard Gecko와 African Fat-tail Gecko 간)

**분석 인사이트**
- 경량 모델인 MobileNetV2가 ResNet34 대비 7%p 높은 정확도 기록
- 데이터가 적은 클래스(126장)에서도 전이학습으로 최소 33% 이상 방어
- 파충류 이미지 특성상 세밀한 특징 추출보다 전반적인 패턴 인식이 효과적

---

## 🍎 CoreML 변환

PyTorch → CoreML 변환 파이프라인

```python
# TorchScript 변환 후 CoreML로 export
traced = torch.jit.trace(wrapped_model, dummy_input)
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="input_1", shape=(1,3,224,224), ...)],
    classifier_config=ct.ClassifierConfig(class_labels),
)
mlmodel.save("MobileNetV2.mlpackage")
```

| 항목 | 내용 |
|---|---|
| 변환 도구 | coremltools 9.0 |
| 출력 포맷 | .mlpackage (iOS 15+) |
| 정규화 | CoreMLWrapper로 모델 내부에 포함 |
| 최소 타겟 | iOS 15 |

---

## 📱 iOS 앱 구조
<!-- 숨길거에요
```
ReptoCare/
├── ContentView.swift       # 메인 화면 (사진 선택/촬영)
├── ClassifierView.swift    # 분류 결과 화면
├── CareInfoView.swift      # LLM 사육 정보 화면
├── MLClassifier.swift      # CoreML 추론 로직
├── LLMService.swift        # LLM API 연동
├── Models/
│   ├── MobileNetV2.mlpackage
│   └── ResNet34.mlpackage
└── Assets.xcassets
```
-->

**기술 스택**

| 항목 | 기술 |
|---|---|
| UI | SwiftUI |
| ML 추론 | CoreML, Vision Framework |
| 카메라/갤러리 | AVFoundation, PhotosUI |
| LLM 연동 | Gemini API |
| 최소 지원 | iOS 15+ |

---

## 📈 학습 Log

학습 로그는 [Google Colab Notebook](링크) 에서 확인 가능합니다.

```
[Epoch 01] Train Loss: x.xxxx | Acc: xx.xx% | Val Loss: x.xxxx | Acc: xx.xx%
...
[Epoch 20] Train Loss: x.xxxx | Acc: xx.xx% | Val Loss: x.xxxx | Acc: xx.xx%
```

---
<!-- 숨길거에요
## 🗂️ 코드 구조

```
ReptoCare/
├── Colab/
│   ├── 01_data_download.ipynb      # 데이터 다운로드 및 필터링
│   ├── 02_train.ipynb              # 전이학습 (MobileNetV2, ResNet34)
│   └── 03_coreml_convert.ipynb     # CoreML 변환
├── iOS/
│   └── ReptoCare/                  # Xcode 프로젝트
└── README.md
```

---


## 🛠️ 실행 방법

**Colab 학습**
```bash
# 1. 데이터 다운로드
pip install roboflow
# Roboflow API로 데이터셋 다운로드

# 2. 학습
# 02_train.ipynb 실행

# 3. CoreML 변환
pip install coremltools
# 03_coreml_convert.ipynb 실행
```

**iOS 앱 실행**
```
1. Xcode에서 ReptoCare.xcodeproj 열기
2. MobileNetV2.mlpackage, ResNet34.mlpackage 추가
3. 실기기 또는 시뮬레이터에서 실행
```
---
<!-- 숨길거에요
## 👤 개발자

- **강찬휘** | iOS Developer
- GitHub: [링크]
-->
