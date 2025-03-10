# 사과게임 인공지능 프로젝트

## 개요
이 프로젝트는 사과게임에서 고득점(120점 이상)을 달성하기 위한 인공지능 개발을 목표로 합니다.  
사과게임은 10×17 크기의 2차원 배열에 1부터 9까지의 숫자가 무작위로 배치되며, 플레이어가 선택한 직사각형 영역 내 숫자들의 합이 정확히 10이 되면 해당 숫자들이 삭제되고 점수를 획득하는 게임입니다.  
숫자 삭제 후 남은 빈 칸은 이후 조합에 영향을 미치므로, 제한시간 120초 내에 최대한 많은 숫자를 제거하여 고득점을 하는입니다.
이 프로젝트에서는 크게 3가지의 분기점이 있습니다. 분기점이란 프로젝트의 영향을 끼치는 중요한 선택들을 의미하며 선택의 이유와 자세한 과정은 아래에 적어놓았습니다.

첫 번째는 데이터 인식방법입니다. 2차원배열과 숫자라는 사과게임의 보드 특성상 CNN을 사용하였습니다.
두 번쨔는 학습 방법입니다. 학습방법에 가장 중요한 결정을 끼친 것은 학습데이터입니다. 학습데이터는 10*17 사이즈에 1~9의 랜덤값을 배치해서 만들었습니다. 유의미한 학습데이터를 무한정 만들어 낼 수 있기 때문에 이를 고려하여 PPO를 사용하였습니다.
세 번째는 값 조절입니다. 값 조절이란 말 그래도 다양한 변수들의 값을 조정하는 것입니다. 예시를 들면 성공보상과 실패패널티가 있습니다. 성공보상이 너무 적으면 올바른 조합을 찾아내지 못하고, 실패 패널티가 너무 과하면 학습이 진행되지 않습니다. 시행착오를 겪으며 실제 결과값을 보며 변수값을 조정했습니다.

## 데이터 인식 방법으로 CNN 선택 이유
  
### 1. 공간적 특성 추출에 최적화
- **게임판의 구조:**  
  사과게임의 게임판 10×17 크기의 2차원 배열로, 각 셀에 1~9의 숫자가 배치되어 있음 
- **CNN의 강점:**  
  합성곱 신경망(CNN)은 합성곱 필터를 사용해 이미지나 2차원 배열의 지역적 패턴과 구조를 효과적으로 학습하기 때문에 게임판 내에서 숫자들의 배치와 국소적인 패턴(예: 합이 10에 근접하는 영역)을 인식하는 데 매우 유리

### 2. 파라미터 효율성과 학습 효율성
- **효율적인 모델링:**  
  CNN은 입력 전체가 아닌, 작은 지역 영역에 집중하여 필터를 적용하기 때문에 전결합 신경망(MLP)보다 적은 파라미터로 높은 표현력을 확보 가능
- **과적합 감소:**  
  지역적 특징 학습에 초점을 맞춤으로써, 게임판 전체의 복잡한 구조를 효과적으로 처리하면서도 과적합 위험 감소

### 3. 고득점 전략 지원
- **전략적 판단:**  
  제한시간 내에 최대한 많은 숫자를 제거하여 120점 이상의 고득점을 달성해야 하는 게임 특성상, 빠르고 정확하게 유의미한 패턴(즉, 숫자들의 합이 10에 근접하거나 정확히 10이 되는 조합)을 인식하는 능력이 필수적임
- **CNN의 역할:**  
  CNN은 게임판의 지역적 조합을 신속하게 분석하여, 효과적인 행동(예: 숫자 제거 조합 선택)을 위한 정보를 추출하는 데 최적의 성능을 발휘

## 대체 데이터 인식 방식과의 비교

- **MLP (전결합 신경망):**
  - 전체 입력을 한 번에 처리하지만, 공간 정보를 효과적으로 활용하지 못해 게임판의 지역적 패턴 인식에 한계점이 존재

- **RNN/LSTM:**
  - 시퀀스 데이터를 처리하는 데 유리하지만, 2차원 배열의 공간적 특징을 추출에 부적합함

- **Vision Transformer (ViT):**
  - 전역적인 관계를 학습할 수 있으나, 대규모 데이터셋이 요구되고 계산 복잡도가 높아 효율성이 떨어짐

- **Capsule Networks / GNN:**
  - 객체의 다양한 속성과 관계를 모델링하는 데 강점을 가짐. 이번 프로젝트에서는 데이터가 노드로 연결되어 있지도 않을 뿐더러 2차원의 데이터를 처리하기엔 너무 효율성이 떨어짐
 
# PPO 학습법 선택 이유

사과 게임은 한 번의 숫자 삭제가 전체 보드의 상태를 급격하게 변화시키고, 삭제 후 생긴 빈 칸과 불필요한 숫자들이 이후 가능한 조합을 방해하는 특성이 있음 
이러한 동적이고 복잡한 환경에서는 PPO가 DQN보다 유리하며, 다음과 같은 이유로 PPO를 선택함

## 1. 빠르게 변화하는 상태 분포와 안정적인 업데이트
- **최신 데이터 기반 on-policy 학습:**  
  PPO는 최신 정책으로부터 직접 데이터를 수집하여 학습하므로, 환경의 급격한 상태 변화에 신속하게 대응 가능 
  반면, DQN은 리플레이 버퍼를 통한 과거 데이터 재활용으로 인해 현재 상태와의 불일치 문제가 발생할 가능성 존재
  
- **클리핑 및 신뢰 영역 최적화:**  
  PPO는 정책 업데이트 시 클리핑(clipping) 기법을 사용해 변경 폭을 제한함으로써, 급격한 성능 저하 없이 안정적인 학습을 보장

## 2. 효과적인 샘플 활용과 실시간 데이터 반영
- **온라인 학습의 장점:**  
  무한하게 양질의 데이터를 생성할 수 있기 때문에, PPO는 환경과의 지속적인 상호작용을 통해 실시간 데이터를 반영하며 정책을 업데이트 용이
  
- **온정책(on-policy) 학습:**  
  PPO는 최신 정책으로부터 수집된 데이터를 이용해 학습함 -> 변화하는 게임 환경에 민감하게 대응하고 적절한 행동을 유도

## 3. 효과적인 탐험과 복잡한 상태 공간 처리
- **다양한 행동 탐색:**  
  PPO는 엔트로피 보너스를 통해 다양한 행동을 탐색하도록 유도.  
  사과 게임과 같이 한 번의 행동이 이후 가능한 조합에 큰 영향을 미치는 환경에서, 충분한 탐험은 최적의 전략 도출에 필수적임
  
- **액터-크리틱 구조를 통한 복잡한 상태 처리:**  
  PPO는 정책(액터)와 가치 함수(크리틱)를 동시에 학습하여, 게임판과 같이 여러 요소가 결합된 복잡한 상태 공간에서도 효과적인 의사결정이 가능함

## 4. CNN과의 시너지 및 구현의 용이성
- **CNN 기반 정책 네트워크 통합:**  
  PPO는 CNN 기반의 CnnPolicy와 자연스럽게 결합되어, 추출된 특징을 바탕으로 최적의 행동을 결정할 수 있음
  
- **구현의 용이성과 검증된 성능:**  
  PPO는 Stable-Baselines3와 같은 라이브러리에서 잘 구현되어 있어 구현 용이


## 결론

사과게임의 고득점 전략은 제한시간 내에 게임판의 2차원 배열에서 **유의미한 지역적 패턴**을 신속하게 인식하고, 이를 활용해 최적의 숫자 조합을 선택하여 최대한 많은 숫자를 제거하는 데 있습니다. 이를 달성하기 위해 본 프로젝트는 다음과 같은 전략적 선택을 하였습니다.

- **CNN 기반 데이터 인식:**  
  게임판은 10×17 크기의 배열에 1부터 9까지의 숫자가 배치된 구조를 가지므로, 지역적 패턴과 국소적인 특징을 효과적으로 추출할 수 있는 CNN이 최적의 선택입니다. CNN은 파라미터 효율성이 높아 복잡한 보드 상태를 효과적으로 모델링할 수 있습니다.

- **PPO 강화학습 알고리즘:**  
  사과게임은 한 번의 숫자 삭제로 전체 보드 상태가 급격히 변하는 동적 환경입니다. PPO는 최신 데이터를 기반으로 on-policy 학습을 수행하여 변화하는 상태 분포에 빠르게 대응하고, 클리핑 및 신뢰 영역 최적화를 통해 안정적인 정책 업데이트를 보장합니다. 또한, 온라인 데이터 활용과 CNN과의 자연스러운 시너지 효과, 그리고 구현의 용이성 등의 이유로 고득점 전략 학습에 가장 적합한 알고리즘으로 선정되었습니다.

이러한 선택들을 통해, 본 프로젝트는 제한시간 120초 내에 최대한 많은 숫자를 제거하여 120점 이상의 고득점을 달성할 수 있는 인공지능 개발을 목표로 하고 있습니다.
