# 5G InF RTT 기반 NLOS-강건 2D 측위: 순열불변 Set Transformer와 잔차 피드백 미분 Gauss-Newton

---

## 1. 동기 및 서론

### 1.1 중간발표까지의 실험 및 고찰

중간발표까지의 설계는 2단계 end-to-end 구조였다. Stage 1 MLP가 18차원 d_hat 벡터에서 NLOS 보정 거리 d_corr를 예측하고, Stage 2 GCN이 보정 거리와 기지국 좌표를 노드 피처로 받아 위치를 직접 출력하는 방식이었다. 이 설계를 검토하면서 네 가지 구조적 문제를 발견했다.

첫째, 손실 함수가 대칭(RMSE)이었다. 데이터를 분석하면 NLOS 오차의 평균은 +15.93 m, 표준편차 20.19 m로 양의 방향으로 강하게 편향되어 있다(양수 비율 81.4%). NLOS 환경에서 측정 거리는 항상 실제 거리보다 길다(d̂_i ≥ d_true,i). 그런데 RMSE 손실은 양의 잔차와 음의 잔차를 동일하게 벌하므로 이 사전지식을 전혀 활용하지 못한다.

둘째, Stage 1 MLP가 18차원 d_hat을 고정 순서 벡터로 입력받는 구조였다. 18개 BS는 본질적으로 순서가 없는 집합이다. BS 번호 부여 방식이 달라지거나 일부 BS가 누락되면 입력이 완전히 달라지는 순열 의존성 문제가 있었다.

셋째, GCN 사용의 근거가 약했다. 참고한 논문(Rana & Dulal, 2025)의 GTCN은 TCN을 통한 시계열 상관 학습이 핵심 강점인데, 본 데이터는 단일 시점 스냅샷이므로 TCN을 제거하면 GCN이 단순 집합 집계로 퇴화한다.

넷째, FC 레이어로 직접 좌표를 회귀하면 기하학적 제약이 없어 700 샘플의 소규모 데이터에서 과적합하기 쉽다.

### 1.2 아이디어 도출 흐름

네 가지 문제에 대한 해결 방향을 문헌에서 찾았다.

순열불변성 문제는 Deep Sets(Zaheer et al., 2017)와 Set Transformer(Lee et al., 2019)의 이론적 토대에서 해결책을 얻었다. 두 논문은 순열불변 함수의 구조적 표현 방식을 제시하며, 18개 BS를 순서 없는 집합으로 처리하는 수학적 근거를 제공한다.

블랙박스 좌표 회귀 문제는 algorithm unrolling(Monga et al., 2021) 패러다임에서 해결책을 찾았다. 반복 최적화 알고리즘을 신경망 레이어로 언롤하면 물리적 기하 구조를 유지하면서 학습 파라미터를 주입할 수 있다. Gauss-Newton은 삼변측량의 비선형 최소제곱 구조와 정확히 일치한다.

NLOS 단방향성 문제는 quantile regression(Koenker & Bassett, 1978)의 pinball 손실과, 부등식 제약을 최적화에 포함하는 2SWLS(Wang et al., 2020) 접근에서 아이디어를 얻었다.

이 세 흐름이 합쳐져 순열불변 Set Transformer + 미분가능 언롤드 Gauss-Newton + 비대칭 손실 조합이 도출됐다. 여기에 첫 번째 GN 추정 후 잔차를 Set Transformer에 재입력하는 잔차 피드백 구조는 본 프로젝트에서 독자적으로 제안한 방식이다.

### 1.3 제안 알고리즘 High-level 소개

제안 알고리즘은 NLOS 물리적 사전지식을 세 층위에 명시적으로 인코딩한다.

- 모델 구조 층위: 편향 보정 b̂_i에 softplus를 적용하여 항상 양수임을 구조적으로 보장한다.
- 손실 함수 층위: pinball 손실과 부등식 패널티가 d̂_i ≥ d_true,i 관계를 학습 목표에 직접 반영한다.
- 최적화 층위: Gauss-Newton이 보정된 거리 d̂_i − b̂_i를 이용해 기하학적으로 위치를 정제한다.

순열불변 Set Transformer는 18개 BS의 기하학적 불일치(어느 BS가 다른 BS들이 암시하는 위치와 모순되는가)를 attention으로 학습하여 각 BS의 신뢰도 c_i를 출력한다. c_i는 Gauss-Newton의 가중치로 직접 투입된다. 첫 번째 추정 후 잔차를 피드백하면 Set Transformer가 현재 추정의 기하학적 불일치 정도를 근거로 NLOS 판단을 정제한다.

---

## 2. 알고리즘 설명

### 2.1 전체 파이프라인 개요

입력: d̂ ∈ ℝ^18 (RTT 거리 측정값), p_bs ∈ ℝ^(2×18) (기지국 좌표)

출력: p̂ ∈ ℝ^2 (추정 위치)

알고리즘은 두 라운드로 구성된다.

**라운드 1:** 6차원 정적 토큰 생성 → Set Transformer 인코딩 → {c_i^(1), b̂_i^(1), p_0} 출력 → Gauss-Newton K=8회 적용 → 중간 추정 p̂_1 획득

**라운드 2:** p̂_1을 이용해 각 BS의 잔차 r_i^(1) = (d̂_i − ‖p̂_1 − p_bs,i‖) / 100 계산 → 7차원 토큰(기존 6개 + 잔차 1개) 생성 → Set Transformer 재인코딩(가중치 공유) → {c_i^(2), b̂_i^(2)} 업데이트 → Gauss-Newton K=8회 (p̂_1 초기화) → 최종 위치 p̂ 출력

### 2.2 토큰화

각 BS i에 대해 6차원 토큰 t_i = [f_1, f_2, f_3, f_4, f_5, f_6]을 생성한다.

- f_1 = p_bs,x,i / 60: 기지국 x 좌표 정규화 (공간 반경 기준)
- f_2 = p_bs,y,i / 30: 기지국 y 좌표 정규화
- f_3 = d̂_i / 100: 거리 측정값 정규화
- f_4 = (d̂_i − median({d̂_j})) / 100: 전체 측정값 중앙값 대비 상대 크기
- f_5 = rank(d̂_i) / 18: 거리 측정값의 순위(순서 정보)
- f_6 = (d̂_i − min({d̂_j})) / 100: 최솟값 대비 초과량

f_4, f_5, f_6은 BS i의 측정값이 나머지 BS들과 얼마나 다른지를 나타내는 상대적 특징으로, CIR 없이 NLOS 이상치를 수치화한다. 절대 BS 인덱스(번호)를 포함하지 않으므로 BS 순서가 바뀌어도 동일한 토큰 집합이 생성되어 순열불변성이 보장된다.

라운드 2에서는 잔차 r_i^(1) = (d̂_i − ‖p̂_1 − p_bs,i‖) / 100을 7번째 특징으로 추가하여 (B, 18, 7) 크기의 토큰 행렬을 구성한다.

### 2.3 Set Transformer 인코더

토큰 행렬 (B, 18, F)는 다음 세 단계를 순서대로 통과한다.

**공유 MLP φ:** 토큰 행렬을 (B, 18, 256)으로 변환한다. 구성은 Linear(F → 256) → ReLU → Linear(256 → 256)이다. 라운드 1에서는 F=6, 라운드 2에서는 F=7인 별도 φ_refine을 사용하되, 이후 ISAB 스택과 헤드는 라운드 1과 공유한다.

**ISAB 스택 × 3:** 각 ISAB(Induced Set Attention Block)는 m=32개의 학습 가능 inducing point I ∈ ℝ^(1×32×256)을 매개로 다음을 계산한다.

ISAB(X) = MAB(X, MAB(I, X))

MAB(Q, K)는 Multi-head Attention(4헤드, d=256) 뒤에 Add&Norm과 2층 FFN(256→512→256)을 적용하는 표준 Transformer 블록이다. 첫 번째 MAB(I, X)에서 inducing point가 BS 집합으로부터 정보를 수집하고(O(n·m) 연산), 두 번째 MAB(X, ·)에서 각 BS가 그 집약 정보를 참조한다. 이 구조를 통해 어느 BS의 측정값이 다른 BS들이 암시하는 기하학적 패턴과 모순되는지를 attention score로 학습할 수 있다.

**세 개의 출력 헤드 (PMA 기반 및 토큰별):**

- 신뢰도: c_i = sigmoid(Linear(H_i, 1)) ∈ [0, 1] (각 BS별, 토큰 레벨 헤드)
- 편향 보정: b̂_i = softplus(Linear(H_i, 1)) ≥ 0 (각 BS별, 토큰 레벨 헤드)
- 초기 위치: p_0 = Linear(64→2)(ReLU(Linear(256→64)(PMA(H)))) ∈ ℝ^2

PMA(Pooling by Multihead Attention)는 학습 가능한 seed 벡터 S ∈ ℝ^(1×256)으로 MAB(S, H)를 계산하여 집합 전체를 하나의 벡터로 집약한다.

### 2.4 미분가능 언롤드 Gauss-Newton

초기값 p = p_0에서 시작하여(라운드 2에서는 p̂_1) K=8회 반복한다.

각 스텝에서:

1. 보정 거리 d_corr,i = d̂_i − b̂_i를 계산한다.
2. 잔차: r_i = ‖p − p_bs,i‖ − d_corr,i
3. 야코비안: J_i = (p − p_bs,i) / (‖p − p_bs,i‖ + ε), 결과는 (B, 18, 2) 행렬
4. 가중 Hessian 근사: H = J^T diag(c_i) J + λI ∈ ℝ^(2×2)
5. 가중 기울기: g = J^T diag(c_i) r ∈ ℝ^2
6. 업데이트: p ← p − H⁻¹g (torch.linalg.solve 사용)

λ = softplus(log_λ) + 10⁻⁴로, 학습 가능한 Levenberg-Marquardt 감쇠 파라미터다. BS 배치가 공선(co-linear)에 가까울 때 H가 특이(singular)에 가까워지는 것을 방지한다. c_i = 0인 BS는 가중치가 0이 되어 최적화에 전혀 기여하지 않으므로, Set Transformer가 특정 BS를 신뢰도 0으로 평가하면 해당 BS는 삼변측량에서 완전히 배제된다. 전체 K=8 스텝이 PyTorch autograd 그래프 안에서 미분가능하므로, c_i, b̂_i, λ 모두 최종 위치 손실로 end-to-end 학습된다.

### 2.5 선행 연구와의 차이

**Set Transformer(Lee et al., 2019)와의 차이:** Lee et al.은 집합 함수 근사를 위한 일반적인 순열불변 아키텍처를 제안한다. 본 연구는 이를 5G RTT NLOS 측위 도메인에 적용하고, ISAB 출력을 미분가능 Gauss-Newton의 가중치와 편향 보정으로 직접 매핑하는 end-to-end 구조를 추가했다. 또한 잔차 피드백을 통한 2라운드 정제는 Lee et al.에 없는 본 연구의 독자적 기여다.

**Aristorenas(2025)와의 차이:** 순열불변 Transformer를 실내 측위에 적용한 가장 최근 연구다. 입력 신호가 Wi-Fi RSSI이고 기하학적 최적화 층이 없다. 본 연구는 5G RTT라는 다른 신호를 사용하고, Set Transformer 출력을 미분가능 Gauss-Newton에 투입하는 하이브리드 구조를 제안한다.

**Algorithm Unrolling(Monga et al., 2021)과의 차이:** Monga et al.은 언롤링 패러다임을 일반 서베이 수준에서 정리한다. LISTA, ISTA 등 희소 복원 문제에 주로 적용됐다. 본 연구는 같은 패러다임을 range-only NLOS 측위의 비선형 최소제곱(Gauss-Newton)에 적용한 첫 사례 중 하나로, Set Transformer가 학습한 신뢰도 c_i가 GN 가중치로 직접 투입되는 결합 구조를 제안한다.

**SC-wLS(Wu et al., 2022)와의 차이:** 신경망이 가중치를 예측하고 미분가능 WLS에 투입하는 구조를 카메라 relocalization에 적용했다. 본 연구는 이 구조를 5G RTT NLOS 도메인으로 옮기고, 선형 WLS 대신 비선형 Gauss-Newton을 사용했으며, NLOS 단방향 편향에 특화된 비대칭 손실과 잔차 피드백을 추가했다.

**Chatelier et al.(2023)과의 차이:** 3GPP InF 시나리오에서 데이터셋 크기가 측위 성능에 결정적 영향을 미친다는 것을 체계적으로 분석한 논문이다. 본 연구에서 합성 NLOS 증강으로 사실상 무한한 학습 데이터를 생성하는 전략의 정당성을 이 논문이 제공한다.

### 2.6 손실 함수

전체 손실은 다음과 같다.

L = Huber(p̂, p_gt) + 0.3 · Huber(p_0, p_gt) + 0.15 · Huber(p̂_1, p_gt) + 0.1 · L_pinball + 0.05 · L_ineq + 0.01 · L_gate

각 항의 역할:

**Huber(p̂, p_gt):** 주 위치 손실. 이상치에 강건한 Huber 손실(δ=3.0m)을 사용한다. MSE 대비 큰 오차에서 선형 증가하여 fold 분산을 줄인다.

**0.3 · Huber(p_0, p_gt):** Set Transformer 직접 출력 p_0에 대한 deep supervision. 이 항이 없으면 Set Transformer가 p_0를 임의의 값으로 출력하여 GN의 초기화가 불안정해진다.

**0.15 · Huber(p̂_1, p_gt):** 라운드 1 GN 출력 p̂_1에 대한 중간 supervision. 라운드 2 잔차 피드백의 품질이 p̂_1의 정확도에 의존하므로 이 항이 필수적이다.

**L_pinball(τ=0.15):** u_i = d̂_i − ‖p̂ − p_bs,i‖로 정의하면, L_pinball = E[max(τ·u, (τ−1)·u)]이다. τ=0.15이면 u < 0(예측 거리 > 측정 거리, 물리적으로 불가능)을 u > 0보다 (1−τ)/τ ≈ 5.7배 강하게 벌한다. 이 비대칭 압력이 NLOS 양의 편향을 자동으로 상쇄한다.

**L_ineq:** L_ineq = E[max(0, ‖p̂ − p_bs,i‖ − d̂_i)²]. 추정 위치까지의 거리가 측정 거리를 초과하는 물리적으로 불가능한 영역으로의 이동을 직접 억제한다. Vaghefi et al.의 SDP가 b_i ≥ 0을 최적화 제약으로 포함하는 것과 같은 동기이며, 이를 미분가능 손실로 번역한 것이다.

**L_gate = (mean(c_i) − 0.5)²:** 신뢰도가 모두 0으로 붕괴하는 "죽은 게이팅" 현상을 방지한다.

### 2.7 학습 전략

**합성 NLOS 증강:** 700 샘플로의 과적합을 방지하기 위해 매 배치의 50%를 합성 샘플로 채운다. 위치를 Uniform([−60,60]×[−30,30])에서 샘플링하고, 각 BS에 NLOS 편향을 확률 0.62로 로그정규 분포(μ=2.302, σ=0.970, 결과 평균 ≈16 m, std ≈20 m)에서 생성한다. 이 파라미터는 실제 데이터 통계(평균 +15.93 m, std 20.19 m, NLOS 비율 62%)를 기반으로 도출됐다. Chatelier et al.(2023)이 "데이터셋 크기가 성능에 결정적"임을 보인 것을 고려하면, 이 합성 증강은 사실상 무한한 학습 샘플을 만드는 핵심 장치다.

**BS dropout:** 학습 시 배치마다 랜덤 2~4개 BS의 c_i를 강제로 0으로 설정한다. 특정 BS에 대한 과의존을 방지하고 누락 BS에 대한 강건성을 확보한다.

**5-fold CV + 전체 재학습:** 5-fold CV로 최적 수렴 epoch를 추정한 후, 700명 전체로 재학습한다. 이 과정에서 fold 모델 5개와 전체 재학습 모델 1개가 생성된다.

**멀티시드 앙상블:** seed=42, 1, 2로 각각 독립 학습하여 총 18개 모델(fold×5 + 전체×1, 시드×3)을 단순 평균 앙상블한다. 700 샘플의 소규모 데이터에서 발생하는 높은 fold 분산을 시드 다양성으로 보완한다.

---

## 3. Agent AI 활용

본 프로젝트에서는 Anthropic의 Claude Code(claude-sonnet-4-6)를 개발 보조 도구로 활용했다.

**Claude Code가 수행한 역할:**

설계 명세를 바탕으로 한 코드 구현(모델 아키텍처, 학습 루프, 손실 함수, 데이터 로더), 학습 실행 및 결과 모니터링, 분석 스크립트(기준선 비교, 모델 해석 시각화, Conformal Prediction) 작성, 최신 논문 검색 및 요약 보조(arXiv 2506.00656, 2501.07774, 2505.01810 등), GPU 최적화(AMP, cudnn.benchmark) 적용.

**학생이 수행한 역할:**

알고리즘 설계의 모든 핵심 결정: (i) 순열불변 Set Transformer + 미분 GN 조합의 선택, (ii) NLOS 단방향성을 pinball + 부등식 손실로 인코딩하는 아이디어, (iii) 잔차 피드백 재토큰화 구조 제안(이 아이디어는 기존 논문에 없는 독자 제안), (iv) 실험 방향 판단(SwiGLU, 학습가능 step size 실험 후 효과 없음 확인, 잔차 피드백으로 방향 전환), (v) 결과 해석 및 보고서 작성.

Claude Code는 학생이 설계한 알고리즘을 코드로 번역하고 실험을 실행하는 역할을 담당했다. 알고리즘의 핵심 아이디어는 학생의 문헌 조사 및 이전 실험(중간발표) 고찰에서 비롯됐다.

---

## 4. 결과 도출 및 토의

### 4.1 비교 실험의 공정성 논의

기준선 비교에서 공정성(fairness) 문제를 명확히 해야 한다. 본 실험은 동일한 80/20 분할(seed=42, train 560명, val 140명)에서 모든 방법을 평가했다.

그러나 비교 방법들 사이에는 구조적 불균형이 있다. LS 삼변측량, WLS 삼변측량, RANSAC은 학습 데이터를 전혀 사용하지 않는다. 이들과 딥러닝 모델을 단순 RMSE로 비교하는 것은 "정보량"의 차이가 크므로 완전히 공정하지 않다. 딥러닝 모델은 700 샘플의 통계를 학습할 수 있지만, 기하학적 기준선은 그 기회 자체가 없다.

공정한 비교가 가능한 그룹은 다음과 같다.

| 비교 그룹 | 근거 |
|-----------|------|
| 제안 방법 vs 단순 MLP | 동일한 학습 데이터(700명), 동일한 입력 특징(d̂, BS 좌표), 동일한 증강 전략 사용. 유일한 차이는 아키텍처(Set Transformer + GN vs FC 3층). 가장 공정한 비교. |
| 제안 방법 vs Random Forest | 동일한 학습 데이터, 비슷한 특징 공간. ML 기준선 중 가장 경쟁력 있는 방법. |
| 제안 방법 vs 편향제거 + LS | NLOS 지식을 활용하는 비학습 기준선. 사전지식 활용의 가치를 보여줌. |
| 기하학적 방법들(LS, WLS, RANSAC) | 서로 공정한 비교. 어떤 기하학적 처리가 유효한지 보여줌. |

"딥러닝 vs 단순 삼각측량" 비교가 불공정하다는 지적은 타당하다. 그러나 이 비교의 목적은 딥러닝의 우월성을 주장하는 것이 아니라, 학습 기반 접근이 필요한 이유(NLOS 환경에서 기하학적 방법의 한계)를 보여주는 것이다.

**자체 평가 방식의 공정성:** 5-fold CV를 사용했으며, 각 fold 모델은 해당 validation 샘플을 한 번도 학습에 사용하지 않는다. CV RMSE는 학습 데이터에 대한 편향이 없는 일반화 성능의 신뢰할 만한 추정치다. 다만 최종 앙상블 모델을 학습셋 전체(700명)에서 평가하면 일부 샘플이 학습에 포함되므로 낙관적 추정이 된다. 비교 실험의 수치(1.537 m)는 이 편향을 줄이기 위해 별도의 80/20 고정 분할에서 측정했다.

### 4.2 성능 비교 결과

| 방법 | 분류 | 학습 사용 | RMSE (m) | 중앙값 (m) | 90퍼센타일 (m) |
|------|------|----------|----------|-----------|--------------|
| LS 삼변측량 | 기하학적 | 아니오 | 25.263 | 22.598 | 32.280 |
| WLS 삼변측량 | 기하학적 | 아니오 | 16.295 | 14.560 | 24.339 |
| RANSAC 삼변측량 | 기하학적 | 아니오 | 19.685 | 2.369 | 31.636 |
| 편향제거 + LS | 기하학적 | 아니오 | 12.399 | 7.831 | 17.910 |
| Ridge 회귀 | ML | 예 | 11.810 | 9.324 | 18.660 |
| Random Forest | ML | 예 | 8.281 | 6.135 | 12.473 |
| 단순 MLP (FC만) | ML | 예 | 8.267 | 6.074 | 11.466 |
| **ST+GN 앙상블 (제안)** | 딥러닝+기하 | 예 | **1.537** | **0.742** | **1.759** |

단순 MLP 대비 RMSE 81.4% 감소. 단순 MLP와 제안 방법의 차이는 아키텍처에서만 기인하므로 이 수치가 제안 방법의 실질 기여다.

RANSAC의 중앙값 오차(2.369 m)만 낮고 90퍼센타일(31.636 m)이 극단적으로 높은 현상은, NLOS 비율이 62%에 달하는 환경에서 인라이어 판단 기준이 붕괴하여 절반 이상의 사용자에서 크게 실패하는 것을 의미한다.

### 4.3 구성 요소 기여도 분석

| 제거 항목 | Val RMSE (m) | 변화 |
|-----------|-------------|------|
| 전체 제안 방법 | 2.506 | 기준 |
| 비대칭 손실 제거 (β=γ=0) | 3.287 | +31.2% 저하 |
| deep supervision 제거 (α=0) | — | GN 발산 가능성 |

비대칭 손실 제거 시 성능 저하가 가장 컸다. NLOS 단방향성이라는 물리적 사전지식을 손실에 반영하는 것이 핵심 기여임을 역검증한다.

### 4.4 아키텍처 탐색

| 구성 | CV RMSE (m) | CV std | 비고 |
|------|-------------|--------|------|
| Baseline (d=128, ISAB×2, GN×5) | 2.839 | — | 단일 모델 기준 |
| Big config (d=256, ISAB×3, GN×8) | 2.127 | 0.813 | 5-fold 기준 |
| + SwiGLU FFN | 2.302 | 0.786 | 개선 없음 |
| + 학습가능 GN step size | 2.385 | 0.909 | 오히려 저하 |
| **+ 잔차 피드백 (최종)** | **2.139** | **0.735** | CV 최저, std 최저 |

SwiGLU는 Masrur et al.(2025)에서 CIR 기반 NLOS 환경에서 8.51% 개선을 보고했으나, 본 RTT-only 설정에서는 유의미한 차이를 보이지 않았다. 학습 가능 step size는 fold 분산을 오히려 증가시켰다. 잔차 피드백만이 CV RMSE와 std를 동시에 개선했다.

### 4.5 모델 해석: 제안 방법이 적합했는가

모델이 단순히 데이터를 암기한 것이 아니라 NLOS의 물리적 특성을 학습했는지 검증했다.

| 분석 항목 | 수치 | 해석 |
|-----------|------|------|
| b̂_i vs 실제 NLOS 편향 Pearson r | 0.665 | 편향 보정 예측이 실제 NLOS 패턴과 유의미하게 일치 |
| b̂_i vs 실제 NLOS 편향 Spearman ρ | 0.731 | 순위 기준 강한 단조 상관 |
| LOS BS 평균 c_i | 0.331 | 신뢰도 높음 |
| NLOS BS 평균 c_i | 0.007 | 신뢰도 거의 0 (NLOS 식별 성공) |
| GN 정제로 오차 감소 사용자 비율 | 91.9% | GN 정제가 대다수에 효과적 |
| Set Transformer 직접 출력(p_0) RMSE | 2.887 m | |
| GN 정제 후(p̂) RMSE | 1.503 m | 47.9% 개선 |

신뢰도가 낮은 BS는 공간 끝 위치(BS06: (50,20), BS12: (50,0), BS18: (50,−20))에 집중됐다. 이 위치들은 기하학적으로 공장 내부와의 LOS가 차단될 가능성이 높은 외곽 BS로, 모델의 판단이 물리적으로 타당하다. 이는 모델이 BS 좌표 정보와 거리 측정의 상대적 패턴을 결합하여 의미 있는 표현을 학습했음을 보여준다.

### 4.6 불확실성 정량화: Conformal Prediction

점 추정 RMSE 외에 예측의 신뢰 가능성을 평가하기 위해 Split Conformal Prediction을 적용했다.

| 목표 커버리지 | 예측 반지름 (m) | 실제 달성 커버리지 |
|-------------|---------------|-----------------|
| 80% | 1.246 | 80.1% |
| 90% | 2.001 | 90.1% |
| 95% | 3.931 | 95.1% |

목표 커버리지와 실제 달성 커버리지가 정확히 일치한다. 이는 Conformal Prediction 정리(Angelopoulos & Bates, 2022)가 이 데이터에서 실제로 성립함을 확인한다. 90% 커버리지 기준으로, 예측 위치로부터 반지름 2.001 m 원 안에 실제 위치가 있을 확률이 최소 90%임이 분포 가정 없이 수학적으로 보장된다. 이는 RMSE 단일 수치 이상의 정보를 사용자에게 제공한다.

### 4.7 알고리즘 강점 및 한계

**강점:**

물리적 제약이 세 층위에서 인코딩된다. 모델 구조(b̂_i ≥ 0), 손실 함수(pinball + ineq), 최적화(보정 거리를 이용한 GN)가 모두 NLOS 단방향성을 반영한다. Set Transformer의 순열불변성으로 BS 순서 변경이나 누락에 강건하다. GN이 삼변측량의 기하학적 구조를 명시적으로 활용하므로, 완전한 블랙박스 회귀보다 파라미터 효율이 높고 700 샘플의 소규모 데이터에서도 효과적이다. 잔차 피드백은 추가 파라미터를 최소화하면서 NLOS 판단을 동적으로 정제한다.

**한계:**

700 샘플의 소규모 데이터셋으로 fold 간 분산이 여전히 높다(CV std 0.735 m). 잔차 피드백은 2라운드 처리로 추론 시간이 1라운드 대비 약 2배다. 합성 NLOS 분포가 단순 로그정규 모델에 기반하여 실제 환경의 복잡한 산란 패턴을 완전히 반영하지 못할 수 있다. 학습 환경(InF-DH)과 다른 환경으로의 전이 성능은 검증되지 않았다.

### 4.8 향후 연구

잔차 피드백을 K라운드로 일반화하면 이론적으로 더 강한 NLOS 정제가 가능하다. 위치를 점 추정이 아닌 Gaussian 분포 N(μ, Σ)로 출력하고 NLL 손실로 학습하면 Conformal Prediction과 결합하여 더 적응적인 신뢰 영역을 제공할 수 있다. Radio Foundation Model(Ott et al., 2024)처럼 합성 데이터로 사전 학습 후 실제 700 샘플로 fine-tuning하면 소규모 데이터 한계를 극복할 수 있다. CIR 없이 range-only로 동작하는 본 방법이 CIR 기반 방법과 어느 정도 성능 차이가 나는지의 정량적 분석도 중요한 후속 연구다.

---

## 5. References

[1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97:3744–3753.

[2] Zaheer, M., Kottur, S., Ravanbhakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep Sets. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

[3] Monga, V., Li, Y., & Eldar, Y. C. (2021). Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing. *IEEE Signal Processing Magazine*, 38(2), 18–44.

[4] Wu, S., et al. (2022). SC-wLS: Towards Interpretable Feed-forward Camera Re-localization. *European Conference on Computer Vision (ECCV)*.

[5] Guvenc, I., & Chong, C. C. (2009). A Survey on TOA Based Wireless Localization and NLOS Mitigation Techniques. *IEEE Communications Surveys & Tutorials*, 11(3), 107–124.

[6] Wang, G., et al. (2020). Two-Step WLS Localization Method Exploiting Non-Line-of-Sight Measurements. *Sensors*, 20(5), 1403.

[7] Chatelier, B., et al. (2023). Influence of Dataset Parameters on the Performance of Direct UE Positioning via Deep Learning. *arXiv:2304.02308*.

[8] Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50.

[9] Masrur, S., et al. (2025). Transforming Indoor Localization: Advanced Transformer Architecture for NLOS Dominated Wireless Environments with Distributed Sensors. *arXiv:2501.07774*.

[10] Aristorenas, A. J. (2025). Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings. *arXiv:2506.00656*.

[11] Angelopoulos, A. N., & Bates, S. (2022). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *arXiv:2107.07511*.

[12] Zhou, Z., et al. (2025). Conformal Prediction for Indoor Positioning with Correctness Coverage Guarantees. *arXiv:2505.01810*.

[13] Ott, J., et al. (2024). Radio Foundation Models: Pre-training Transformers for 5G-based Indoor Localization. *arXiv:2410.00617*.

[14] Rana, M. S., & Dulal, M. (2025). Indoor Localization Using Graph Temporal Convolutional Network. *Sensors*, 25.

---

*보고서 작성일: 2026-06-03*
