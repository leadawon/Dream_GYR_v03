# 분석: 256 Step은 정답(1.0), 128 Step은 오답(0.0)인 경우

## 전체 통계
- **총 케이스 수**: 29개 (전체 401개 문제 중 약 7.2%)

## 주요 패턴 및 인사이트

### 1. **계산 실수 및 논리적 오류**

#### Case 1: Doc ID 14 (Dance class percentage)
- **Target**: 60
- **256 filtered**: 60 ✓
- **128 filtered**: 80 ✗
- **문제점**: 128 step에서 마지막 계산 부분에서 "12 students - 4 students = 8 students" 라는 잘못된 계산을 함. 실제로는 12명이 hip-hop에 등록했어야 하는데, 엉뚱한 계산을 추가로 수행.

#### Case 2: Doc ID 15 (Jewelry vs Electronics profit)
- **Target**: 125
- **256 filtered**: $125 ✓
- **128 filtered**: $25 ✗
- **문제점**: 128 step의 마지막 답변에서 "$\boxed{\$25}$"로 잘못 출력. 실제 계산은 $125가 맞게 나왔지만, 최종 답변을 쓸 때 오타 또는 잘못된 참조로 $25를 씀.

#### Case 3: Doc ID 24 (Book discount calculation)
- **Target**: 26
- **256 filtered**: 26 ✓
- **128 filtered**: 19.6 ✗
- **문제점**: 128 step에서 "Kyle Kyle paid 5/4 of the original price"라고 잘못 이해함. 25% 할인이면 75% (3/4)를 지불해야 하는데, 5/4 (125%)로 잘못 계산함.

### 2. **텍스트 생성 품질 문제 (Typos, Repetition, Gibberish)**

#### Case 4: Doc ID 33 (Gretchen's coins)
- **Target**: 70
- **256 filtered**: 70 ✓
- **128 filtered**: 100 ✗
- **문제점**: 128 step response: "Let's be careful. Gretchen has 110 coins in total... So,chen has 100 coins coins..." - 문장이 깨지고 논리가 불완전함.

#### Case 8: Doc ID 60 (Oranges)
- **Target**: 17
- **256 filtered**: 17 ✓
- **128 filtered**: 1717 ✗
- **문제점**: 128 step에서 답을 추출할 때 "17"이 아닌 "1717"로 중복 추출됨. 텍스트 생성 과정에서 숫자가 반복되었을 가능성.

### 3. **중간 단계에서 논리 비약**

#### Case 5: Doc ID 37 (John's lego sets)
- **Target**: 2
- **256 filtered**: 2. ✓
- **128 filtered**: 11. ✗
- **문제점**: 128 step에서 "He has $5 left, so he spent $195 - $5 = $190" - 잘못된 추론. 실제로는 $35가 남아야 함.

#### Case 10: Doc ID 70 (Judy's teaching income)
- **Target**: 7425
- **256 filtered**: 7425 ✓
- **128 filtered**: 5625 ✗
- **문제점**: 128 step에서 토요일 8개 클래스를 누락하고 weekday 25개 클래스만 계산함.

### 4. **추출(Extraction) 문제**

#### Case 2: Doc ID 15
- 계산은 맞았지만 최종 답변 형식에서 $125 대신 $25로 잘못 추출

#### Case 6: Doc ID 38 (Running speed)
- **Target**: 10
- **256 filtered**: 10 ✓
- **128 filtered**: 3.33. ✗
- **문제점**: 128 step에서 중간 계산 값(10/3)을 최종 답으로 추출함.

## 핵심 인사이트

### 💡 **1. 추론 길이와 품질의 Trade-off**
- **128 step**: 더 짧은 생성으로 인해:
  - 중간 계산 단계 생략 가능성 ↑
  - 논리적 검증 부족
  - 급하게 결론 도출 → 오류 발생

- **256 step**: 더 긴 생성으로:
  - 단계별 계산을 더 명확하게 서술
  - 자체 검증 기회 증가
  - 최종 답변 전 재확인 가능

### 💡 **2. 텍스트 생성 안정성**
- 128 step에서 더 많은 typo와 repetition 발생:
  - "Kyle Kyle paid"
  - "coins coins"
  - "each4 * 2"
  - "classes classes"
  
- 이는 디코딩 과정에서 128 step이 덜 안정적일 수 있음을 시사

### 💡 **3. 수학적 개념 이해**
- 128 step에서 기본 개념 오류가 더 많음:
  - 25% 할인 → 5/4를 지불한다는 잘못된 이해 (Doc 24)
  - 백분율 계산 오류 (Doc 14)

### 💡 **4. Multi-step Reasoning 능력**
- 256 step이 복잡한 multi-step 문제에서 더 강력:
  - 여러 조건을 동시에 고려
  - 중간 결과를 정확히 추적
  - 최종 답변까지 일관성 유지

### 💡 **5. 답변 추출(Extraction) 정확도**
- flexible-extract 방식에서:
  - 256 step: 더 명확한 형식으로 최종 답변 제시
  - 128 step: 중간 계산값이나 잘못된 값을 추출하는 경우 발생

## 결론

**256 step이 128 step보다 우수한 이유:**

1. **충분한 생성 길이** → 완전한 추론 체인 구축 가능
2. **텍스트 생성 안정성** → typo와 오류 감소
3. **자체 검증 기회** → 답변 작성 후 재확인 가능
4. **명확한 답변 형식** → 추출 오류 감소

**128 step의 한계:**

1. 추론 단계 압축으로 인한 논리 비약
2. 텍스트 생성 품질 저하
3. 복잡한 문제에서 중간 단계 누락
4. 최종 답변 추출 시 오류 증가

## 권장사항

- **복잡한 수학 문제**: 256 step 이상 사용 권장
- **간단한 문제**: 128 step도 충분할 수 있음
- **품질 개선**: Step 수를 늘리는 것이 정확도 향상에 효과적
