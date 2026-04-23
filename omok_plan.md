# 오목 강화학습 프로젝트 계획

## 0. 최신 운영 상태

- 기준 시점: `2026-04-16`
- 현재 주력 라인: `tactical_rule_value_agent`
- 최근 완료 버전: `v38`
- 현재 진행 배치: `v39`
- 현재 실험 설정:
  - `reference_rule_agent_level=super_easy`
  - `reference_rule_opening_moves=20`
  - `reference_rule_followup_probability=0.10`
  - `reference_rule_only_agent_level=super_easy`
  - `teacher_rule_agent_level=hard`
  - `teacher_weight=1.0`
  - candidate 첫 수는 중앙 `7x7` 내부 랜덤

## 1. 현재 방향

현재 운영선은 `tactical_rule_value_agent`를 기준으로, refer + overlay + hard teacher 조합을 안정화하는 것이다. 여기에 rule-only reference를 추가해서 외곽 오프닝이 통하지 않는 상대를 함께 둔다.
overlay는 초반 `20수` 동안 강하게 걸고, 이후에도 작은 확률로 rule 판단을 다시 따르도록 해서 reference 쪽 의사결정이 rule base에 조금 더 끌리게 만든다.

## 2. 현재 상태 및 refer 풀

- tactical rule refer: `v6~v38`
- tactical value refer: `v65`, `v69`, `v76`, `v86`, `v88`, `v91`, `v94`, `v95`, `v96`, `v100~v114`
- value refer: `v198`, `v199`, `v201`, `v202`, `v203`
- rule-only reference: `super_easy`를 추가로 넣는 방식이 유효한지 실험 중
- `v39`의 첫 시도는 첫 수 외곽 전략 문제로 중단했고, 같은 버전 번호로 중앙 `7x7` 제한을 넣어 재시작했다

## 3. rule 커리큘럼 상태

### 난이도 체계

- `super_easy`
- `easy`
- `normal`
- `hard`

### 실험 결과

- `v1`: hard rule 상대, 전패
- `v2`: hard rule 상대, 최신 tactical seed 사용, 전패
- `v3`: easy rule 상대, 전패
- `v4`: easy rule 상대 + teacher loss, 전패
- 이후 `v6`부터는 refer + teacher 구조로 전환
- `v39` 첫 시도에서는 외곽 첫 수가 승률을 유지하는 현상이 보여 중단
- 현재 `v39` 재시작 배치는 중앙 `7x7` 첫 수 제한과 rule-only reference를 함께 검증하는 상태
- `v40`은 그 다음 비교군으로 두면 된다
- `v99` 계열 무한루프는 오염된 체인으로 확인되어 종료

결론:
- 현재는 `super_easy` overlay가 더 적절
- `easy` overlay는 아직 과한 것으로 판단
- 한 번에 여러 조건을 바꾸지 말고, 다음엔 한 변수만 바꿔야 함
- 외곽 오프닝이 잘 먹히는 상대를 막기 위해 첫 수 제한과 rule-only reference를 같이 써야 함

## 4. 현재 코드 상태

### `train_value_reference.py`

- 현재 주력 스크립트
- reference ensemble에 rule overlay를 적용할 수 있다
- `reference_rule_only_agent_level`로 pure rule-based reference를 추가할 수 있다
- hard teacher 항상 적용
- `super_easy`일 때 candidate 흑 고정
- 흑/백 분리 승률 기록 및 winrate 로그 생성
- `save-every=1000` 단발 배치는 중간 체크포인트가 없다

### `train_tactical_value.py`

- 현재는 주력보다 보조 실험용

### `agent/torch_cnn_value_agent.py`

- 첫 수는 중앙 `7x7` 내부에서만 랜덤
- 15수 전 랜덤 금지
- 짧은 패배 3배 보정
- teacher 보조 loss 지원

## 5. 다음 추천 실험

### 추천 1

- seed: `models/tactical_rule_value_agent_v38.pt`
- candidate prefix: `tactical_rule_value_agent`
- refer 범위: 현재 남아 있는 전체 refer
- value refer overlay: `super_easy`
- rule-only reference: `super_easy`
- teacher: `hard`
- `teacher_weight`: `1.0`

### 추천 2

- `v39` 종료 후
- 첫 수 제한만 유지한 채 rule-only reference 유무를 비교
- 또는 refer 범위와 overlay 난이도 중 하나만 바꿔 비교

## 6. 보류 사항

- 렌주룰
- 흑 33/44 금지
- policy/hybrid 운영 학습 연결

이 셋은 아직 보류다. 지금은 refer + overlay + teacher 구조 안정화가 우선이다.

## 7. 추가 메모

- refer 관리는 [`models/refer/`](/home/yphong/omok_deeplearning/models/refer) 기준이다.
- `tactical_rule_value_agent v10~v38`는 refer로 누적 승격되었다.
- `super_easy` overlay는 모든 refer에 적용하는 방향으로 운영한다.
- `hard` teacher만 `open three`를 강제 차단한다.
- `forever` launcher는 공용 락 파일을 사용한다.
- 현재 학습 장치는 WSL 기준 CPU다.

## 8. 히스토리 로그

- `v1~v4`는 하드/이지 rule 직결 학습이 실패했다.
- `v6` 이후 refer 기반 커리큘럼으로 바뀌었다.
- `v39` 첫 시도는 중앙 첫 수 제한 전 마지막 실험으로 남아 있다.
- 현재 진행 중인 `v39` 재시작 배치부터는 첫 수 중앙 `7x7`과 rule-only reference가 기본 전제다.
- `v41`~`v45`는 동일한 `super_easy` / `hard` 설정으로 연속 배치를 돌리며 안정 구간을 확인했다.
- `v45` 이후에는 forever 반복 모드로 넘어가서, 배치 수를 더 쌓는 방향으로 운영했다.
- 승률이 62% 근처에서 정체처럼 보였지만, 배치 몇 번만으로 단정하지 않고 추가 배치를 돌리기로 했다.
