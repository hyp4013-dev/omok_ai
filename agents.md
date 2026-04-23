# AGENTS.md

## 최신 운영 스냅샷

- 기준 시점: `2026-04-16`
- 현재 주력 라인: `tactical_rule_value_agent`
- 최근 완료 버전: `v40`
- 현재 진행 버전: `v41`
- 현재 운영 방식:
  - 단발 배치는 버전 번호를 1개만 올려서 1000게임 기준으로 진행한다
  - 장기 반복은 `forever` 런처를 별도으로 켠다
  - 실행 로그와 산출물은 버전별로 남긴다
- 자세한 학습 규칙과 실험 파라미터는 [`omok_plan.md`](/home/yphong/omok_deeplearning/omok_plan.md)와 [`PROJECT_HANDOFF.md`](/home/yphong/omok_deeplearning/PROJECT_HANDOFF.md)에 둔다

## 스크립트 레지스트리

### 학습 / 실행

- [`train_value_reference.py`](/home/yphong/omok_deeplearning/train_value_reference.py)
  - reference 세트와 candidate를 비교하는 메인 실행 스크립트
  - 후보 버전 생성, 로그 기록, 체크포인트 저장, reference 로딩을 담당한다
- [`train_tactical_value.py`](/home/yphong/omok_deeplearning/train_tactical_value.py)
  - tactical 성격의 보조 실험 스크립트
  - 특정 tactical 기준선을 확인하거나 행동 경향을 점검할 때 쓴다
- [`train_competitive.py`](/home/yphong/omok_deeplearning/train_competitive.py)
  - 두 에이전트 간 경쟁 실험을 돌리는 스크립트
  - 학습 자체보다 대결 결과와 균형 여부를 보는 데 가깝다
- [`play_random.py`](/home/yphong/omok_deeplearning/play_random.py)
  - 랜덤 대 랜덤 대국을 빠르게 돌리고 로그를 남기는 스크립트
  - 환경과 로그 포맷이 정상인지 확인할 때 쓴다
- [`play_human_vs_rule.py`](/home/yphong/omok_deeplearning/play_human_vs_rule.py)
  - 사람이 터미널에서 직접 두는 대국 스크립트
  - 규칙 기반 상대를 `super_easy` / `easy` / `normal` / `hard`로 골라 둘 수 있다

### 로그 / 재생

- [`log_viewer.py`](/home/yphong/omok_deeplearning/log_viewer.py)
  - Tkinter 기반 로그 리플레이 UI
  - 저장된 게임 로그를 열어서 수순과 판세를 다시 본다
- [`log_parser.py`](/home/yphong/omok_deeplearning/log_parser.py)
  - 학습 로그와 시뮬레이션 로그를 구조화해서 읽는 파서
  - `log_viewer.py`와 후처리 도구가 이 모듈을 공유한다
- [`log_utils.py`](/home/yphong/omok_deeplearning/log_utils.py)
  - 로그 경로 찾기 같은 공용 보조 함수 모음
  - 최신 로그 자동 선택 같은 작업에 쓴다

### 반복 실행 셸

- `run_value_reference_forever.sh`
  - `value_agent` 라인의 배치 실행을 반복하는 쉘 래퍼
- `run_tactical_value_reference_forever.sh`
  - `tactical_value_agent` 라인의 배치 실행을 반복하는 쉘 래퍼
- `run_tactical_rule_value_reference_forever.sh`
  - `tactical_rule_value_agent` 라인의 배치 실행을 반복하는 쉘 래퍼
  - 퇴근 시 무한 반복용으로 주로 사용한다

## 프로젝트 목적

이 문서는 오목 프로젝트의 운영 방식과 스크립트 역할을 설명한다.
학습 전략의 세부 파라미터와 커리큘럼은 다른 문서에서 관리한다.

## 문서 원칙

- 중요한 의사결정, 구현 방향, 운영 규칙은 삭제하지 않는다.
- 내용이 바뀌면 기존 기록은 남기고, 최신 판단을 추가하는 누적 방식으로 정리한다.
- 예전 모델 구현이나 과거 판단은 문서 맨 아래 `히스토리` 문단으로 옮겨 보관한다.
- 잘못된 내용이 있으면 삭제보다 수정과 보강을 우선한다.
- 이 저장소의 `md` 파일은 사용자가 명시적으로 요청할 때만 새로 작성하거나 수정한다.

## 보관 메모

- refer 관리는 [`models/refer/`](/home/yphong/omok_deeplearning/models/refer) 폴더에 파일을 넣고 빼는 방식으로 한다.
- `tactical_rule_value_agent`의 버전은 reference 승격 여부와 별개로 관리한다.
- `forever` launcher는 공용 락 파일을 써서 중복 실행을 막는다.
- 현재 기준값은 위의 최신 운영 스냅샷과 현재 판단을 따른다.

## 테스트 세트

### 1batch 실행 템플릿

- 기본 형식:

```bash
./.venv/bin/python train_value_reference.py --games 1000 --save-every 1000 --pretrain-positions 0 --candidate-prefix tactical_rule_value_agent --candidate-version <N> --candidate-init-model models/tactical_rule_value_agent_v<N-1>.pt --reference-cycle-length 10 --reference-rule-agent-level super_easy --reference-rule-opening-moves 20 --reference-rule-followup-probability 0.10 --reference-rule-only-agent-level super_easy --teacher-rule-agent-level hard --teacher-weight 1.0 --device cpu
```

- 현재 실제 예시:

```bash
./.venv/bin/python train_value_reference.py --games 1000 --save-every 1000 --pretrain-positions 0 --candidate-prefix tactical_rule_value_agent --candidate-version 41 --candidate-init-model models/tactical_rule_value_agent_v40.pt --reference-cycle-length 10 --reference-rule-agent-level super_easy --reference-rule-opening-moves 20 --reference-rule-followup-probability 0.10 --reference-rule-only-agent-level super_easy --teacher-rule-agent-level hard --teacher-weight 1.0 --device cpu
```

### 1batch

- `python3 -m compileall train_value_reference.py agent/torch_cnn_value_agent.py tests/test_gomoku_env.py`
- `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_torch_cnn_value_agent_randomizes_opening_within_central_seven_by_seven`
- `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_rule_only_reference_agent_uses_rule_agent_directly`
- `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_reference_training_can_include_rule_only_reference`
- `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_reference_training_accepts_multiple_references`
- `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_progressive_reference_training_promotes_each_block`
- `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_reference_training_can_include_rule_only_reference tests.test_gomoku_env.GomokuEnvTest.test_reference_training_accepts_multiple_references`

### 무한 반복

- `./run_tactical_rule_value_reference_forever.sh`
- `./run_tactical_value_reference_forever.sh`
- `./run_value_reference_forever.sh`

### 확인용 로그 명령

- `ls -1t logs/*tactical_rule_value_reference_training.log | head -n 1`
- `sed -n '1,20p' logs/<latest_log>.log`
- `ps -ef | rg "train_value_reference.py|run_tactical_rule_value_reference_forever.sh"`
