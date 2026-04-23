# 오목 학습 인수인계 문서

## 0. 최신 운영 스냅샷

- 기준 시점: `2026-04-16`
- 현재 주력 라인: `tactical_rule_value_agent`
- 최근 완료 버전: `v38`
- 현재 진행 배치: `v39`
- 현재 배치 로그: [`logs/20260416_095540_tactical_rule_value_reference_training.log`](/home/yphong/omok_deeplearning/logs/20260416_095540_tactical_rule_value_reference_training.log)
- 현재 배치 설정:
  - `--games 1000`
  - `--save-every 1000`
  - `--pretrain-positions 0`
  - `--reference-cycle-length 10`
  - `--reference-rule-agent-level super_easy`
  - `--reference-rule-opening-moves 20`
  - `--reference-rule-followup-probability 0.10`
  - `--reference-rule-only-agent-level super_easy`
  - `--teacher-rule-agent-level hard`
  - `--teacher-weight 1.0`
- 첫 수 정책:
  - `TorchCNNValueAgent`는 첫 수를 `top 10`에서 고르지 않고, 중앙 `7x7` 내부 랜덤으로 두도록 바뀌었다.
- 참조 정책:
- 일반 reference는 `super_easy` overlay를 받는다.
- overlay는 opening 20수 이후에도 10% 확률로 rule 판단을 다시 따른다.
- `super_easy` rule-only reference를 추가해서 외곽 오프닝 대응을 직접 강화한다.

## 0-1. 현재 결론

- 외곽 오프닝이 승률을 올리면 보상만으로는 그 패턴이 유지될 수 있다.
- 그래서 첫 수 중앙 제한과 rule-only reference를 같이 넣는 방향으로 정리했다.
- `v39`는 이 변경 전후를 가르는 실험으로 취급한다.

## 0-2. 당장 보는 로그와 명령

- 배치 시작 로그는 `logs/` 아래의 `*_tactical_rule_value_reference_training.log` 파일을 확인한다.
- 실행 명령은 `train_value_reference.py`를 기준으로 본다.
- forever 런처는 `run_tactical_rule_value_reference_forever.sh`다.
- 단발 배치는 `--candidate-version`만 지정하고 1000게임 기준으로 한 번만 돌린다.

## 1. 현재 상태

- 프로젝트 경로: `/home/yphong/omok_deeplearning`
- 실행 환경: `/home/yphong/omok_deeplearning/.venv/bin/python`
- 현재 날짜 기준 기록: `2026-04-14`
- 빠른 재시작용 압축본: [`CONTEXT_SNAPSHOT.md`](/home/yphong/omok_deeplearning/CONTEXT_SNAPSHOT.md)
- 현재 주력 실험선: `tactical_rule_value_agent`
- 현재 진행 배치: `tactical_rule_value_agent_v11`
- 현재 seed 기준점: `models/tactical_rule_value_agent_v10.pt`
- 현재 실행 로그: [`logs/20260414_114512_tactical_rule_value_reference_training.log`](/home/yphong/omok_deeplearning/logs/20260414_114512_tactical_rule_value_reference_training.log)

## 3. 현재 학습 구조

- 우리 쪽 모델:
  - CNN value 기반 `tactical_rule_value_agent`
- teacher:
  - `hard` rule agent 고정
  - `teacher_weight=1.0`
  - candidate의 모든 턴에 적용
- 상대:
  - 현재 남아 있는 refer 전체
  - `value_agent_*_reference`에만 rule overlay 적용
  - 현재 overlay 난이도는 `super_easy`
  - overlay는 초반 `20수` 동안만 적용

## 4. 색상 규칙

- `super_easy` overlay일 때:
  - candidate는 항상 `흑`
- `easy` 이상일 때:
  - 흑백 번갈이

## 5. 현재 코드 반영 상태

### [`train_value_reference.py`](/home/yphong/omok_deeplearning/train_value_reference.py)

- `reference_rule_agent_level`, `reference_rule_opening_moves` 지원
- `teacher_rule_agent_level`, `teacher_weight` 지원
- value refer에만 rule overlay 적용
- candidate 턴마다
  - reference imitation loss
  - teacher loss
  둘 다 반영
- `15수 이내 패배`는 학습률 배율 `3배`

### [`agent/tactical_rule_agent.py`](/home/yphong/omok_deeplearning/agent/tactical_rule_agent.py)

- `super_easy`, `easy`, `normal`, `hard` 지원
- `super_easy`는
  - 즉승/즉방 유지
  - `30수 이후 5%` 확률로 상위 후보 중 랜덤수
  - 강제수 상황에서는 랜덤 미발동

### [`agent/torch_cnn_value_agent.py`](/home/yphong/omok_deeplearning/agent/torch_cnn_value_agent.py)

- 15수 전 랜덤 금지
- 랜덤 탐험은 상위 10개 중 1등 제외
- `[random]`, `[forced]` 로그 표시
- `teacher_board_tensor` 지원
- `15수 이내 패배` 강화 학습 유지

## 6. 최근 중요한 사실

- 어제 무한루프가 샌드박스 밖에서 실제로 살아 있었고, `v99` 체인이 계속 로그를 쓰고 있었다
- `train_value_reference.py ... candidate-version 99`
- `run_tactical_rule_value_reference_forever.sh`
- 위 두 프로세스는 시스템 전체 기준으로 종료 완료
- 이후 `v11`을 단독 재시작함

## 7. 현재 판단

- `easy` overlay는 아직 과한 편으로 확인됨
- 현재는 다시 `super_easy`로 내리고 refer 범위를 전체로 확장한 상태
- 다음 의사결정은 `v11` 결과를 보고
  - refer 범위
  - overlay 난이도
  둘 중 하나만 바꾸는 식으로 진행하는 것이 적절

## 8. 추가 누적 기록

- refer 관리는 이제 [`models/refer/`](/home/yphong/omok_deeplearning/models/refer) 폴더 기준이다.
- refer 추가/삭제는 폴더에 `.pt` 파일을 넣고 빼는 방식으로만 관리한다.
- `tactical_rule_value_agent v10~v30`는 `_reference.pt`로 [`models/refer/`](/home/yphong/omok_deeplearning/models/refer)로 승격되었다.
- `tactical_rule_value_agent v30+`는 오버피팅 후보로 보고 refer에서 제거했다.
- `super_easy` rule overlay는 이제 모든 refer에 적용된다. 예전처럼 `value_agent`만 한정하지 않는다.
- `hard` teacher는 `open three` 차단까지 포함한다. `super_easy/easy/normal`은 `open three`를 강제하지 않는다.
- `forever` launcher는 공용 락 파일을 사용하도록 정리했다.
- WSL 환경에서는 Windows에 NPU가 보여도 이 Linux 환경에서 NPU가 보이지 않을 수 있다. 현재 학습은 CPU 기준이다.

## 9. 테스트 / 디버깅 메모

- 스크립트 문법 확인:
  - `python3 -m compileall train_value_reference.py agent/torch_cnn_value_agent.py tests/test_gomoku_env.py`
- 단위 테스트:
  - `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_torch_cnn_value_agent_randomizes_opening_within_central_seven_by_seven`
  - `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_rule_only_reference_agent_uses_rule_agent_directly`
  - `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_reference_training_can_include_rule_only_reference`
  - `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_reference_training_accepts_multiple_references`
  - `python3 -m unittest tests.test_gomoku_env.GomokuEnvTest.test_progressive_reference_training_promotes_each_block`
- 학습이 중간에 멈춘 것처럼 보이면 먼저 로그 마지막 줄과 `ps`에서 `train_value_reference.py` 생존 여부를 확인한다.
- 중간 체크포인트가 없으면 `--save-every 1000` 설정일 가능성이 높다.

## 10. 오늘 작업 기록

- `v41`부터 `v45`까지 단발 배치를 연속으로 실행했다.
- `v45` 이후에는 `run_tactical_rule_value_reference_forever.sh`를 실행해서 자동 반복으로 전환했다.
- 현재 학습은 `tactical_rule_value_agent` 기준이며, `policy` 계열은 운영 배치에 쓰지 않는다.
- `tactical_rule_agent`는 reference와 teacher 역할을 맡는 규칙 기반 상대다.
- 내일 이어서 볼 때는 최신 로그와 `models/tactical_rule_value_agent_v45.pt` 이후 산출물을 확인하면 된다.
