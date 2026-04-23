# Venv 및 Torch 환경

## 1. 현재 학습 환경

- 프로젝트 루트: `/home/yphong/omok_deeplearning`
- Python 실행 파일: `/home/yphong/omok_deeplearning/.venv/bin/python`
- 가상환경 활성화:

```bash
source /home/yphong/omok_deeplearning/.venv/bin/activate
```

## 2. 확인값

- Torch 버전: `2.11.0+cu130`
- `torch.cuda.is_available()`: `False`
- 현재 기준 학습 장치: CPU

## 3. 확인 명령

```bash
/home/yphong/omok_deeplearning/.venv/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. 현재 자주 쓰는 실행 예시

### tactical value main line

```bash
/home/yphong/omok_deeplearning/.venv/bin/python train_value_reference.py \
  --games 1000 \
  --save-every 1000 \
  --pretrain-positions 0 \
  --candidate-prefix tactical_value_agent \
  --candidate-version 115 \
  --reference-cycle-length 10 \
  --device cpu
```

### tactical rule value 실험

```bash
/home/yphong/omok_deeplearning/.venv/bin/python train_tactical_value.py \
  --games 1000 \
  --save-every 1000 \
  --candidate-prefix tactical_rule_value_agent \
  --candidate-version 5 \
  --candidate-init-model models/tactical_value_agent_v114.pt \
  --rule-agent-level super_easy \
  --teacher-rule-agent-level hard \
  --teacher-weight 0.5 \
  --device cpu
```

## 5. 주의사항

- `/usr/bin/python3`는 사용하지 말 것
- 현재 머신에서 CUDA는 사용 불가
- 모든 학습은 `.venv` 인터프리터 기준으로 실행

## 6. 추가 메모

- Windows 호스트에는 NPU가 보이더라도, 현재 WSL 학습 환경에서는 NPU가 잡히지 않았다.
- 현재 PyTorch는 `torch.npu`와 `torch_npu`를 사용하지 못한다.
- 따라서 현재 학습 장치는 CPU로 본다.
- `forever` launcher를 쓸 때도 `.venv/bin/python` 기준으로만 실행한다.
