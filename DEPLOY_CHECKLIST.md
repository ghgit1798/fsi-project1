# Streamlit Deployment Checklist

## 1) App entrypoint
- `app.py`가 배포 엔트리포인트인지 확인

## 2) Dependencies
- `requirements.txt`가 저장소 루트에 있는지 확인
- 배포 후 빌드 로그에서 패키지 설치 실패가 없는지 확인

## 3) Secrets / environment variables
- `.env`는 업로드하지 않고 플랫폼 Secrets 사용
- 필수:
  - `OPENAI_API_KEY`
  - `HF_TOKEN` (GPQA 벤치마크 실행 시 필요)
- 예시는 `.streamlit/secrets.toml.example` 참고

## 4) Runtime behavior checks
- 챗봇 질문/응답이 정상 동작하는지 확인
- 라우팅 결과(`ai_security`, `device_security`, `account_security`, `general_security`)가 출력되는지 확인
- 벤치마크 UI 토글(`> 오픈소스 벤치마크`)이 열리는지 확인

## 5) Benchmark checks
- `MMLU` 실행 확인
- `TruthfulQA` 실행 확인
- `GPQA` 실행 확인 (`HF_TOKEN` 없으면 실패 가능)
- 결과 파일 생성 확인:
  - `benchmarks/results/mmlu_results.json`
  - `benchmarks/results/truthfulqa_results.json`
  - `benchmarks/results/gpqa_results.json`

## 6) Known caveats
- Hugging Face 데이터셋은 첫 실행 시 다운로드로 느릴 수 있음
- 네트워크/레이트리밋 상태에 따라 초기 실행 지연 가능
- GPQA는 gated 접근이므로 토큰 권한 확인 필요
