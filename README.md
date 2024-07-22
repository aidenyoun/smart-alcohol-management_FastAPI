# smart-alcohol-management Project - 인공지능 모델을 활용하기 위한 FastAPI 서버 개발

## 1. POST http://localhost/ai/upload
사용자의 음성 파일을 서버에 업로드,
1번째 POST: 이 엔드포인트는 세션을 초기화하고 이후 요청에 사용해야 하는 Session-ID를 반환

응답
- 성공 (200):

  {
    "file_number": <number>,
    "session_id": "<Session-ID>"
  }

이후 POST : session_id를 사용하여 음성 파일을 POST

## 2. POST http://localhost/ai/predict
부여 받은 Session-ID 를 이용하여 POST

단, 10개 이상의 파일을 업로드 하지 않을시, 오류 JSON 반환

10개 이상의 파일을 업로드 한 후, predict API에 POST 요청
결과에 대략 1.8k ms ~ 2.5k ms
