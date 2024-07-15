# smart-alcohol-management Project - 인공지능 모델을 활용하기 위한 FastAPI 서버 개발

## 1. POST http://localhost/ai/upload
아래 사진의 Headers를 확인해보시면 Session-ID 가 활성화 되어있지만 빈칸인 것을 확인할 수 있음. 이는 첫 파일 전송일 경우에만 가능함.

정상적으로 POST가 완료된 경우, JSON으로 몇 번째 파일 업로드인지와 함께 Session-ID 를 부여받음.

이후에는 부여 받은 Session-ID 를 사용하여 POST.

## 2. POST http://localhost/ai/predict
부여 받은 Session-ID 를 이용하여 POST 했을 때의 화면

단, 10개 이상의 파일을 업로드 하지 않아 오류 JSON 반환

10개 이상의 파일을 업로드 한 후, predict API에 POST 요청 보냈을 때의 결과.

결과에 대략 1.8k ms ~ 2.5k ms
