
- 라즈베리 파이와 같은 임베디드 장치에서 **Chromium, WebKit, WPE** 등의 웹 엔진 기반 OTT 애플리케이션을 실행할 때, 초기 화면이 표시되기까지 긴 지연이 발생하는 문제
- 메모리, CPU 자원이 제한된 임베디드 환경에서는 cold start delay가 두드러짐.
### Cache prefetching 및 preload
- Window 8.1에서 앱이 자주 쓰는 웹 콘텐츠를 사전에 가져와 HTTP 캐시에 저장해두는 **콘텐츠 프리패처(ContentPrefetcher)** 를 제공 [1]
### IORap
- Android 11에 도입된 **IORap** 기능은 앱 실행 시 필요한 라이브러리/리소스의 디스크 읽기를 미리 수행하여 **콜드 스타트 시간을 평균 5% 개선**했고, 특정 앱들에서는 **20% 이상의 속도 향상**도 관찰됨[2]
- 
### Reference
[1] https://blogs.windows.com/windowsdeveloper/2014/05/01/launch-apps-faster-with-prefetched-content/#:~:text=Prefetching%20data%20can%20significantly%20boost,required%20to%20achieve%20similar%20performance 
[2] https://medium.com/androiddevelopers/improving-app-startup-with-i-o-prefetching-62fbdb9c9020#:~:text=In%20Android%2011%2C%20we%20introduced,without%20any%20developer%20app%20changes
[3] https://web.dev/case-studies/hotstar-inp?hl=ko&utm_source=chatgpt.com