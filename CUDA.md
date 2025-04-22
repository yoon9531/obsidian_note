## About CUDA

CUDA(Compute Unified Device Architecture)는 NVIDIA에서 개발한 병렬 컴퓨팅 플랫폼 및 프로그래밍 모델이다. 이는 개발자가 NVIDIA GPU를 일반적인 처리에 사용할 수 있도록 하여 GPU의 대규모 병렬 처리를 활용해 애플리케이션을 최적화할 수 있고, NVIDIA GPU에서만 사용할 수 있다.

## 내 GPU 확인
간단하게 `nvidia-smi` 명령어로 확인할 수 있다.
```
nvidia-smi
```
입력하면 다음과 같이 GPU 모델, 메모리 사용량, 드라이버 버전 등의 정보가 나온다.
![[Pasted image 20250419155320.png]]
## CUDA 개발 환경 세팅하기
- CUDA 프로그램을 작성하고 컴파일 하기 위해서는 전용 컴파일 도구인 CUDA 툴킷이 필요하다.
- https://developer.nvidia.com/cuda-downloads
- 설치한 후 명령 프롬프트에서 `nvcc --version` 을 입력해 설치가 정상적으로 완료됐음을 확인할 수 있다.
	 ![[Pasted image 20250419155545.png]]
- **Trouble shooting**
	- `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`
		- `nvcc` 의 호스트 컴파일러로 MSVC `cl.exe`를 못 찾음
		- 시스템 환경 변수 설정에서 Path 변수에 MSVC bin 폴더 저장
		`C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64`
	- `cannot open source file "cuda_runtime.h"`
		- `Ctrl+Shift+P` $\rightarrow$ `C/C++ : Edit Configurations(JSON)` 클릭
		- `.vscode/c_cpp_properties.json` 에 CUDA `include` 폴더 추가
			```
			"includePath": [
				"${workspaceFolder}/**",
				"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include"
			],
			```
	- warning C4819: The file contains a character that cannot be represented in the current code page (949). Save the file in Unicode format to prevent data loss
		- 외부 헤더 유니코드 문자를 처리 못함
		- `Ctrl + ,` $\rightarrow$ Encoding 검색 $\rightarrow$ `Files:Encoding`을 `UTF-8`에서 `UTF-8 with BOM`로 변경
		- Vscode 윈도우 하단에 `UTF-8` 클릭 $\rightarrow$ `Save with Encoding` $\rightarrow$ `UTF-8 with BOM`로 변경
	- 